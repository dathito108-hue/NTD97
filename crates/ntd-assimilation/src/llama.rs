#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use ntd_capsule::{
    encode_native_tensor, encode_tensor_descriptors, NativeTensor, QuantizationMetadata,
    SectionKind, TensorDescriptor,
};
use ntd_ir::{
    DType, Graph, IrVersion, Node, NodeId, OpKind, TensorOp, ValueDecl, ValueId, ValueType,
};

use crate::{GgufError, GgufModel, GgufTensorInfo, GgufValue, NativeSection};

#[derive(Debug, Clone, PartialEq)]
pub struct LlamaConfig {
    pub context_length: u32,
    pub embedding_length: u32,
    pub block_count: u32,
    pub feed_forward_length: u32,
    pub head_count: u32,
    pub head_count_kv: u32,
    pub head_dimension: u32,
    pub rope_dimension_count: u32,
    pub rope_frequency_base: f32,
    pub rms_epsilon: f32,
    pub vocabulary_size: u32,
}

impl LlamaConfig {
    pub fn from_gguf(model: &GgufModel) -> Result<Self, LlamaLoweringError> {
        if model.architecture().map_err(LlamaLoweringError::Gguf)? != "llama" {
            return Err(LlamaLoweringError::UnsupportedArchitecture(
                model
                    .architecture()
                    .map_err(LlamaLoweringError::Gguf)?
                    .to_owned(),
            ));
        }

        let context_length = metadata_u32(model, "llama.context_length")?;
        let embedding_length = metadata_u32(model, "llama.embedding_length")?;
        let block_count = metadata_u32(model, "llama.block_count")?;
        let feed_forward_length = metadata_u32(model, "llama.feed_forward_length")?;
        let head_count = metadata_u32(model, "llama.attention.head_count")?;
        let head_count_kv = model
            .metadata
            .get("llama.attention.head_count_kv")
            .map(|value| {
                value.as_u32().ok_or(LlamaLoweringError::InvalidMetadata(
                    "llama.attention.head_count_kv",
                ))
            })
            .transpose()?
            .unwrap_or(head_count);

        if context_length == 0
            || embedding_length == 0
            || block_count == 0
            || feed_forward_length == 0
            || head_count == 0
            || head_count_kv == 0
            || embedding_length % head_count != 0
            || head_count % head_count_kv != 0
        {
            return Err(LlamaLoweringError::InvalidArchitecture);
        }

        let head_dimension = embedding_length / head_count;
        let rope_dimension_count = model
            .metadata
            .get("llama.rope.dimension_count")
            .map(|value| {
                value.as_u32().ok_or(LlamaLoweringError::InvalidMetadata(
                    "llama.rope.dimension_count",
                ))
            })
            .transpose()?
            .unwrap_or(head_dimension);
        if rope_dimension_count != head_dimension || rope_dimension_count % 2 != 0 {
            return Err(LlamaLoweringError::UnsupportedFeature(
                "partial or odd rotary dimensions are not yet canonical".into(),
            ));
        }

        for (key, expected) in [
            ("llama.attention.key_length", head_dimension),
            ("llama.attention.value_length", head_dimension),
        ] {
            if let Some(actual) = model.metadata.get(key).and_then(GgufValue::as_u32) {
                if actual != expected {
                    return Err(LlamaLoweringError::UnsupportedFeature(format!(
                        "{key}={actual} differs from canonical head dimension {expected}"
                    )));
                }
            }
        }

        if let Some(kind) = model
            .metadata
            .get("llama.rope.scaling.type")
            .and_then(GgufValue::as_string)
        {
            if kind != "none" {
                return Err(LlamaLoweringError::UnsupportedFeature(format!(
                    "rope scaling '{kind}' requires a dedicated NTD97 semantic"
                )));
            }
        }

        let rope_frequency_base = model
            .metadata
            .get("llama.rope.freq_base")
            .map(|value| {
                value.as_f32().ok_or(LlamaLoweringError::InvalidMetadata(
                    "llama.rope.freq_base",
                ))
            })
            .transpose()?
            .unwrap_or(10_000.0);
        let rms_epsilon = metadata_f32(model, "llama.attention.layer_norm_rms_epsilon")?;
        if !rope_frequency_base.is_finite()
            || rope_frequency_base <= 1.0
            || !rms_epsilon.is_finite()
            || rms_epsilon <= 0.0
        {
            return Err(LlamaLoweringError::InvalidArchitecture);
        }

        let tokenizer = model.tokenizer().map_err(LlamaLoweringError::Gguf)?;
        let vocabulary_size =
            u32::try_from(tokenizer.tokens.len()).map_err(|_| LlamaLoweringError::Overflow)?;
        if let Some(declared) = model.metadata.get("llama.vocab_size").and_then(GgufValue::as_u32)
        {
            if declared != vocabulary_size {
                return Err(LlamaLoweringError::InvalidArchitecture);
            }
        }

        Ok(Self {
            context_length,
            embedding_length,
            block_count,
            feed_forward_length,
            head_count,
            head_count_kv,
            head_dimension,
            rope_dimension_count,
            rope_frequency_base,
            rms_epsilon,
            vocabulary_size,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlamaWeightBinding {
    pub name: String,
    pub tensor_id: u32,
    pub graph_value: ValueId,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LlamaNativeDraft {
    pub config: LlamaConfig,
    pub graph: Graph,
    pub tensors: Vec<NativeTensor>,
    pub weight_bindings: Vec<LlamaWeightBinding>,
    pub token_input: ValueId,
    pub logits_output: ValueId,
    pub tied_output: bool,
}

impl LlamaNativeDraft {
    pub fn tensor_sections(&self) -> Result<Vec<NativeSection>, LlamaLoweringError> {
        let descriptors = self
            .tensors
            .iter()
            .map(|tensor| tensor.descriptor.clone())
            .collect::<Vec<_>>();
        let mut sections = vec![NativeSection {
            kind: SectionKind::Tensors,
            bytes: encode_tensor_descriptors(&descriptors)
                .map_err(|error| LlamaLoweringError::Capsule(format!("{error:?}")))?,
        }];
        for tensor in &self.tensors {
            sections.push(NativeSection {
                kind: SectionKind::Tensors,
                bytes: encode_native_tensor(tensor)
                    .map_err(|error| LlamaLoweringError::Capsule(format!("{error:?}")))?,
            });
        }
        Ok(sections)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LlamaLoweringError {
    Gguf(GgufError),
    UnsupportedArchitecture(String),
    MissingMetadata(&'static str),
    InvalidMetadata(&'static str),
    InvalidArchitecture,
    MissingTensor(String),
    DuplicateTensor(String),
    InvalidTensorShape(String),
    UnsupportedTensorType { name: String, ggml_type: u32 },
    UnsupportedFeature(String),
    TensorDataOutOfBounds(String),
    Capsule(String),
    Overflow,
}

pub fn lower_llama_f32_f16(
    model: &GgufModel,
    source_bytes: &[u8],
) -> Result<LlamaNativeDraft, LlamaLoweringError> {
    let config = LlamaConfig::from_gguf(model)?;
    let tensors_by_name = tensor_map(model)?;
    reject_unsupported_biases(&tensors_by_name)?;

    let mut required = Vec::new();
    push_required(
        &mut required,
        "token_embd.weight",
        vec![
            u64::from(config.embedding_length),
            u64::from(config.vocabulary_size),
        ],
    );
    push_required(
        &mut required,
        "output_norm.weight",
        vec![u64::from(config.embedding_length)],
    );

    let tied_output = !tensors_by_name.contains_key("output.weight");
    if !tied_output {
        push_required(
            &mut required,
            "output.weight",
            vec![
                u64::from(config.embedding_length),
                u64::from(config.vocabulary_size),
            ],
        );
    }

    let q_width = u64::from(config.head_count) * u64::from(config.head_dimension);
    let kv_width = u64::from(config.head_count_kv) * u64::from(config.head_dimension);
    for layer in 0..config.block_count {
        let prefix = format!("blk.{layer}");
        push_required(
            &mut required,
            format!("{prefix}.attn_norm.weight"),
            vec![u64::from(config.embedding_length)],
        );
        push_required(
            &mut required,
            format!("{prefix}.attn_q.weight"),
            vec![u64::from(config.embedding_length), q_width],
        );
        push_required(
            &mut required,
            format!("{prefix}.attn_k.weight"),
            vec![u64::from(config.embedding_length), kv_width],
        );
        push_required(
            &mut required,
            format!("{prefix}.attn_v.weight"),
            vec![u64::from(config.embedding_length), kv_width],
        );
        push_required(
            &mut required,
            format!("{prefix}.attn_output.weight"),
            vec![q_width, u64::from(config.embedding_length)],
        );
        push_required(
            &mut required,
            format!("{prefix}.ffn_norm.weight"),
            vec![u64::from(config.embedding_length)],
        );
        push_required(
            &mut required,
            format!("{prefix}.ffn_gate.weight"),
            vec![
                u64::from(config.embedding_length),
                u64::from(config.feed_forward_length),
            ],
        );
        push_required(
            &mut required,
            format!("{prefix}.ffn_up.weight"),
            vec![
                u64::from(config.embedding_length),
                u64::from(config.feed_forward_length),
            ],
        );
        push_required(
            &mut required,
            format!("{prefix}.ffn_down.weight"),
            vec![
                u64::from(config.feed_forward_length),
                u64::from(config.embedding_length),
            ],
        );
    }

    for (name, dimensions) in &required {
        let info = tensors_by_name
            .get(name.as_str())
            .ok_or_else(|| LlamaLoweringError::MissingTensor(name.clone()))?;
        if &info.dimensions != dimensions {
            return Err(LlamaLoweringError::InvalidTensorShape(name.clone()));
        }
        if !matches!(info.ggml_type, 0 | 1) {
            return Err(LlamaLoweringError::UnsupportedTensorType {
                name: name.clone(),
                ggml_type: info.ggml_type,
            });
        }
    }

    let token_input = ValueId(0);
    let mut inputs = vec![tensor_decl(token_input, DType::I32, 1)];
    let mut tensors = Vec::new();
    let mut bindings = Vec::new();
    let mut values = BTreeMap::new();
    let mut next_value = 1u32;
    let mut next_tensor = 1u32;

    for (name, _) in &required {
        let info = tensors_by_name
            .get(name.as_str())
            .ok_or_else(|| LlamaLoweringError::MissingTensor(name.clone()))?;
        let graph_value = ValueId(next_value);
        next_value = next_value.checked_add(1).ok_or(LlamaLoweringError::Overflow)?;
        let native = materialize_direct_tensor(
            model,
            source_bytes,
            info,
            next_tensor,
            graph_value,
        )?;
        next_tensor = next_tensor
            .checked_add(1)
            .ok_or(LlamaLoweringError::Overflow)?;
        inputs.push(tensor_decl(
            graph_value,
            native.descriptor.dtype,
            u8::try_from(native.descriptor.shape.len()).map_err(|_| LlamaLoweringError::Overflow)?,
        ));
        values.insert(name.clone(), graph_value);
        bindings.push(LlamaWeightBinding {
            name: name.clone(),
            tensor_id: native.descriptor.id,
            graph_value,
        });
        tensors.push(native);
    }

    let epsilon = push_constant(
        &mut inputs,
        &mut tensors,
        &mut next_value,
        &mut next_tensor,
        Vec::new(),
        vec![config.rms_epsilon],
    )?;
    let rope_base = push_constant(
        &mut inputs,
        &mut tensors,
        &mut next_value,
        &mut next_tensor,
        Vec::new(),
        vec![config.rope_frequency_base],
    )?;
    let q_shape = push_constant(
        &mut inputs,
        &mut tensors,
        &mut next_value,
        &mut next_tensor,
        vec![3],
        vec![
            -1.0,
            config.head_count as f32,
            config.head_dimension as f32,
        ],
    )?;
    let kv_shape = push_constant(
        &mut inputs,
        &mut tensors,
        &mut next_value,
        &mut next_tensor,
        vec![3],
        vec![
            -1.0,
            config.head_count_kv as f32,
            config.head_dimension as f32,
        ],
    )?;
    let flat_shape = push_constant(
        &mut inputs,
        &mut tensors,
        &mut next_value,
        &mut next_tensor,
        vec![2],
        vec![-1.0, config.embedding_length as f32],
    )?;

    let mut builder = GraphBuilder::new(next_value);
    let positions = builder.tensor_node(TensorOp::PositionIds, vec![token_input], 1)?;
    let embedding = value(&values, "token_embd.weight")?;
    let mut hidden = builder.tensor_node(TensorOp::Gather, vec![embedding, token_input], 2)?;

    for layer in 0..config.block_count {
        let prefix = format!("blk.{layer}");
        let attn_norm = builder.tensor_node(
            TensorOp::RmsNorm,
            vec![
                hidden,
                value(&values, &format!("{prefix}.attn_norm.weight"))?,
                epsilon,
            ],
            2,
        )?;
        let q = builder.tensor_node(
            TensorOp::Linear,
            vec![
                attn_norm,
                value(&values, &format!("{prefix}.attn_q.weight"))?,
            ],
            2,
        )?;
        let k = builder.tensor_node(
            TensorOp::Linear,
            vec![
                attn_norm,
                value(&values, &format!("{prefix}.attn_k.weight"))?,
            ],
            2,
        )?;
        let v = builder.tensor_node(
            TensorOp::Linear,
            vec![
                attn_norm,
                value(&values, &format!("{prefix}.attn_v.weight"))?,
            ],
            2,
        )?;
        let q = builder.tensor_node(TensorOp::Reshape, vec![q, q_shape], 3)?;
        let k = builder.tensor_node(TensorOp::Reshape, vec![k, kv_shape], 3)?;
        let v = builder.tensor_node(TensorOp::Reshape, vec![v, kv_shape], 3)?;
        let q = builder.tensor_node(
            TensorOp::RotaryPosition,
            vec![q, positions, rope_base],
            3,
        )?;
        let k = builder.tensor_node(
            TensorOp::RotaryPosition,
            vec![k, positions, rope_base],
            3,
        )?;
        let attention =
            builder.tensor_node(TensorOp::CausalAttention, vec![q, k, v], 3)?;
        let attention =
            builder.tensor_node(TensorOp::Reshape, vec![attention, flat_shape], 2)?;
        let attention = builder.tensor_node(
            TensorOp::Linear,
            vec![
                attention,
                value(&values, &format!("{prefix}.attn_output.weight"))?,
            ],
            2,
        )?;
        let residual = builder.tensor_node(TensorOp::Add, vec![hidden, attention], 2)?;

        let ffn_norm = builder.tensor_node(
            TensorOp::RmsNorm,
            vec![
                residual,
                value(&values, &format!("{prefix}.ffn_norm.weight"))?,
                epsilon,
            ],
            2,
        )?;
        let gate = builder.tensor_node(
            TensorOp::Linear,
            vec![
                ffn_norm,
                value(&values, &format!("{prefix}.ffn_gate.weight"))?,
            ],
            2,
        )?;
        let gate = builder.tensor_node(TensorOp::Silu, vec![gate], 2)?;
        let up = builder.tensor_node(
            TensorOp::Linear,
            vec![
                ffn_norm,
                value(&values, &format!("{prefix}.ffn_up.weight"))?,
            ],
            2,
        )?;
        let gated = builder.tensor_node(TensorOp::Mul, vec![gate, up], 2)?;
        let down = builder.tensor_node(
            TensorOp::Linear,
            vec![
                gated,
                value(&values, &format!("{prefix}.ffn_down.weight"))?,
            ],
            2,
        )?;
        hidden = builder.tensor_node(TensorOp::Add, vec![residual, down], 2)?;
    }

    let final_norm = builder.tensor_node(
        TensorOp::RmsNorm,
        vec![hidden, value(&values, "output_norm.weight")?, epsilon],
        2,
    )?;
    let output_weight = if tied_output {
        embedding
    } else {
        value(&values, "output.weight")?
    };
    let logits = builder.tensor_node(TensorOp::Linear, vec![final_norm, output_weight], 2)?;

    let graph = Graph {
        version: IrVersion::CURRENT,
        inputs,
        outputs: vec![logits],
        nodes: builder.nodes,
    };
    graph
        .validate()
        .map_err(|error| LlamaLoweringError::Capsule(format!("{error:?}")))?;

    Ok(LlamaNativeDraft {
        config,
        graph,
        tensors,
        weight_bindings: bindings,
        token_input,
        logits_output: logits,
        tied_output,
    })
}

fn tensor_map<'a>(
    model: &'a GgufModel,
) -> Result<BTreeMap<&'a str, &'a GgufTensorInfo>, LlamaLoweringError> {
    let mut map = BTreeMap::new();
    for tensor in &model.tensors {
        if map.insert(tensor.name.as_str(), tensor).is_some() {
            return Err(LlamaLoweringError::DuplicateTensor(tensor.name.clone()));
        }
    }
    Ok(map)
}

fn reject_unsupported_biases(
    tensors: &BTreeMap<&str, &GgufTensorInfo>,
) -> Result<(), LlamaLoweringError> {
    let unsupported = tensors.keys().find(|name| {
        name.ends_with(".bias")
            && (name.contains(".attn_")
                || name.contains(".ffn_")
                || **name == "output.bias")
    });
    if let Some(name) = unsupported {
        return Err(LlamaLoweringError::UnsupportedFeature(format!(
            "bias tensor '{name}' is not part of the canonical Llama lowering"
        )));
    }
    Ok(())
}

fn materialize_direct_tensor(
    model: &GgufModel,
    source_bytes: &[u8],
    info: &GgufTensorInfo,
    tensor_id: u32,
    graph_value: ValueId,
) -> Result<NativeTensor, LlamaLoweringError> {
    let (dtype, width) = match info.ggml_type {
        0 => (DType::F32, 4u64),
        1 => (DType::F16, 2u64),
        other => {
            return Err(LlamaLoweringError::UnsupportedTensorType {
                name: info.name.clone(),
                ggml_type: other,
            })
        }
    };
    let elements = info.dimensions.iter().try_fold(1u64, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or(LlamaLoweringError::Overflow)
    })?;
    let byte_len = elements
        .checked_mul(width)
        .ok_or(LlamaLoweringError::Overflow)?;
    let start = model
        .data_offset
        .checked_add(info.data_offset)
        .ok_or(LlamaLoweringError::Overflow)?;
    let end = start
        .checked_add(byte_len)
        .ok_or(LlamaLoweringError::Overflow)?;
    let payload = source_bytes
        .get(
            usize::try_from(start).map_err(|_| LlamaLoweringError::Overflow)?
                ..usize::try_from(end).map_err(|_| LlamaLoweringError::Overflow)?,
        )
        .ok_or_else(|| LlamaLoweringError::TensorDataOutOfBounds(info.name.clone()))?
        .to_vec();

    let shape = info.dimensions.iter().rev().copied().collect::<Vec<_>>();
    Ok(NativeTensor {
        descriptor: TensorDescriptor {
            id: tensor_id,
            dtype,
            shape,
            byte_len,
        },
        graph_value: Some(graph_value),
        quantization: QuantizationMetadata::None,
        payload,
    })
}

fn push_constant(
    inputs: &mut Vec<ValueDecl>,
    tensors: &mut Vec<NativeTensor>,
    next_value: &mut u32,
    next_tensor: &mut u32,
    shape: Vec<u64>,
    values: Vec<f32>,
) -> Result<ValueId, LlamaLoweringError> {
    let graph_value = ValueId(*next_value);
    *next_value = next_value
        .checked_add(1)
        .ok_or(LlamaLoweringError::Overflow)?;
    let tensor_id = *next_tensor;
    *next_tensor = next_tensor
        .checked_add(1)
        .ok_or(LlamaLoweringError::Overflow)?;

    let rank = u8::try_from(shape.len()).map_err(|_| LlamaLoweringError::Overflow)?;
    let payload = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let byte_len = u64::try_from(payload.len()).map_err(|_| LlamaLoweringError::Overflow)?;
    inputs.push(tensor_decl(graph_value, DType::F32, rank));
    tensors.push(NativeTensor {
        descriptor: TensorDescriptor {
            id: tensor_id,
            dtype: DType::F32,
            shape,
            byte_len,
        },
        graph_value: Some(graph_value),
        quantization: QuantizationMetadata::None,
        payload,
    });
    Ok(graph_value)
}

fn metadata_u32(model: &GgufModel, key: &'static str) -> Result<u32, LlamaLoweringError> {
    model
        .metadata
        .get(key)
        .ok_or(LlamaLoweringError::MissingMetadata(key))?
        .as_u32()
        .ok_or(LlamaLoweringError::InvalidMetadata(key))
}

fn metadata_f32(model: &GgufModel, key: &'static str) -> Result<f32, LlamaLoweringError> {
    model
        .metadata
        .get(key)
        .ok_or(LlamaLoweringError::MissingMetadata(key))?
        .as_f32()
        .ok_or(LlamaLoweringError::InvalidMetadata(key))
}

fn value(values: &BTreeMap<String, ValueId>, name: &str) -> Result<ValueId, LlamaLoweringError> {
    values
        .get(name)
        .copied()
        .ok_or_else(|| LlamaLoweringError::MissingTensor(name.to_owned()))
}

fn push_required(required: &mut Vec<(String, Vec<u64>)>, name: impl Into<String>, dims: Vec<u64>) {
    required.push((name.into(), dims));
}

fn tensor_decl(id: ValueId, dtype: DType, rank: u8) -> ValueDecl {
    ValueDecl {
        id,
        ty: ValueType::Tensor { dtype, rank },
    }
}

struct GraphBuilder {
    next_value: u32,
    next_node: u32,
    nodes: Vec<Node>,
}

impl GraphBuilder {
    fn new(next_value: u32) -> Self {
        Self {
            next_value,
            next_node: 0,
            nodes: Vec::new(),
        }
    }

    fn tensor_node(
        &mut self,
        op: TensorOp,
        inputs: Vec<ValueId>,
        rank: u8,
    ) -> Result<ValueId, LlamaLoweringError> {
        let output = ValueId(self.next_value);
        self.next_value = self
            .next_value
            .checked_add(1)
            .ok_or(LlamaLoweringError::Overflow)?;
        let node = Node {
            id: NodeId(self.next_node),
            op: OpKind::Tensor(op),
            inputs,
            outputs: vec![tensor_decl(output, DType::F32, rank)],
        };
        self.next_node = self
            .next_node
            .checked_add(1)
            .ok_or(LlamaLoweringError::Overflow)?;
        self.nodes.push(node);
        Ok(output)
    }
}
