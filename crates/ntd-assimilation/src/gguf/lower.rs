#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use ntd_capsule::{
    encode_native_tensor, encode_native_tokenizer, encode_tensor_descriptors, NativeTensor,
    NativeTokenizerDescriptor, NativeTokenizerModel, QuantizationMetadata, SectionKind,
    TensorDescriptor,
};
use ntd_ir::{
    DType, Graph, IrVersion, Node, NodeId, OpKind, TensorOp, ValueDecl, ValueId, ValueType,
};

use super::{llama_spm_blocker, transcode_tensor, GgufError, GgufModel, GgufTensorInfo, GgufValue};
use crate::{NativeCandidate, NativeSection, RegressionCase, RegressionProbe};

#[derive(Debug, Clone, PartialEq)]
pub struct LlamaConfig {
    pub context_length: u32,
    pub embedding_length: u32,
    pub block_count: u32,
    pub feed_forward_length: u32,
    pub head_count: u32,
    pub head_count_kv: u32,
    pub head_dim: u32,
    pub rms_epsilon: f32,
    pub rope_dimension_count: u32,
    pub rope_freq_base: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlamaTensorBinding {
    pub source_name: String,
    pub tensor_id: u32,
    pub graph_value: ValueId,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoweredLlamaModel {
    pub config: LlamaConfig,
    pub graph: Graph,
    pub tensors: Vec<NativeTensor>,
    pub tokenizer: NativeTokenizerDescriptor,
    pub token_input: ValueId,
    pub distribution_output: usize,
    pub vocabulary_size: usize,
    pub bindings: Vec<LlamaTensorBinding>,
}

pub fn lower_llama_model(file: &[u8], model: &GgufModel) -> Result<LoweredLlamaModel, GgufError> {
    if model.architecture()? != "llama" {
        return Err(GgufError::UnsupportedArchitecture(
            model.architecture()?.to_owned(),
        ));
    }

    let config = LlamaConfig::from_model(model)?;
    let source_tokenizer = model.tokenizer()?;
    if source_tokenizer.model != "llama" {
        return Err(GgufError::UnsupportedModelFeature(format!(
            "tokenizer model '{}' requires a native source-equivalent tokenizer implementation",
            source_tokenizer.model
        )));
    }
    if let Some(blocker) = llama_spm_blocker(&source_tokenizer) {
        return Err(GgufError::UnsupportedModelFeature(blocker));
    }

    let score_bits = source_tokenizer
        .scores
        .ok_or_else(|| GgufError::UnsupportedModelFeature("missing LLaMA tokenizer scores".into()))?
        .into_iter()
        .map(f32::to_bits)
        .collect::<Vec<_>>();
    let token_types = source_tokenizer.token_types.ok_or_else(|| {
        GgufError::UnsupportedModelFeature("missing LLaMA tokenizer token types".into())
    })?;
    let add_space_prefix = source_tokenizer.add_space_prefix.ok_or_else(|| {
        GgufError::UnsupportedModelFeature("missing LLaMA add-space-prefix policy".into())
    })?;
    let add_bos_token = source_tokenizer
        .add_bos_token
        .ok_or_else(|| GgufError::UnsupportedModelFeature("missing LLaMA add-BOS policy".into()))?;
    let add_eos_token = source_tokenizer
        .add_eos_token
        .ok_or_else(|| GgufError::UnsupportedModelFeature("missing LLaMA add-EOS policy".into()))?;

    let tokenizer = NativeTokenizerDescriptor {
        tokens: source_tokenizer.tokens,
        bos_token: source_tokenizer.bos_token,
        eos_token: source_tokenizer.eos_token,
        unknown_token: source_tokenizer.unknown_token,
        model: NativeTokenizerModel::LlamaSpm {
            score_bits,
            token_types,
            add_space_prefix,
            add_bos_token,
            add_eos_token,
        },
    };
    encode_native_tokenizer(&tokenizer)
        .map_err(|error| GgufError::NativeLowering(format!("tokenizer: {error:?}")))?;
    let vocabulary_size = tokenizer.tokens.len();
    let vocab_u64 = u64::try_from(vocabulary_size).map_err(|_| GgufError::LimitExceeded)?;

    let source_tensors = SourceTensorTable::new(file, model);

    let mut consumed = BTreeSet::new();
    let mut builder = GraphBuilder::new();

    let token_input = builder.add_external_input(DType::I32, 1)?;
    let positions = builder.add_node(TensorOp::PositionIds, vec![token_input], DType::I32, 1)?;

    let epsilon = builder.add_constant_f32_scalar(config.rms_epsilon)?;
    let query_shape = builder.add_constant_f32_vector(&[
        0.0,
        config.head_count as f32,
        config.head_dim as f32,
    ])?;
    let kv_shape = builder.add_constant_f32_vector(&[
        0.0,
        config.head_count_kv as f32,
        config.head_dim as f32,
    ])?;
    let flatten_shape = builder.add_constant_f32_vector(&[0.0, config.embedding_length as f32])?;

    let token_embedding = source_tensors.add(
        &mut builder,
        &mut consumed,
        "token_embd.weight",
        &[u64::from(config.embedding_length), vocab_u64],
        false,
    )?;
    let mut hidden = builder.add_node(
        TensorOp::Gather,
        vec![token_embedding, token_input],
        DType::F32,
        2,
    )?;

    let kv_width = config
        .head_dim
        .checked_mul(config.head_count_kv)
        .ok_or(GgufError::Overflow)?;

    for layer in 0..config.block_count {
        let attn_norm = source_tensors.add(
            &mut builder,
            &mut consumed,
            &format!("blk.{layer}.attn_norm.weight"),
            &[u64::from(config.embedding_length)],
            false,
        )?;
        let q_weight = source_tensors.add(
            &mut builder,
            &mut consumed,
            &format!("blk.{layer}.attn_q.weight"),
            &[
                u64::from(config.embedding_length),
                u64::from(config.embedding_length),
            ],
            true,
        )?;
        let k_weight = source_tensors.add(
            &mut builder,
            &mut consumed,
            &format!("blk.{layer}.attn_k.weight"),
            &[u64::from(config.embedding_length), u64::from(kv_width)],
            true,
        )?;
        let v_weight = source_tensors.add(
            &mut builder,
            &mut consumed,
            &format!("blk.{layer}.attn_v.weight"),
            &[u64::from(config.embedding_length), u64::from(kv_width)],
            true,
        )?;
        let attn_output = source_tensors.add(
            &mut builder,
            &mut consumed,
            &format!("blk.{layer}.attn_output.weight"),
            &[
                u64::from(config.embedding_length),
                u64::from(config.embedding_length),
            ],
            true,
        )?;
        let ffn_norm = source_tensors.add(
            &mut builder,
            &mut consumed,
            &format!("blk.{layer}.ffn_norm.weight"),
            &[u64::from(config.embedding_length)],
            false,
        )?;
        let ffn_gate = source_tensors.add(
            &mut builder,
            &mut consumed,
            &format!("blk.{layer}.ffn_gate.weight"),
            &[
                u64::from(config.embedding_length),
                u64::from(config.feed_forward_length),
            ],
            true,
        )?;
        let ffn_up = source_tensors.add(
            &mut builder,
            &mut consumed,
            &format!("blk.{layer}.ffn_up.weight"),
            &[
                u64::from(config.embedding_length),
                u64::from(config.feed_forward_length),
            ],
            true,
        )?;
        let ffn_down = source_tensors.add(
            &mut builder,
            &mut consumed,
            &format!("blk.{layer}.ffn_down.weight"),
            &[
                u64::from(config.feed_forward_length),
                u64::from(config.embedding_length),
            ],
            true,
        )?;

        let normed = builder.add_node(
            TensorOp::RmsNorm,
            vec![hidden, attn_norm, epsilon],
            DType::F32,
            2,
        )?;
        let q = builder.add_node(TensorOp::MatMul, vec![normed, q_weight], DType::F32, 2)?;
        let k = builder.add_node(TensorOp::MatMul, vec![normed, k_weight], DType::F32, 2)?;
        let v = builder.add_node(TensorOp::MatMul, vec![normed, v_weight], DType::F32, 2)?;

        let q = builder.add_node(TensorOp::Reshape, vec![q, query_shape], DType::F32, 3)?;
        let k = builder.add_node(TensorOp::Reshape, vec![k, kv_shape], DType::F32, 3)?;
        let v = builder.add_node(TensorOp::Reshape, vec![v, kv_shape], DType::F32, 3)?;
        let q = builder.add_node(TensorOp::RotaryPosition, vec![q, positions], DType::F32, 3)?;
        let k = builder.add_node(TensorOp::RotaryPosition, vec![k, positions], DType::F32, 3)?;
        let attention =
            builder.add_node(TensorOp::CausalAttention, vec![q, k, v], DType::F32, 3)?;
        let attention = builder.add_node(
            TensorOp::Reshape,
            vec![attention, flatten_shape],
            DType::F32,
            2,
        )?;
        let attention = builder.add_node(
            TensorOp::MatMul,
            vec![attention, attn_output],
            DType::F32,
            2,
        )?;
        let residual = builder.add_node(TensorOp::Add, vec![hidden, attention], DType::F32, 2)?;

        let ffn_input = builder.add_node(
            TensorOp::RmsNorm,
            vec![residual, ffn_norm, epsilon],
            DType::F32,
            2,
        )?;
        let gate = builder.add_node(TensorOp::MatMul, vec![ffn_input, ffn_gate], DType::F32, 2)?;
        let gate = builder.add_node(TensorOp::Silu, vec![gate], DType::F32, 2)?;
        let up = builder.add_node(TensorOp::MatMul, vec![ffn_input, ffn_up], DType::F32, 2)?;
        let gated = builder.add_node(TensorOp::Mul, vec![gate, up], DType::F32, 2)?;
        let down = builder.add_node(TensorOp::MatMul, vec![gated, ffn_down], DType::F32, 2)?;
        hidden = builder.add_node(TensorOp::Add, vec![residual, down], DType::F32, 2)?;
    }

    let output_norm = source_tensors.add(
        &mut builder,
        &mut consumed,
        "output_norm.weight",
        &[u64::from(config.embedding_length)],
        false,
    )?;
    let hidden = builder.add_node(
        TensorOp::RmsNorm,
        vec![hidden, output_norm, epsilon],
        DType::F32,
        2,
    )?;

    let output_source = source_tensors.tensors.get("output.weight").copied();
    let output = match output_source {
        Some(_) => source_tensors.add(
            &mut builder,
            &mut consumed,
            "output.weight",
            &[u64::from(config.embedding_length), vocab_u64],
            true,
        )?,
        None => source_tensors.add(
            &mut builder,
            &mut consumed,
            "token_embd.weight",
            &[u64::from(config.embedding_length), vocab_u64],
            true,
        )?,
    };
    let logits = builder.add_node(TensorOp::MatMul, vec![hidden, output], DType::F32, 2)?;

    reject_unconsumed_model_tensors(model, &consumed)?;

    let graph = builder.finish(vec![logits])?;
    Ok(LoweredLlamaModel {
        config,
        graph,
        tensors: builder.tensors,
        tokenizer,
        token_input,
        distribution_output: 0,
        vocabulary_size,
        bindings: builder.bindings,
    })
}

pub fn lowered_llama_candidate(
    asset_id: impl Into<String>,
    version: u32,
    model: &LoweredLlamaModel,
) -> Result<NativeCandidate, GgufError> {
    let asset_id = asset_id.into().trim().to_owned();
    if asset_id.is_empty() || version == 0 {
        return Err(GgufError::NativeLowering(
            "invalid native model identity".into(),
        ));
    }

    let descriptors = model
        .tensors
        .iter()
        .map(|tensor| tensor.descriptor.clone())
        .collect::<Vec<_>>();
    let mut sections = Vec::with_capacity(model.tensors.len().saturating_add(2));
    sections.push(NativeSection {
        kind: SectionKind::Tensors,
        bytes: encode_tensor_descriptors(&descriptors)
            .map_err(|error| GgufError::NativeLowering(format!("descriptors: {error:?}")))?,
    });
    for tensor in &model.tensors {
        sections.push(NativeSection {
            kind: SectionKind::Tensors,
            bytes: encode_native_tensor(tensor)
                .map_err(|error| GgufError::NativeLowering(format!("tensor: {error:?}")))?,
        });
    }
    sections.push(NativeSection {
        kind: SectionKind::Tokenizer,
        bytes: encode_native_tokenizer(&model.tokenizer)
            .map_err(|error| GgufError::NativeLowering(format!("tokenizer: {error:?}")))?,
    });

    let candidate = NativeCandidate::Intelligence {
        asset_id,
        version,
        graph: model.graph.clone(),
        sections,
        regressions: vec![RegressionCase {
            name: "nir97-roundtrip".into(),
            probe: RegressionProbe::GraphRoundTrip,
        }],
    };
    candidate
        .validate()
        .map_err(|error| GgufError::NativeLowering(format!("candidate: {error:?}")))?;
    Ok(candidate)
}

impl LlamaConfig {
    pub fn from_model(model: &GgufModel) -> Result<Self, GgufError> {
        if model.architecture()? != "llama" {
            return Err(GgufError::UnsupportedArchitecture(
                model.architecture()?.to_owned(),
            ));
        }

        let context_length = required_u32(model, "llama.context_length")?;
        let embedding_length = required_u32(model, "llama.embedding_length")?;
        let block_count = required_u32(model, "llama.block_count")?;
        let feed_forward_length = required_u32(model, "llama.feed_forward_length")?;
        let head_count = required_u32(model, "llama.attention.head_count")?;
        let head_count_kv =
            optional_u32(model, "llama.attention.head_count_kv")?.unwrap_or(head_count);
        let rms_epsilon = required_f32(model, "llama.attention.layer_norm_rms_epsilon")?;

        if context_length == 0
            || embedding_length == 0
            || block_count == 0
            || feed_forward_length == 0
            || head_count == 0
            || head_count_kv == 0
            || embedding_length % head_count != 0
            || head_count % head_count_kv != 0
            || !rms_epsilon.is_finite()
            || rms_epsilon <= 0.0
        {
            return Err(GgufError::InvalidModelConfig("llama dimensions"));
        }

        let head_dim = embedding_length / head_count;
        let rope_dimension_count =
            optional_u32(model, "llama.rope.dimension_count")?.unwrap_or(head_dim);
        if rope_dimension_count != head_dim {
            return Err(GgufError::UnsupportedModelFeature(format!(
                "partial rotary dimension {rope_dimension_count} != head_dim {head_dim}"
            )));
        }

        let rope_freq_base = optional_f32(model, "llama.rope.freq_base")?.unwrap_or(10_000.0);
        if !rope_freq_base.is_finite() || (rope_freq_base - 10_000.0).abs() > 0.01 {
            return Err(GgufError::UnsupportedModelFeature(format!(
                "RoPE frequency base {rope_freq_base} requires parameterized RotaryPosition"
            )));
        }

        for (key, value) in &model.metadata {
            if key.starts_with("llama.rope.scaling") {
                if key.ends_with(".type") && value.as_string() == Some("none") {
                    continue;
                }
                return Err(GgufError::UnsupportedModelFeature(format!(
                    "RoPE scaling metadata '{key}' is not supported yet"
                )));
            }
        }

        Ok(Self {
            context_length,
            embedding_length,
            block_count,
            feed_forward_length,
            head_count,
            head_count_kv,
            head_dim,
            rms_epsilon,
            rope_dimension_count,
            rope_freq_base,
        })
    }
}

fn required_u32(model: &GgufModel, key: &'static str) -> Result<u32, GgufError> {
    model
        .metadata
        .get(key)
        .ok_or(GgufError::MissingMetadata(key))?
        .as_u32()
        .ok_or(GgufError::InvalidMetadataType(key))
}

fn optional_u32(model: &GgufModel, key: &'static str) -> Result<Option<u32>, GgufError> {
    model
        .metadata
        .get(key)
        .map(|value| value.as_u32().ok_or(GgufError::InvalidMetadataType(key)))
        .transpose()
}

fn required_f32(model: &GgufModel, key: &'static str) -> Result<f32, GgufError> {
    optional_f32(model, key)?.ok_or(GgufError::MissingMetadata(key))
}

fn optional_f32(model: &GgufModel, key: &'static str) -> Result<Option<f32>, GgufError> {
    model
        .metadata
        .get(key)
        .map(|value| numeric_f32(value).ok_or(GgufError::InvalidMetadataType(key)))
        .transpose()
}

fn numeric_f32(value: &GgufValue) -> Option<f32> {
    match value {
        GgufValue::Float32(value) => Some(*value),
        GgufValue::Float64(value) if value.is_finite() => Some(*value as f32),
        GgufValue::Uint8(value) => Some(f32::from(*value)),
        GgufValue::Uint16(value) => Some(f32::from(*value)),
        GgufValue::Uint32(value) => Some(*value as f32),
        GgufValue::Int8(value) => Some(f32::from(*value)),
        GgufValue::Int16(value) => Some(f32::from(*value)),
        GgufValue::Int32(value) => Some(*value as f32),
        _ => None,
    }
}

struct SourceTensorTable<'a> {
    file: &'a [u8],
    model: &'a GgufModel,
    tensors: BTreeMap<&'a str, &'a GgufTensorInfo>,
}

impl<'a> SourceTensorTable<'a> {
    fn new(file: &'a [u8], model: &'a GgufModel) -> Self {
        let tensors = model
            .tensors
            .iter()
            .map(|tensor| (tensor.name.as_str(), tensor))
            .collect::<BTreeMap<_, _>>();
        Self {
            file,
            model,
            tensors,
        }
    }

    fn add(
        &self,
        builder: &mut GraphBuilder,
        consumed: &mut BTreeSet<String>,
        name: &str,
        expected_source_dimensions: &[u64],
        transpose_2d: bool,
    ) -> Result<ValueId, GgufError> {
        let tensor = self
            .tensors
            .get(name)
            .copied()
            .ok_or_else(|| GgufError::MissingTensor(name.to_owned()))?;
        if tensor.dimensions.as_slice() != expected_source_dimensions {
            return Err(GgufError::UnsupportedModelFeature(format!(
                "tensor '{name}' shape {:?} != expected {:?}",
                tensor.dimensions, expected_source_dimensions
            )));
        }

        let transcoded = transcode_tensor(self.file, self.model, tensor, transpose_2d)?;
        consumed.insert(name.to_owned());
        builder.add_native_tensor(name, transcoded.dtype, transcoded.shape, transcoded.payload)
    }
}

fn reject_unconsumed_model_tensors(
    model: &GgufModel,
    consumed: &BTreeSet<String>,
) -> Result<(), GgufError> {
    for tensor in &model.tensors {
        if !consumed.contains(&tensor.name) {
            return Err(GgufError::UnsupportedModelFeature(format!(
                "unconsumed model tensor '{}'",
                tensor.name
            )));
        }
    }
    Ok(())
}

struct GraphBuilder {
    next_value: u32,
    next_node: u32,
    next_tensor: u32,
    inputs: Vec<ValueDecl>,
    nodes: Vec<Node>,
    tensors: Vec<NativeTensor>,
    bindings: Vec<LlamaTensorBinding>,
}

impl GraphBuilder {
    fn new() -> Self {
        Self {
            next_value: 0,
            next_node: 0,
            next_tensor: 1,
            inputs: Vec::new(),
            nodes: Vec::new(),
            tensors: Vec::new(),
            bindings: Vec::new(),
        }
    }

    fn add_external_input(&mut self, dtype: DType, rank: u8) -> Result<ValueId, GgufError> {
        let value = self.allocate_value()?;
        self.inputs.push(ValueDecl {
            id: value,
            ty: ValueType::Tensor { dtype, rank },
        });
        Ok(value)
    }

    fn add_native_tensor(
        &mut self,
        source_name: &str,
        dtype: DType,
        shape: Vec<u64>,
        payload: Vec<u8>,
    ) -> Result<ValueId, GgufError> {
        let value = self.allocate_value()?;
        let rank = u8::try_from(shape.len()).map_err(|_| GgufError::Overflow)?;
        self.inputs.push(ValueDecl {
            id: value,
            ty: if rank == 0 {
                ValueType::Scalar(dtype)
            } else {
                ValueType::Tensor { dtype, rank }
            },
        });

        let tensor_id = self.next_tensor;
        self.next_tensor = self.next_tensor.checked_add(1).ok_or(GgufError::Overflow)?;
        let byte_len = u64::try_from(payload.len()).map_err(|_| GgufError::LimitExceeded)?;
        self.tensors.push(NativeTensor {
            descriptor: TensorDescriptor {
                id: tensor_id,
                dtype,
                shape,
                byte_len,
            },
            graph_value: Some(value),
            quantization: QuantizationMetadata::None,
            payload,
        });
        self.bindings.push(LlamaTensorBinding {
            source_name: source_name.to_owned(),
            tensor_id,
            graph_value: value,
        });
        Ok(value)
    }

    fn add_constant_f32_scalar(&mut self, value: f32) -> Result<ValueId, GgufError> {
        if !value.is_finite() {
            return Err(GgufError::InvalidTensor);
        }
        self.add_native_tensor(
            "ntd97.constant.scalar",
            DType::F32,
            Vec::new(),
            value.to_le_bytes().to_vec(),
        )
    }

    fn add_constant_f32_vector(&mut self, values: &[f32]) -> Result<ValueId, GgufError> {
        if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
            return Err(GgufError::InvalidTensor);
        }
        let mut payload =
            Vec::with_capacity(values.len().checked_mul(4).ok_or(GgufError::Overflow)?);
        for value in values {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        self.add_native_tensor(
            "ntd97.constant.vector",
            DType::F32,
            vec![u64::try_from(values.len()).map_err(|_| GgufError::LimitExceeded)?],
            payload,
        )
    }

    fn add_node(
        &mut self,
        op: TensorOp,
        inputs: Vec<ValueId>,
        dtype: DType,
        rank: u8,
    ) -> Result<ValueId, GgufError> {
        let output = self.allocate_value()?;
        let node_id = NodeId(self.next_node);
        self.next_node = self.next_node.checked_add(1).ok_or(GgufError::Overflow)?;
        self.nodes.push(Node {
            id: node_id,
            op: OpKind::Tensor(op),
            inputs,
            outputs: vec![ValueDecl {
                id: output,
                ty: if rank == 0 {
                    ValueType::Scalar(dtype)
                } else {
                    ValueType::Tensor { dtype, rank }
                },
            }],
        });
        Ok(output)
    }

    fn allocate_value(&mut self) -> Result<ValueId, GgufError> {
        let value = ValueId(self.next_value);
        self.next_value = self.next_value.checked_add(1).ok_or(GgufError::Overflow)?;
        Ok(value)
    }

    fn finish(&mut self, outputs: Vec<ValueId>) -> Result<Graph, GgufError> {
        let graph = Graph {
            version: IrVersion::CURRENT,
            inputs: self.inputs.clone(),
            outputs,
            nodes: self.nodes.clone(),
        };
        graph
            .validate()
            .map_err(|error| GgufError::NativeLowering(format!("graph: {error:?}")))?;
        Ok(graph)
    }
}
