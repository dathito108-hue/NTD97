use std::collections::BTreeMap;

use ntd_assimilation::{
    lower_llama_f32_f16, GgufModel, GgufTensorInfo, GgufValue, GgufValueType,
    LlamaLoweringError,
};
use ntd_ir::ValueId;
use ntd_runtime::{
    CpuReferenceProvider, GraphExecutor, QuantizationParams, Tensor, TensorLoader,
};

fn metadata() -> BTreeMap<String, GgufValue> {
    BTreeMap::from([
        ("general.architecture".into(), GgufValue::String("llama".into())),
        ("general.name".into(), GgufValue::String("ntd97-mini-llama".into())),
        ("llama.context_length".into(), GgufValue::Uint32(32)),
        ("llama.embedding_length".into(), GgufValue::Uint32(4)),
        ("llama.block_count".into(), GgufValue::Uint32(1)),
        ("llama.feed_forward_length".into(), GgufValue::Uint32(8)),
        ("llama.attention.head_count".into(), GgufValue::Uint32(2)),
        ("llama.attention.head_count_kv".into(), GgufValue::Uint32(1)),
        ("llama.rope.dimension_count".into(), GgufValue::Uint32(2)),
        ("llama.rope.freq_base".into(), GgufValue::Float32(10_000.0)),
        (
            "llama.attention.layer_norm_rms_epsilon".into(),
            GgufValue::Float32(1.0e-5),
        ),
        ("llama.vocab_size".into(), GgufValue::Uint32(4)),
        ("tokenizer.ggml.model".into(), GgufValue::String("llama".into())),
        (
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array {
                element_type: GgufValueType::String,
                values: ["<unk>", "<s>", "</s>", "a"]
                    .into_iter()
                    .map(|value| GgufValue::String(value.into()))
                    .collect(),
            },
        ),
        ("tokenizer.ggml.bos_token_id".into(), GgufValue::Uint32(1)),
        ("tokenizer.ggml.eos_token_id".into(), GgufValue::Uint32(2)),
        ("tokenizer.ggml.unknown_token_id".into(), GgufValue::Uint32(0)),
    ])
}

fn required_tensors() -> Vec<(String, Vec<u64>)> {
    vec![
        ("token_embd.weight".into(), vec![4, 4]),
        ("output_norm.weight".into(), vec![4]),
        ("blk.0.attn_norm.weight".into(), vec![4]),
        ("blk.0.attn_q.weight".into(), vec![4, 4]),
        ("blk.0.attn_k.weight".into(), vec![4, 2]),
        ("blk.0.attn_v.weight".into(), vec![4, 2]),
        ("blk.0.attn_output.weight".into(), vec![4, 4]),
        ("blk.0.ffn_norm.weight".into(), vec![4]),
        ("blk.0.ffn_gate.weight".into(), vec![4, 8]),
        ("blk.0.ffn_up.weight".into(), vec![4, 8]),
        ("blk.0.ffn_down.weight".into(), vec![8, 4]),
    ]
}

fn fixture() -> (GgufModel, Vec<u8>) {
    let mut bytes = Vec::new();
    let mut tensors = Vec::new();
    for (name, dimensions) in required_tensors() {
        let data_offset = bytes.len() as u64;
        let elements = dimensions.iter().product::<u64>() as usize;
        let value = if name.ends_with("norm.weight") || name == "output_norm.weight" {
            1.0f32
        } else {
            0.0f32
        };
        for _ in 0..elements {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        tensors.push(GgufTensorInfo {
            name,
            dimensions,
            ggml_type: 0,
            data_offset,
        });
    }

    (
        GgufModel {
            version: 3,
            alignment: 32,
            metadata: metadata(),
            tensors,
            data_offset: 0,
            file_len: bytes.len() as u64,
        },
        bytes,
    )
}

#[test]
fn mini_llama_lowers_to_executable_nir97_graph() {
    let (model, bytes) = fixture();
    let draft = lower_llama_f32_f16(&model, &bytes).expect("lower");
    assert_eq!(draft.config.embedding_length, 4);
    assert_eq!(draft.config.head_count, 2);
    assert_eq!(draft.config.head_count_kv, 1);
    assert!(draft.tied_output);
    assert_eq!(draft.graph.validate(), Ok(()));

    let mut inputs = BTreeMap::new();
    for native in &draft.tensors {
        let value = native.graph_value.expect("bound tensor");
        let tensor = TensorLoader::load(
            native.descriptor.dtype,
            &native.descriptor.shape,
            QuantizationParams::None,
            &native.payload,
        )
        .expect("materialize");
        inputs.insert(value, tensor);
    }
    inputs.insert(
        draft.token_input,
        Tensor::new(vec![1], vec![3.0]).expect("token"),
    );

    let outputs = GraphExecutor::new(CpuReferenceProvider)
        .execute(&draft.graph, inputs)
        .expect("execute");
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].shape(), &[1, 4]);
    assert_eq!(outputs[0].data(), &[0.0, 0.0, 0.0, 0.0]);

    let sections = draft.tensor_sections().expect("sections");
    assert_eq!(sections.len(), draft.tensors.len() + 1);
}

#[test]
fn missing_required_weight_fails_closed() {
    let (mut model, bytes) = fixture();
    model.tensors.retain(|tensor| tensor.name != "blk.0.attn_q.weight");
    assert_eq!(
        lower_llama_f32_f16(&model, &bytes),
        Err(LlamaLoweringError::MissingTensor(
            "blk.0.attn_q.weight".into()
        ))
    );
}

#[test]
fn quantized_weight_is_not_silently_reinterpreted() {
    let (mut model, bytes) = fixture();
    model
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == "blk.0.ffn_gate.weight")
        .expect("gate")
        .ggml_type = 8;
    assert_eq!(
        lower_llama_f32_f16(&model, &bytes),
        Err(LlamaLoweringError::UnsupportedTensorType {
            name: "blk.0.ffn_gate.weight".into(),
            ggml_type: 8,
        })
    );
}

#[test]
fn nonstandard_rope_scaling_fails_closed() {
    let (mut model, bytes) = fixture();
    model.metadata.insert(
        "llama.rope.scaling.type".into(),
        GgufValue::String("yarn".into()),
    );
    assert!(matches!(
        lower_llama_f32_f16(&model, &bytes),
        Err(LlamaLoweringError::UnsupportedFeature(_))
    ));
}

#[test]
fn all_weight_bindings_are_unique() {
    let (model, bytes) = fixture();
    let draft = lower_llama_f32_f16(&model, &bytes).expect("lower");
    let mut values = draft
        .weight_bindings
        .iter()
        .map(|binding| binding.graph_value)
        .collect::<Vec<ValueId>>();
    values.sort_by_key(|value| value.0);
    values.dedup();
    assert_eq!(values.len(), draft.weight_bindings.len());
}
