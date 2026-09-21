use std::collections::BTreeMap;

use ntd_assimilation::{
    build_native_package, lower_llama_model, lowered_llama_candidate, verify_native_package,
    AssimilationIdentity, ForgeSandbox, GgufModel, GgufValueType, LicenseRecord,
    NativeValidationSandbox, SourcePackage, GGUF_MAGIC, GGUF_VERSION,
};
use ntd_capsule::{load_native_generative_program, CapsuleView, MemoryContentStore};
use ntd_runtime::{
    CpuReferenceProvider, DistributionKind, GenerationConfig, GraphGenerator, QuantizationParams,
    SamplingMode, TensorLoader,
};

const ALIGNMENT: usize = 32;

struct TensorFixture {
    name: &'static str,
    dimensions: Vec<u64>,
    values: Vec<f32>,
    offset: u64,
}

fn push_string(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

fn push_string_kv(out: &mut Vec<u8>, key: &str, value: &str) {
    push_string(out, key);
    out.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
    push_string(out, value);
}

fn push_u32_kv(out: &mut Vec<u8>, key: &str, value: u32) {
    push_string(out, key);
    out.extend_from_slice(&(GgufValueType::Uint32 as u32).to_le_bytes());
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_f32_kv(out: &mut Vec<u8>, key: &str, value: f32) {
    push_string(out, key);
    out.extend_from_slice(&(GgufValueType::Float32 as u32).to_le_bytes());
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_tokens(out: &mut Vec<u8>, tokens: &[&str]) {
    push_string(out, "tokenizer.ggml.tokens");
    out.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
    out.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
    out.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for token in tokens {
        push_string(out, token);
    }
}

fn align(out: &mut Vec<u8>) {
    while out.len() % ALIGNMENT != 0 {
        out.push(0);
    }
}

fn identity_2() -> Vec<f32> {
    vec![1.0, 0.0, 0.0, 1.0]
}

fn fixture() -> Vec<u8> {
    let mut tensors = vec![
        TensorFixture {
            name: "token_embd.weight",
            dimensions: vec![2, 3],
            values: vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            offset: 0,
        },
        TensorFixture {
            name: "blk.0.attn_norm.weight",
            dimensions: vec![2],
            values: vec![1.0, 1.0],
            offset: 0,
        },
        TensorFixture {
            name: "blk.0.attn_q.weight",
            dimensions: vec![2, 2],
            values: identity_2(),
            offset: 0,
        },
        TensorFixture {
            name: "blk.0.attn_k.weight",
            dimensions: vec![2, 2],
            values: identity_2(),
            offset: 0,
        },
        TensorFixture {
            name: "blk.0.attn_v.weight",
            dimensions: vec![2, 2],
            values: identity_2(),
            offset: 0,
        },
        TensorFixture {
            name: "blk.0.attn_output.weight",
            dimensions: vec![2, 2],
            values: identity_2(),
            offset: 0,
        },
        TensorFixture {
            name: "blk.0.ffn_norm.weight",
            dimensions: vec![2],
            values: vec![1.0, 1.0],
            offset: 0,
        },
        TensorFixture {
            name: "blk.0.ffn_gate.weight",
            dimensions: vec![2, 2],
            values: identity_2(),
            offset: 0,
        },
        TensorFixture {
            name: "blk.0.ffn_up.weight",
            dimensions: vec![2, 2],
            values: identity_2(),
            offset: 0,
        },
        TensorFixture {
            name: "blk.0.ffn_down.weight",
            dimensions: vec![2, 2],
            values: identity_2(),
            offset: 0,
        },
        TensorFixture {
            name: "output_norm.weight",
            dimensions: vec![2],
            values: vec![1.0, 1.0],
            offset: 0,
        },
        TensorFixture {
            name: "output.weight",
            dimensions: vec![2, 3],
            values: vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            offset: 0,
        },
    ];

    let mut data = Vec::new();
    for tensor in &mut tensors {
        align(&mut data);
        tensor.offset = data.len() as u64;
        for value in &tensor.values {
            data.extend_from_slice(&value.to_le_bytes());
        }
    }

    let mut out = Vec::new();
    out.extend_from_slice(&GGUF_MAGIC);
    out.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&15u64.to_le_bytes());

    push_string_kv(&mut out, "general.architecture", "llama");
    push_string_kv(&mut out, "general.name", "ntd97-tiny-llama");
    push_u32_kv(&mut out, "general.alignment", ALIGNMENT as u32);
    push_string_kv(&mut out, "tokenizer.ggml.model", "llama");
    push_tokens(&mut out, &["a", "b", "<eos>"]);
    push_u32_kv(&mut out, "tokenizer.ggml.bos_token_id", 0);
    push_u32_kv(&mut out, "tokenizer.ggml.eos_token_id", 2);
    push_u32_kv(&mut out, "llama.context_length", 8);
    push_u32_kv(&mut out, "llama.embedding_length", 2);
    push_u32_kv(&mut out, "llama.block_count", 1);
    push_u32_kv(&mut out, "llama.feed_forward_length", 2);
    push_u32_kv(&mut out, "llama.attention.head_count", 1);
    push_u32_kv(&mut out, "llama.attention.head_count_kv", 1);
    push_f32_kv(&mut out, "llama.attention.layer_norm_rms_epsilon", 1.0e-5);
    push_u32_kv(&mut out, "llama.rope.dimension_count", 2);

    for tensor in &tensors {
        push_string(&mut out, tensor.name);
        out.extend_from_slice(&(tensor.dimensions.len() as u32).to_le_bytes());
        for dimension in &tensor.dimensions {
            out.extend_from_slice(&dimension.to_le_bytes());
        }
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&tensor.offset.to_le_bytes());
    }

    align(&mut out);
    out.extend_from_slice(&data);
    out
}

#[test]
fn tiny_llama_lowers_and_generates_through_native_runtime() {
    let bytes = fixture();
    let model = GgufModel::parse(&bytes).expect("parse");
    let lowered = lower_llama_model(&bytes, &model).expect("lower");

    assert_eq!(lowered.config.block_count, 1);
    assert_eq!(lowered.config.embedding_length, 2);
    assert_eq!(lowered.vocabulary_size, 3);
    assert!(!lowered.tensors.is_empty());
    lowered.graph.validate().expect("graph");

    let mut static_inputs = BTreeMap::new();
    for native in &lowered.tensors {
        let value = native.graph_value.expect("binding");
        let tensor = TensorLoader::load(
            native.descriptor.dtype,
            &native.descriptor.shape,
            QuantizationParams::None,
            &native.payload,
        )
        .expect("load native tensor");
        static_inputs.insert(value, tensor);
    }

    let generator = GraphGenerator::new(
        lowered.graph.clone(),
        CpuReferenceProvider,
        static_inputs,
        lowered.token_input,
        lowered.distribution_output,
        lowered.vocabulary_size,
    )
    .expect("generator");

    let config = GenerationConfig {
        max_new_tokens: 2,
        context_limit: 8,
        eos_token: None,
        distribution: DistributionKind::Logits,
        sampling: SamplingMode::Greedy,
    };
    let first = generator.generate_tokens(&[0], config).expect("first");
    let second = generator.generate_tokens(&[0], config).expect("second");

    assert_eq!(first, second);
    assert!(!first.generated_tokens.is_empty());
    assert!(first
        .generated_tokens
        .iter()
        .all(|token| (*token as usize) < lowered.vocabulary_size));
}

#[test]
fn lowering_rejects_unsupported_rope_scaling_instead_of_drifting() {
    let mut bytes = fixture();
    let model = GgufModel::parse(&bytes).expect("parse baseline");
    assert!(lower_llama_model(&bytes, &model).is_ok());

    let needle = b"llama.rope.dimension_count";
    let index = bytes
        .windows(needle.len())
        .position(|window| window == needle)
        .expect("rope metadata");
    let value_offset = index + needle.len() + 4;
    bytes[value_offset..value_offset + 4].copy_from_slice(&1u32.to_le_bytes());

    let model = GgufModel::parse(&bytes).expect("parse modified");
    assert!(lower_llama_model(&bytes, &model).is_err());
}

#[test]
fn lowered_llama_packages_as_signed_native_generative_intelligence() {
    let bytes = fixture();
    let model = GgufModel::parse(&bytes).expect("parse");
    let lowered = lower_llama_model(&bytes, &model).expect("lower");
    let candidate =
        lowered_llama_candidate("model.ntd97-tiny-llama", 1, &lowered).expect("candidate");

    let source = SourcePackage::new(
        "application/x-gguf",
        bytes,
        "memory://ntd97-tiny-llama.gguf",
        LicenseRecord::new("MIT", "test fixture").expect("license"),
        "NTD97 deterministic GGUF fixture",
    )
    .expect("source");

    let mut sandbox = NativeValidationSandbox;
    let report = sandbox.validate(&candidate).expect("sandbox");
    let identity = AssimilationIdentity::from_seed([97; 32]);
    let package = build_native_package(
        &candidate,
        &source.provenance,
        "ntd97.gguf.v3",
        &report,
        &identity,
    )
    .expect("package");
    verify_native_package(&package, &identity.verify_key()).expect("verify package");

    let view = CapsuleView::read(&package.native_capsule).expect("capsule");
    let loaded =
        load_native_generative_program(&view, &MemoryContentStore::default()).expect("load");
    assert_eq!(loaded.program.graph, lowered.graph);
    assert_eq!(loaded.program.tensors, lowered.tensors);
    assert_eq!(loaded.tokenizer, lowered.tokenizer);
}
