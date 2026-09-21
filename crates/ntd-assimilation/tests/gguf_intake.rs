use ntd_assimilation::{
    GgufConversionPlan, GgufError, GgufModel, GgufValueType, GGUF_MAGIC, GGUF_VERSION,
};

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

fn fixture() -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&GGUF_MAGIC);
    out.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    out.extend_from_slice(&1u64.to_le_bytes());
    out.extend_from_slice(&7u64.to_le_bytes());

    push_string_kv(&mut out, "general.architecture", "llama");
    push_string_kv(&mut out, "general.name", "ntd97-fixture");
    push_u32_kv(&mut out, "general.alignment", 32);
    push_string_kv(&mut out, "tokenizer.ggml.model", "llama");

    push_string(&mut out, "tokenizer.ggml.tokens");
    out.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
    out.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
    out.extend_from_slice(&3u64.to_le_bytes());
    push_string(&mut out, "<unk>");
    push_string(&mut out, "a");
    push_string(&mut out, "b");

    push_u32_kv(&mut out, "tokenizer.ggml.bos_token_id", 1);
    push_u32_kv(&mut out, "tokenizer.ggml.eos_token_id", 2);

    push_string(&mut out, "token_embd.weight");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&3u64.to_le_bytes());
    out.extend_from_slice(&4u64.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());

    while out.len() % 32 != 0 {
        out.push(0);
    }
    out.extend_from_slice(&[0u8; 48]);
    out
}

#[test]
fn parses_v3_metadata_tokenizer_and_tensor_table() {
    let model = GgufModel::parse(&fixture()).expect("parse");
    assert_eq!(model.version, 3);
    assert_eq!(model.alignment, 32);
    assert_eq!(model.architecture(), Ok("llama"));
    assert_eq!(model.model_name(), Some("ntd97-fixture"));
    assert_eq!(model.tensors.len(), 1);
    assert_eq!(model.tensors[0].dimensions, vec![3, 4]);

    let tokenizer = model.tokenizer().expect("tokenizer");
    assert_eq!(tokenizer.model, "llama");
    assert_eq!(
        tokenizer.tokens,
        vec![b"<unk>".to_vec(), b"a".to_vec(), b"b".to_vec()]
    );
    assert_eq!(tokenizer.bos_token, Some(1));
    assert_eq!(tokenizer.eos_token, Some(2));
}

#[test]
fn conversion_plan_never_activates_unverified_lowering() {
    let model = GgufModel::parse(&fixture()).expect("parse");
    let plan = GgufConversionPlan::from_model(&model).expect("plan");
    assert_eq!(plan.architecture, "llama");
    assert_eq!(plan.direct_tensor_count, 1);
    assert_eq!(plan.transcode_tensor_count, 0);
    assert!(!plan.activation_ready());
    assert!(plan
        .blockers
        .iter()
        .any(|item| item.contains("semantic equivalence")));
}

#[test]
fn rejects_future_version_and_truncation() {
    let mut bytes = fixture();
    bytes[4..8].copy_from_slice(&4u32.to_le_bytes());
    assert_eq!(
        GgufModel::parse(&bytes),
        Err(GgufError::UnsupportedVersion(4))
    );

    let bytes = fixture();
    assert_eq!(GgufModel::parse(&bytes[..20]), Err(GgufError::Truncated));
}

#[test]
fn rejects_non_power_of_two_alignment() {
    let mut bytes = fixture();
    let needle = b"general.alignment";
    let key = bytes
        .windows(needle.len())
        .position(|window| window == needle)
        .expect("alignment key");
    let value_offset = key + needle.len() + 4;
    bytes[value_offset..value_offset + 4].copy_from_slice(&3u32.to_le_bytes());
    assert_eq!(GgufModel::parse(&bytes), Err(GgufError::InvalidAlignment));
}

#[test]
fn rejects_out_of_range_special_token() {
    let mut bytes = fixture();
    let needle = b"tokenizer.ggml.eos_token_id";
    let key = bytes
        .windows(needle.len())
        .position(|window| window == needle)
        .expect("eos key");
    let value_offset = key + needle.len() + 4;
    bytes[value_offset..value_offset + 4].copy_from_slice(&99u32.to_le_bytes());
    assert_eq!(GgufModel::parse(&bytes), Err(GgufError::InvalidTokenizer));
}

#[test]
fn conversion_plan_distinguishes_supported_transcode_from_unsupported_types() {
    let mut model = GgufModel::parse(&fixture()).expect("parse");

    model.tensors[0].ggml_type = 2;
    let q4 = GgufConversionPlan::from_model(&model).expect("q4 plan");
    assert_eq!(q4.direct_tensor_count, 0);
    assert_eq!(q4.transcode_tensor_count, 1);
    assert_eq!(q4.unsupported_tensor_count, 0);
    assert!(!q4
        .blockers
        .iter()
        .any(|item| item.contains("unsupported GGML")));

    model.tensors[0].ggml_type = 30;
    let bf16 = GgufConversionPlan::from_model(&model).expect("bf16 plan");
    assert_eq!(bf16.direct_tensor_count, 1);
    assert_eq!(bf16.transcode_tensor_count, 0);
    assert_eq!(bf16.unsupported_tensor_count, 0);

    model.tensors[0].ggml_type = 12;
    let q4_k = GgufConversionPlan::from_model(&model).expect("q4_k plan");
    assert_eq!(q4_k.direct_tensor_count, 0);
    assert_eq!(q4_k.transcode_tensor_count, 0);
    assert_eq!(q4_k.unsupported_tensor_count, 1);
    assert!(q4_k
        .blockers
        .iter()
        .any(|item| item.contains("unsupported GGML")));
}

#[test]
fn conversion_plan_keeps_real_tokenizer_semantics_as_activation_blocker() {
    let model = GgufModel::parse(&fixture()).expect("parse");
    let plan = GgufConversionPlan::from_model(&model).expect("plan");
    assert!(!plan.activation_ready());
    assert!(plan.blockers.iter().any(|item| {
        item.contains("source-equivalent tokenization")
    }));
}
