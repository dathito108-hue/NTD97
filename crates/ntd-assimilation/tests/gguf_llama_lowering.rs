use std::{
    cell::Cell,
    collections::BTreeMap,
    sync::atomic::{AtomicU64, Ordering},
};

use ntd_assimilation::{
    activate_thin_generative_capsule, build_native_package, build_streamed_native_package,
    lower_llama_model, lower_llama_model_from_source, lower_llama_model_to_shards,
    lowered_llama_candidate, streamed_llama_thin_capsule, verify_native_package,
    verify_native_package_with_shards, AssimilationIdentity, FileBackedTensorResolver,
    FileTensorShardStore, ForgeSandbox, GgufByteSource, GgufError, GgufModel, GgufValueType,
    LicenseRecord, NativeAssetStore, NativeValidationSandbox, SliceGgufSource, SourcePackage,
    GGUF_MAGIC, GGUF_VERSION,
};
use ntd_capsule::{
    load_native_generative_program, CapsuleKind, CapsuleView, MemoryContentStore,
    NativeTokenizerModel,
};
use ntd_runtime::{
    CpuReferenceProvider, DistributionKind, GenerationConfig, Gpt2BpeConfig, Gpt2BpeTokenizer,
    GraphGenerator, LlamaSpmConfig, LlamaSpmTokenizer, QuantizationParams, SamplingMode,
    TensorLoader,
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

fn push_bool_kv(out: &mut Vec<u8>, key: &str, value: bool) {
    push_string(out, key);
    out.extend_from_slice(&(GgufValueType::Bool as u32).to_le_bytes());
    out.push(u8::from(value));
}

fn push_f32_array(out: &mut Vec<u8>, key: &str, values: &[f32]) {
    push_string(out, key);
    out.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
    out.extend_from_slice(&(GgufValueType::Float32 as u32).to_le_bytes());
    out.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

fn push_i32_array(out: &mut Vec<u8>, key: &str, values: &[i32]) {
    push_string(out, key);
    out.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
    out.extend_from_slice(&(GgufValueType::Int32 as u32).to_le_bytes());
    out.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
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

fn push_string_array(out: &mut Vec<u8>, key: &str, values: &[String]) {
    push_string(out, key);
    out.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
    out.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
    out.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        push_string(out, value);
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
    out.extend_from_slice(&20u64.to_le_bytes());

    push_string_kv(&mut out, "general.architecture", "llama");
    push_string_kv(&mut out, "general.name", "ntd97-tiny-llama");
    push_u32_kv(&mut out, "general.alignment", ALIGNMENT as u32);
    push_string_kv(&mut out, "tokenizer.ggml.model", "llama");
    push_tokens(&mut out, &["a", "b", "<eos>"]);
    push_f32_array(&mut out, "tokenizer.ggml.scores", &[0.0, 0.0, -1000.0]);
    push_i32_array(&mut out, "tokenizer.ggml.token_type", &[1, 1, 3]);
    push_bool_kv(&mut out, "tokenizer.ggml.add_space_prefix", false);
    push_bool_kv(&mut out, "tokenizer.ggml.add_bos_token", false);
    push_bool_kv(&mut out, "tokenizer.ggml.add_eos_token", false);
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

fn gpt2_byte_encoder() -> [char; 256] {
    let mut output = ['\0'; 256];
    let mut assigned = [false; 256];

    for byte in 0x21u16..=0x7Eu16 {
        let index = usize::from(byte);
        output[index] = char::from_u32(u32::from(byte)).expect("ascii");
        assigned[index] = true;
    }
    for byte in 0xA1u16..=0xACu16 {
        let index = usize::from(byte);
        output[index] = char::from_u32(u32::from(byte)).expect("latin1");
        assigned[index] = true;
    }
    for byte in 0xAEu16..=0xFFu16 {
        let index = usize::from(byte);
        output[index] = char::from_u32(u32::from(byte)).expect("latin1");
        assigned[index] = true;
    }

    let mut fallback = 0u32;
    for byte in 0u16..=0xFFu16 {
        let index = usize::from(byte);
        if assigned[index] {
            continue;
        }
        output[index] = char::from_u32(256 + fallback).expect("gpt2 byte codepoint");
        fallback += 1;
    }
    output
}

fn gpt2_fixture() -> Vec<u8> {
    let encoder = gpt2_byte_encoder();
    let mut tokens = encoder
        .iter()
        .map(|character| character.to_string())
        .collect::<Vec<_>>();
    tokens.push("ab".into());
    tokens.push("<|endoftext|>".into());
    let vocab_size = tokens.len();
    let vocab_u64 = u64::try_from(vocab_size).expect("vocab");

    let mut tensors = vec![
        TensorFixture {
            name: "token_embd.weight",
            dimensions: vec![2, vocab_u64],
            values: vec![0.0; 2 * vocab_size],
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
            dimensions: vec![2, vocab_u64],
            values: vec![0.0; 2 * vocab_size],
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
    out.extend_from_slice(&18u64.to_le_bytes());

    push_string_kv(&mut out, "general.architecture", "llama");
    push_string_kv(&mut out, "general.name", "ntd97-tiny-gpt2-tokenized-llama");
    push_u32_kv(&mut out, "general.alignment", ALIGNMENT as u32);
    push_string_kv(&mut out, "tokenizer.ggml.model", "gpt2");
    push_string_array(&mut out, "tokenizer.ggml.tokens", &tokens);
    push_string_array(&mut out, "tokenizer.ggml.merges", &["a b".to_owned()]);
    push_string_kv(&mut out, "tokenizer.ggml.pre", "gpt-2");
    push_bool_kv(&mut out, "tokenizer.ggml.add_bos_token", false);
    push_bool_kv(&mut out, "tokenizer.ggml.add_eos_token", false);
    push_u32_kv(
        &mut out,
        "tokenizer.ggml.eos_token_id",
        u32::try_from(vocab_size - 1).expect("eos"),
    );
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

static NEXT_SHARD_DIR: AtomicU64 = AtomicU64::new(1);

fn shard_root() -> std::path::PathBuf {
    let id = NEXT_SHARD_DIR.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "ntd97-streamed-lowering-{}-{id}",
        std::process::id()
    ))
}

struct CountingSource<'a> {
    bytes: &'a [u8],
    reads: Cell<usize>,
    max_read: Cell<usize>,
}

impl GgufByteSource for CountingSource<'_> {
    fn byte_len(&self) -> Result<u64, GgufError> {
        u64::try_from(self.bytes.len()).map_err(|_| GgufError::LimitExceeded)
    }

    fn read_exact_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, GgufError> {
        self.reads.set(self.reads.get() + 1);
        self.max_read.set(self.max_read.get().max(len));
        SliceGgufSource::new(self.bytes).read_exact_at(offset, len)
    }
}

#[test]
fn source_backed_lowering_matches_in_memory_lowering_with_bounded_reads() {
    let bytes = fixture();
    let source = CountingSource {
        bytes: &bytes,
        reads: Cell::new(0),
        max_read: Cell::new(0),
    };

    let source_model = GgufModel::parse_source(&source).expect("parse source");
    let source_lowered =
        lower_llama_model_from_source(&source, &source_model).expect("lower source");

    let memory_model = GgufModel::parse(&bytes).expect("parse memory");
    let memory_lowered = lower_llama_model(&bytes, &memory_model).expect("lower memory");

    assert_eq!(source_lowered, memory_lowered);
    assert!(source.reads.get() > source_model.tensors.len());
    assert!(source.max_read.get() < bytes.len());
}

#[test]
fn streamed_ntp97_shards_reload_through_canonical_thin_capsule() {
    let bytes = fixture();
    let source = SliceGgufSource::new(&bytes);
    let model = GgufModel::parse_source(&source).expect("parse");
    let expected = lower_llama_model(&bytes, &model).expect("in-memory lower");

    let root = shard_root();
    let mut shard_store = FileTensorShardStore::open(&root).expect("shard store");
    let streamed =
        lower_llama_model_to_shards(&source, &model, &mut shard_store).expect("stream lower");

    assert_eq!(streamed.graph, expected.graph);
    assert_eq!(streamed.tokenizer, expected.tokenizer);
    assert_eq!(streamed.bindings, expected.bindings);
    assert_eq!(streamed.tensor_shards.len(), expected.tensors.len());
    assert_eq!(
        streamed
            .tensor_shards
            .iter()
            .map(|shard| shard.descriptor.clone())
            .collect::<Vec<_>>(),
        expected
            .tensors
            .iter()
            .map(|tensor| tensor.descriptor.clone())
            .collect::<Vec<_>>()
    );

    let capsule =
        streamed_llama_thin_capsule(*b"NTD97-STREAM-001", &streamed).expect("thin capsule");
    let view = CapsuleView::read(&capsule).expect("capsule");
    assert_eq!(view.kind, CapsuleKind::Thin);
    assert_eq!(
        view.chunks
            .iter()
            .filter(|chunk| matches!(chunk.storage, ntd_capsule::ChunkStorageView::External))
            .count(),
        streamed.tensor_shards.len()
    );

    let mut content = MemoryContentStore::default();
    for shard in &streamed.tensor_shards {
        let bytes = shard_store.read_verified(shard).expect("read shard");
        content
            .insert_verified(shard.hash, bytes)
            .expect("insert verified");
    }
    let loaded = load_native_generative_program(&view, &content).expect("canonical load");
    assert_eq!(loaded.program.graph, expected.graph);
    assert_eq!(loaded.program.tensors, expected.tensors);
    assert_eq!(loaded.tokenizer, expected.tokenizer);

    std::fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn signed_thin_package_activates_and_generates_with_lazy_file_backed_tensors() {
    let bytes = fixture();
    let source_bytes = SliceGgufSource::new(&bytes);
    let model = GgufModel::parse_source(&source_bytes).expect("parse");
    let expected = lower_llama_model(&bytes, &model).expect("in-memory lower");

    let root = shard_root();
    let mut shard_store = FileTensorShardStore::open(&root).expect("shard store");
    let streamed =
        lower_llama_model_to_shards(&source_bytes, &model, &mut shard_store).expect("stream lower");

    let candidate =
        lowered_llama_candidate("model.ntd97-thin-reference", 1, &expected).expect("candidate");
    let mut sandbox = NativeValidationSandbox;
    let report = sandbox.validate(&candidate).expect("sandbox");
    let source = SourcePackage::new(
        "application/x-gguf",
        bytes.clone(),
        "memory://ntd97-thin.gguf",
        LicenseRecord::new("MIT", "test fixture").expect("license"),
        "NTD97 deterministic Thin NCC97 fixture",
    )
    .expect("source");
    let identity = AssimilationIdentity::from_seed([33; 32]);

    let package = build_streamed_native_package(
        "model.ntd97-thin",
        1,
        &streamed,
        &source.provenance,
        "ntd97.gguf.v3",
        &report,
        &shard_store,
        &identity,
    )
    .expect("signed thin package");
    verify_native_package_with_shards(&package, &identity.verify_key(), &shard_store)
        .expect("verify thin package");

    let view = CapsuleView::read(&package.native_capsule).expect("capsule");
    assert_eq!(view.kind, CapsuleKind::Thin);

    let mut asset_store = NativeAssetStore::new(identity.verify_key());
    asset_store
        .commit_batch_with_shards(vec![package.clone()], &shard_store)
        .expect("commit thin package");
    assert_eq!(
        asset_store
            .active("model.ntd97-thin")
            .expect("active package")
            .capsule_hash,
        package.capsule_hash
    );

    let activation =
        activate_thin_generative_capsule(&package.native_capsule, &shard_store).expect("activate");
    assert_eq!(activation.graph, streamed.graph);
    assert_eq!(activation.tokenizer, streamed.tokenizer);
    assert_eq!(activation.manifest.token_input, streamed.token_input);
    assert_eq!(
        usize::try_from(activation.manifest.distribution_output).expect("distribution output"),
        streamed.distribution_output
    );
    assert_eq!(
        usize::try_from(activation.manifest.vocabulary_size).expect("vocabulary size"),
        streamed.vocabulary_size
    );
    assert_eq!(activation.tensor_shards.len(), streamed.tensor_shards.len());

    let resolver = FileBackedTensorResolver::from_activation(shard_store.clone(), &activation);
    assert_eq!(resolver.binding_count(), streamed.tensor_shards.len());

    let mut static_inputs = BTreeMap::new();
    for native in &expected.tensors {
        let value = native.graph_value.expect("binding");
        let tensor = TensorLoader::load(
            native.descriptor.dtype,
            &native.descriptor.shape,
            QuantizationParams::None,
            &native.payload,
        )
        .expect("load reference tensor");
        static_inputs.insert(value, tensor);
    }

    let reference_generator = GraphGenerator::new(
        expected.graph.clone(),
        CpuReferenceProvider,
        static_inputs,
        expected.token_input,
        expected.distribution_output,
        expected.vocabulary_size,
    )
    .expect("reference generator");
    let lazy_generator = GraphGenerator::new(
        activation.graph.clone(),
        CpuReferenceProvider,
        BTreeMap::new(),
        activation.manifest.token_input,
        usize::try_from(activation.manifest.distribution_output).expect("distribution output"),
        usize::try_from(activation.manifest.vocabulary_size).expect("vocabulary size"),
    )
    .expect("lazy generator");

    let config = GenerationConfig {
        max_new_tokens: 2,
        context_limit: 8,
        eos_token: None,
        distribution: DistributionKind::Logits,
        sampling: SamplingMode::Greedy,
    };
    let reference = reference_generator
        .generate_tokens(&[0], config)
        .expect("reference generation");
    let lazy = lazy_generator
        .generate_tokens_with_resolver(&resolver, &[0], config)
        .expect("lazy generation");
    assert_eq!(lazy, reference);

    let first_shard = streamed.tensor_shards.first().expect("first shard");
    std::fs::write(shard_store.shard_path(&first_shard.hash), b"tampered").expect("tamper shard");
    assert!(
        verify_native_package_with_shards(&package, &identity.verify_key(), &shard_store).is_err(),
        "external shard tampering must invalidate Thin package activation"
    );

    std::fs::remove_dir_all(root).expect("cleanup");
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
fn canonical_gpt2_lowers_packages_and_executes_native_tokenizer() {
    let bytes = gpt2_fixture();
    let model = GgufModel::parse(&bytes).expect("parse");
    let lowered = lower_llama_model(&bytes, &model).expect("lower");

    let candidate =
        lowered_llama_candidate("model.ntd97-tiny-gpt2", 1, &lowered).expect("candidate");
    let source = SourcePackage::new(
        "application/x-gguf",
        bytes,
        "memory://ntd97-tiny-gpt2.gguf",
        LicenseRecord::new("MIT", "test fixture").expect("license"),
        "NTD97 deterministic GPT-2 tokenizer GGUF fixture",
    )
    .expect("source");
    let mut sandbox = NativeValidationSandbox;
    let report = sandbox.validate(&candidate).expect("sandbox");
    let identity = AssimilationIdentity::from_seed([42; 32]);
    let package = build_native_package(
        &candidate,
        &source.provenance,
        "ntd97.gguf.v3",
        &report,
        &identity,
    )
    .expect("package");
    verify_native_package(&package, &identity.verify_key()).expect("verify");

    let view = CapsuleView::read(&package.native_capsule).expect("capsule");
    let loaded =
        load_native_generative_program(&view, &MemoryContentStore::default()).expect("load");
    assert_eq!(loaded.tokenizer, lowered.tokenizer);
    let manifest = loaded.manifest.expect("generative manifest");
    assert_eq!(manifest.token_input, lowered.token_input);
    assert_eq!(
        usize::try_from(manifest.distribution_output).expect("distribution output"),
        lowered.distribution_output
    );
    assert_eq!(
        usize::try_from(manifest.vocabulary_size).expect("vocabulary size"),
        lowered.vocabulary_size
    );

    let NativeTokenizerModel::Gpt2Bpe {
        merges,
        add_bos_token,
        add_eos_token,
    } = &loaded.tokenizer.model
    else {
        panic!("expected native GPT-2 BPE tokenizer");
    };
    let runtime = Gpt2BpeTokenizer::new(
        loaded.tokenizer.tokens.clone(),
        merges.clone(),
        Gpt2BpeConfig {
            bos_token: loaded.tokenizer.bos_token,
            eos_token: loaded.tokenizer.eos_token,
            unknown_token: loaded.tokenizer.unknown_token,
            add_bos_token: *add_bos_token,
            add_eos_token: *add_eos_token,
        },
    )
    .expect("runtime tokenizer");

    assert_eq!(runtime.encode("ab!", false).expect("encode"), vec![256, 33]);
    assert_eq!(runtime.decode(&[256, 33], false).expect("decode"), "ab!");
}

#[test]
fn lowering_rejects_incomplete_llama_spm_metadata() {
    let bytes = fixture();
    let mut model = GgufModel::parse(&bytes).expect("parse");
    model.metadata.remove("tokenizer.ggml.scores");

    assert!(matches!(
        lower_llama_model(&bytes, &model),
        Err(ntd_assimilation::GgufError::UnsupportedModelFeature(message))
            if message.contains("missing scores/token types")
    ));
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
    let manifest = loaded.manifest.expect("generative manifest");
    assert_eq!(manifest.token_input, lowered.token_input);
    assert_eq!(
        usize::try_from(manifest.distribution_output).expect("distribution output"),
        lowered.distribution_output
    );
    assert_eq!(
        usize::try_from(manifest.vocabulary_size).expect("vocabulary size"),
        lowered.vocabulary_size
    );

    let NativeTokenizerModel::LlamaSpm {
        score_bits,
        token_types,
        add_space_prefix,
        add_bos_token,
        add_eos_token,
    } = &loaded.tokenizer.model
    else {
        panic!("expected native LLaMA SPM tokenizer");
    };
    let runtime_tokenizer = LlamaSpmTokenizer::new(
        loaded.tokenizer.tokens.clone(),
        score_bits.clone(),
        token_types.clone(),
        LlamaSpmConfig {
            bos_token: loaded.tokenizer.bos_token,
            eos_token: loaded.tokenizer.eos_token,
            unknown_token: loaded.tokenizer.unknown_token,
            add_space_prefix: *add_space_prefix,
            add_bos_token: *add_bos_token,
            add_eos_token: *add_eos_token,
        },
    )
    .expect("runtime tokenizer");
    assert_eq!(
        runtime_tokenizer.encode("ab", true).expect("tokenize"),
        vec![0, 1]
    );
}
