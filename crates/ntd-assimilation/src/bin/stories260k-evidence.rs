#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

use ntd_assimilation::{
    activate_thin_generative_capsule, build_streamed_native_package, lower_llama_model_to_shards,
    verify_native_package_with_shards, AssimilationIdentity, FileBackedTensorResolver,
    FileGgufSource, FileTensorShardStore, GgufConversionPlan, GgufModel, LicenseRecord,
    NativeAssetStore, ProvenanceRecord, SandboxReport, StreamedPackageSpec,
};
use ntd_capsule::{decode_graph, encode_graph, sha256, Digest, NativeTokenizerModel};
use ntd_runtime::{
    CpuReferenceProvider, DistributionKind, GenerationConfig, GraphGenerator, LlamaSpmConfig,
    LlamaSpmTokenizer, SamplingMode,
};

const PINNED_MODEL_LEN: usize = 1_185_376;
const PINNED_MODEL_SHA256: Digest = [
    0x04, 0x7b, 0xf4, 0x64, 0x55, 0xa5, 0x44, 0x93, 0x1c, 0xff, 0x6f, 0xef, 0x14, 0xd7, 0x91, 0x01,
    0x54, 0xc5, 0x6a, 0xfb, 0xc2, 0x3a, 0xb1, 0xc5, 0xe5, 0x6a, 0x72, 0xe6, 0x99, 0x12, 0xc0, 0x4b,
];
const SOURCE_GOLDEN_STEPS: usize = 200;
const TOKENIZER_PROMPTS: [&str; 7] = [
    "",
    "hello",
    "hello!",
    "Once upon a time",
    "Lily's ball.",
    " red ball",
    "one  two",
];

fn main() {
    if let Err(error) = run() {
        eprintln!("stories260K evidence FAILED: {error}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args();
    let program = args
        .next()
        .unwrap_or_else(|| "stories260k-evidence".to_owned());
    let usage =
        || format!("usage: {program} <stories260K.gguf> <source-output> <source-token-ids>");
    let model_path = args.next().ok_or_else(usage)?;
    let source_output_path = args.next().ok_or_else(usage)?;
    let source_token_ids_path = args.next().ok_or_else(usage)?;
    if args.next().is_some() {
        return Err(usage());
    }

    let model_path = PathBuf::from(model_path);
    let source_output_path = PathBuf::from(source_output_path);
    let source_token_ids_path = PathBuf::from(source_token_ids_path);
    let source_bytes =
        fs::read(&model_path).map_err(|error| format!("read pinned source: {error}"))?;
    if source_bytes.len() != PINNED_MODEL_LEN {
        return Err(format!(
            "pinned source length mismatch: expected {PINNED_MODEL_LEN}, got {}",
            source_bytes.len()
        ));
    }
    let source_digest = sha256(&source_bytes);
    if source_digest != PINNED_MODEL_SHA256 {
        return Err(format!(
            "pinned source SHA-256 mismatch: expected {}, got {}",
            digest_hex(&PINNED_MODEL_SHA256),
            digest_hex(&source_digest)
        ));
    }
    drop(source_bytes);

    let mut source_output = fs::read(&source_output_path)
        .map_err(|error| format!("read source reference output: {error}"))?;
    if source_output.last() == Some(&b'\n') {
        source_output.pop();
    }
    if source_output.is_empty() {
        return Err("source reference output is empty".into());
    }
    let source_output_digest = sha256(&source_output);
    let source_token_ids = parse_source_token_ids(&source_token_ids_path)?;

    let source = FileGgufSource::open(&model_path)
        .map_err(|error| format!("open file-backed GGUF: {error:?}"))?;
    let model =
        GgufModel::parse_source(&source).map_err(|error| format!("parse GGUF: {error:?}"))?;
    let plan = GgufConversionPlan::from_model(&model)
        .map_err(|error| format!("conversion plan: {error:?}"))?;

    validate_plan(&plan)?;

    let shard_root = evidence_shard_root();
    if shard_root.exists() {
        fs::remove_dir_all(&shard_root)
            .map_err(|error| format!("clear shard root {}: {error}", shard_root.display()))?;
    }
    let mut shard_store = FileTensorShardStore::open(&shard_root)
        .map_err(|error| format!("open shard store: {error:?}"))?;

    let result = (|| -> Result<(), String> {
        let streamed = lower_llama_model_to_shards(&source, &model, &mut shard_store)
            .map_err(|error| format!("streamed lowering: {error:?}"))?;
        validate_streamed_model(&streamed)?;

        let encoded_graph =
            encode_graph(&streamed.graph).map_err(|error| format!("encode graph: {error:?}"))?;
        let decoded_graph =
            decode_graph(&encoded_graph).map_err(|error| format!("decode graph: {error:?}"))?;
        if decoded_graph != streamed.graph {
            return Err("NIR97 graph round-trip drift".into());
        }

        let provenance = ProvenanceRecord {
            source_uri: "https://huggingface.co/ggml-org/tiny-llamas/resolve/main/stories260K.gguf"
                .into(),
            source_digest,
            license: LicenseRecord::new(
                "MIT",
                "stories260K reference lineage from karpathy/llama2.c TinyStories example",
            )
            .map_err(|error| format!("license record: {error:?}"))?,
            attribution: "Karpathy llama2.c stories260K; GGUF conversion by llama.cpp".into(),
        };
        let report = SandboxReport {
            isolated: true,
            network_used: false,
            external_write_used: false,
            passed_regressions: vec!["nir97-roundtrip".into()],
        };
        let identity = AssimilationIdentity::from_seed([0x97; 32]);
        let package = build_streamed_native_package(
            StreamedPackageSpec::new(
                "model.ntd97.stories260k",
                1,
                "ntd97.gguf.v3.stories260k-evidence",
            )
            .map_err(|error| format!("package spec: {error:?}"))?,
            &streamed,
            &provenance,
            &report,
            &shard_store,
            &identity,
        )
        .map_err(|error| format!("build signed Thin package: {error:?}"))?;
        verify_native_package_with_shards(&package, &identity.verify_key(), &shard_store)
            .map_err(|error| format!("verify signed Thin package: {error:?}"))?;

        let mut asset_store = NativeAssetStore::new(identity.verify_key());
        asset_store
            .commit_batch_with_shards(vec![package.clone()], &shard_store)
            .map_err(|error| format!("commit signed Thin package: {error:?}"))?;
        let active = asset_store
            .active("model.ntd97.stories260k")
            .map_err(|error| format!("load active package: {error:?}"))?;
        if active.capsule_hash != package.capsule_hash {
            return Err("active package digest drift".into());
        }

        let activation = activate_thin_generative_capsule(&package.native_capsule, &shard_store)
            .map_err(|error| format!("activate signed Thin package: {error:?}"))?;
        let tokenizer = build_tokenizer(&activation.tokenizer)?;
        let bos = tokenizer
            .bos_token()
            .ok_or_else(|| "stories260K tokenizer has no BOS token".to_owned())?;
        if bos != 1 {
            return Err(format!("stories260K BOS mismatch: expected 1, got {bos}"));
        }

        validate_tokenizer_equivalence(&tokenizer, &source_token_ids)?;
        println!("stories260k_tokenizer_equivalence=PASS");

        let context_limit = usize::try_from(streamed.config.context_length)
            .map_err(|_| "context length does not fit usize".to_owned())?;
        let source_steps = SOURCE_GOLDEN_STEPS.min(context_limit);
        let resolver = FileBackedTensorResolver::from_activation(shard_store.clone(), &activation);
        let generator = GraphGenerator::new(
            activation.graph.clone(),
            CpuReferenceProvider,
            BTreeMap::new(),
            activation.manifest.token_input,
            usize::try_from(activation.manifest.distribution_output)
                .map_err(|_| "distribution output does not fit usize".to_owned())?,
            usize::try_from(activation.manifest.vocabulary_size)
                .map_err(|_| "vocabulary size does not fit usize".to_owned())?,
        )
        .map_err(|error| format!("build native generator: {error:?}"))?;

        let generated = generator
            .generate_tokens_with_resolver(
                &resolver,
                &[bos],
                GenerationConfig {
                    max_new_tokens: source_steps,
                    context_limit,
                    eos_token: Some(bos),
                    distribution: DistributionKind::Logits,
                    sampling: SamplingMode::Greedy,
                },
            )
            .map_err(|error| format!("native generation: {error:?}"))?;
        let text = tokenizer
            .decode_after(Some(bos), &generated.generated_tokens, true)
            .map_err(|error| format!("native decode: {error:?}"))?;
        let text_digest = sha256(text.as_bytes());

        println!("source_sha256={}", digest_hex(&source_digest));
        println!("context_length={}", streamed.config.context_length);
        println!("source_steps={source_steps}");
        println!("generated_token_count={}", generated.generated_tokens.len());
        println!("source_text_bytes={}", source_output.len());
        println!("source_text_sha256={}", digest_hex(&source_output_digest));
        println!("generated_text_bytes={}", text.len());
        println!("generated_text_sha256={}", digest_hex(&text_digest));

        if text.as_bytes() != source_output.as_slice() {
            let mismatch = first_mismatch(text.as_bytes(), &source_output);
            return Err(format!(
                "source output mismatch at byte {mismatch}: source {} bytes / {}, NTD97 {} bytes / {}",
                source_output.len(),
                digest_hex(&source_output_digest),
                text.len(),
                digest_hex(&text_digest)
            ));
        }

        println!("stories260k_source_equivalence=PASS");
        Ok(())
    })();

    let _ = fs::remove_dir_all(&shard_root);
    result
}

fn validate_plan(plan: &GgufConversionPlan) -> Result<(), String> {
    if plan.architecture != "llama"
        || plan.tokenizer_model != "llama"
        || plan.vocabulary_size != 512
        || plan.tensor_count != 48
        || plan.direct_tensor_count != 48
        || plan.transcode_tensor_count != 0
        || plan.unsupported_tensor_count != 0
    {
        return Err(format!("pinned stories260K plan drift: {plan:?}"));
    }
    Ok(())
}

fn validate_streamed_model(
    model: &ntd_assimilation::StreamedLoweredLlamaModel,
) -> Result<(), String> {
    if model.config.embedding_length != 64
        || model.config.feed_forward_length != 172
        || model.config.block_count != 5
        || model.config.head_count != 8
        || model.config.head_count_kv != 4
        || model.config.rope_dimension_count != 8
        || (model.config.rms_epsilon - 1.0e-5).abs() > f32::EPSILON
        || model.vocabulary_size != 512
        || model.tensor_shards.len() != 52
        || model.bindings.len() != 52
    {
        return Err(format!(
            "pinned stories260K model config drift: context={} embd={} ff={} blocks={} heads={}/{} rope={} rms={} vocab={} shards={} bindings={}",
            model.config.context_length,
            model.config.embedding_length,
            model.config.feed_forward_length,
            model.config.block_count,
            model.config.head_count,
            model.config.head_count_kv,
            model.config.rope_dimension_count,
            model.config.rms_epsilon,
            model.vocabulary_size,
            model.tensor_shards.len(),
            model.bindings.len()
        ));
    }
    Ok(())
}

fn build_tokenizer(
    descriptor: &ntd_capsule::NativeTokenizerDescriptor,
) -> Result<LlamaSpmTokenizer, String> {
    let NativeTokenizerModel::LlamaSpm {
        score_bits,
        token_types,
        add_space_prefix,
        add_bos_token,
        add_eos_token,
    } = &descriptor.model
    else {
        return Err("stories260K did not activate as native LLaMA SPM".into());
    };

    LlamaSpmTokenizer::new(
        descriptor.tokens.clone(),
        score_bits.clone(),
        token_types.clone(),
        LlamaSpmConfig {
            bos_token: descriptor.bos_token,
            eos_token: descriptor.eos_token,
            unknown_token: descriptor.unknown_token,
            add_space_prefix: *add_space_prefix,
            add_bos_token: *add_bos_token,
            add_eos_token: *add_eos_token,
        },
    )
    .map_err(|error| format!("build native tokenizer: {error:?}"))
}

fn parse_source_token_ids(path: &PathBuf) -> Result<Vec<Vec<u32>>, String> {
    let text =
        fs::read_to_string(path).map_err(|error| format!("read source tokenizer ids: {error}"))?;
    let lines = text.lines().collect::<Vec<_>>();
    if lines.len() != TOKENIZER_PROMPTS.len() {
        return Err(format!(
            "source tokenizer trace count mismatch: expected {}, got {}",
            TOKENIZER_PROMPTS.len(),
            lines.len()
        ));
    }

    lines
        .into_iter()
        .enumerate()
        .map(|(case, line)| {
            if line.trim().is_empty() {
                return Err(format!("source tokenizer trace case {case} is empty"));
            }
            line.split(',')
                .map(|value| {
                    value.parse::<u32>().map_err(|error| {
                        format!("invalid source tokenizer id in case {case}: {value:?}: {error}")
                    })
                })
                .collect()
        })
        .collect()
}

fn validate_tokenizer_equivalence(
    tokenizer: &LlamaSpmTokenizer,
    source_token_ids: &[Vec<u32>],
) -> Result<(), String> {
    for (case, prompt) in TOKENIZER_PROMPTS.iter().enumerate() {
        let native = tokenizer
            .encode(prompt, true)
            .map_err(|error| format!("native tokenizer case {case}: {error:?}"))?;
        let source = source_token_ids
            .get(case)
            .ok_or_else(|| format!("missing source tokenizer case {case}"))?;
        if &native != source {
            return Err(format!(
                "tokenizer mismatch case {case} prompt={prompt:?}: source={source:?} NTD97={native:?}"
            ));
        }
    }
    Ok(())
}

fn evidence_shard_root() -> PathBuf {
    env::temp_dir().join(format!("ntd97-stories260k-evidence-{}", process::id()))
}

fn first_mismatch(left: &[u8], right: &[u8]) -> usize {
    let shared = left.len().min(right.len());
    left.iter()
        .zip(right.iter())
        .position(|(left, right)| left != right)
        .unwrap_or(shared)
}

fn digest_hex(digest: &Digest) -> String {
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("write digest");
    }
    out
}
