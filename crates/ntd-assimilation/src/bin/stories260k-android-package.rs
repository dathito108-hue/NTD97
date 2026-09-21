#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use ntd_assimilation::{
    activate_thin_generative_capsule, build_streamed_native_package, lower_llama_model_to_shards,
    verify_native_package_with_shards, AssimilationIdentity, FileBackedTensorResolver,
    FileGgufSource, FileTensorShardStore, GgufModel, LicenseRecord, ProvenanceRecord,
    SandboxReport, StreamedPackageSpec,
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
const ANDROID_PROBE_TOKENS: usize = 16;

fn main() {
    if let Err(error) = run() {
        eprintln!("stories260K Android package export FAILED: {error}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args();
    let program = args
        .next()
        .unwrap_or_else(|| "stories260k-android-package".to_owned());
    let model_path = args
        .next()
        .ok_or_else(|| format!("usage: {program} <stories260K.gguf> <output-dir>"))?;
    let output_dir = args
        .next()
        .ok_or_else(|| format!("usage: {program} <stories260K.gguf> <output-dir>"))?;
    if args.next().is_some() {
        return Err(format!("usage: {program} <stories260K.gguf> <output-dir>"));
    }

    let model_path = PathBuf::from(model_path);
    let output_dir = PathBuf::from(output_dir);
    if output_dir.exists() {
        fs::remove_dir_all(&output_dir)
            .map_err(|error| format!("clear output directory: {error}"))?;
    }
    fs::create_dir_all(&output_dir).map_err(|error| format!("create output directory: {error}"))?;

    let source_bytes =
        fs::read(&model_path).map_err(|error| format!("read pinned source: {error}"))?;
    if source_bytes.len() != PINNED_MODEL_LEN || sha256(&source_bytes) != PINNED_MODEL_SHA256 {
        return Err("pinned stories260K GGUF identity mismatch".into());
    }
    let source_digest = sha256(&source_bytes);
    drop(source_bytes);

    let source = FileGgufSource::open(&model_path)
        .map_err(|error| format!("open file-backed GGUF: {error:?}"))?;
    let model =
        GgufModel::parse_source(&source).map_err(|error| format!("parse GGUF: {error:?}"))?;

    let shard_root = output_dir.join("ntp97-shards");
    let mut shard_store = FileTensorShardStore::open(&shard_root)
        .map_err(|error| format!("open shard store: {error:?}"))?;
    let streamed = lower_llama_model_to_shards(&source, &model, &mut shard_store)
        .map_err(|error| format!("streamed lowering: {error:?}"))?;

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
            "ntd97.gguf.v3.android-evidence",
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

    let activation = activate_thin_generative_capsule(&package.native_capsule, &shard_store)
        .map_err(|error| format!("activate signed Thin package: {error:?}"))?;
    let tokenizer = build_tokenizer(&activation.tokenizer)?;
    let bos = tokenizer
        .bos_token()
        .ok_or_else(|| "stories260K tokenizer has no BOS token".to_owned())?;
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
                max_new_tokens: ANDROID_PROBE_TOKENS,
                context_limit: usize::try_from(streamed.config.context_length)
                    .map_err(|_| "context length does not fit usize".to_owned())?,
                eos_token: Some(bos),
                distribution: DistributionKind::Logits,
                sampling: SamplingMode::Greedy,
            },
        )
        .map_err(|error| format!("host probe generation: {error:?}"))?;

    fs::write(output_dir.join("model.ncc97"), &package.native_capsule)
        .map_err(|error| format!("write capsule: {error}"))?;
    fs::write(output_dir.join("verify-key.bin"), identity.verify_key())
        .map_err(|error| format!("write verify key: {error}"))?;
    fs::write(
        output_dir.join("expected-token-ids.txt"),
        format_token_ids(&generated.generated_tokens),
    )
    .map_err(|error| format!("write expected token ids: {error}"))?;

    let manifest = format!(
        "source_sha256={}\ncapsule_sha256={}\nshard_count={}\nprobe_tokens={}\nexpected_token_ids={}\n",
        digest_hex(&source_digest),
        digest_hex(&package.capsule_hash),
        streamed.tensor_shards.len(),
        generated.generated_tokens.len(),
        format_token_ids(&generated.generated_tokens)
    );
    fs::write(output_dir.join("android-evidence-manifest.txt"), manifest)
        .map_err(|error| format!("write evidence manifest: {error}"))?;

    validate_output_tree(&output_dir, streamed.tensor_shards.len())?;
    println!(
        "stories260k_android_package=PASS capsule={} shards={} tokens={}",
        digest_hex(&package.capsule_hash),
        streamed.tensor_shards.len(),
        generated.generated_tokens.len()
    );
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

fn validate_output_tree(root: &Path, expected_shards: usize) -> Result<(), String> {
    for required in [
        "model.ncc97",
        "verify-key.bin",
        "expected-token-ids.txt",
        "android-evidence-manifest.txt",
    ] {
        if !root.join(required).is_file() {
            return Err(format!("missing exported file {required}"));
        }
    }
    let shard_count = fs::read_dir(root.join("ntp97-shards"))
        .map_err(|error| format!("read shard directory: {error}"))?
        .filter_map(Result::ok)
        .filter(|entry| entry.path().is_file())
        .count();
    if shard_count != expected_shards {
        return Err(format!(
            "exported shard count mismatch: expected {expected_shards}, got {shard_count}"
        ));
    }
    Ok(())
}

fn format_token_ids(tokens: &[u32]) -> String {
    tokens
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn digest_hex(digest: &Digest) -> String {
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("write digest");
    }
    out
}
