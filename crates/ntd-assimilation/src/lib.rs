#![forbid(unsafe_code)]

mod activation;
mod forge;
mod gguf;
mod importer;
mod package;
mod sandbox;
mod source;
mod store;

pub use activation::{
    activate_thin_generative_capsule, FileBackedTensorResolver, ThinActivationError,
    ThinGenerativeActivation,
};
pub use forge::{CapabilityForge, ForgePolicy};
pub use gguf::{
    ggml_tensor_byte_len, ggml_type_supported, gguf_tensor_bytes, gguf_tensor_bytes_from_source,
    lower_llama_model, lower_llama_model_from_source, lower_llama_model_to_shards,
    lowered_llama_candidate, parse_gguf, parse_gguf_source, streamed_llama_thin_capsule,
    tensor_disposition, transcode_tensor, transcode_tensor_from_source, FileGgufSource,
    FileTensorShardStore, GgufByteSource, GgufConversionPlan, GgufError, GgufModel,
    GgufTensorDisposition, GgufTensorInfo, GgufTokenizer, GgufValue, GgufValueType, LlamaConfig,
    LlamaSpmPolicy, LlamaTensorBinding, LoweredLlamaModel, SliceGgufSource,
    StreamedLoweredLlamaModel,
    TensorShardRef, TensorShardSink, TranscodedTensor, GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC,
    GGUF_VERSION,
};
pub use importer::{ImporterRegistry, SourceImporter};
pub use package::{
    build_native_package, build_streamed_native_package, load_native_capability,
    verify_native_package, verify_native_package_with_shards, AssimilationIdentity, NativePackage,
    StreamedPackageSpec,
};
pub use sandbox::{ForgeSandbox, NativeValidationSandbox, SandboxReport};
pub use source::{
    AssetKind, Discovery, LicenseRecord, NativeAdapter, NativeCandidate, NativeSection,
    ProvenanceRecord, RegressionCase, RegressionProbe, SourcePackage,
};
pub use store::{CommitReceipt, NativeAssetStore};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssimilationError {
    InvalidSource,
    InvalidProvenance,
    LicenseDenied(String),
    SourceTooLarge,
    DuplicateImporter(String),
    DuplicateMediaType(String),
    MissingImporter(String),
    ImportRejected(String),
    InvalidCandidate(String),
    SandboxNotIsolated,
    SandboxRejected(String),
    RegressionMissing(String),
    Capsule(String),
    InvalidSignature,
    UntrustedSigner,
    InvalidPackage,
    VersionConflict,
    MissingAsset(String),
    MissingVersion { asset_id: String, version: u32 },
    Overflow,
}
