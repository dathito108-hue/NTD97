#![forbid(unsafe_code)]

mod forge;
mod gguf;
mod importer;
mod package;
mod sandbox;
mod source;
mod store;

pub use forge::{CapabilityForge, ForgePolicy};
pub use gguf::{
    parse_gguf, tensor_disposition, GgufConversionPlan, GgufError, GgufModel,
    GgufTensorDisposition, GgufTensorInfo, GgufTokenizer, GgufValue, GgufValueType,
    GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC, GGUF_VERSION,
};
pub use importer::{ImporterRegistry, SourceImporter};
pub use package::{
    build_native_package, load_native_capability, verify_native_package, AssimilationIdentity,
    NativePackage,
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
