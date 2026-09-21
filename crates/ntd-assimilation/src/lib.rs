#![forbid(unsafe_code)]

mod forge;
mod importer;
mod package;
mod sandbox;
mod source;
mod store;

pub use forge::{CapabilityForge, ForgePolicy};
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
