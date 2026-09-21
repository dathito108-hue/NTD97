#![forbid(unsafe_code)]

mod asset;
mod backup;
mod crypto;
mod store;

pub use asset::{AssetClass, PortableAsset, RestoredSet};
pub use backup::{restore_backup, BackupKind, PortableBackup, PortableBackupBuilder};
pub use crypto::BackupKey;
pub use store::{EncryptedObject, SovereignObjectStore};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PortabilityError {
    InvalidAsset,
    InvalidCapsule(String),
    InvalidBackup,
    CryptoFailure,
    IntegrityMismatch,
    MissingObject([u8; 32]),
    DuplicateAsset,
    EmptyBackup,
    Overflow,
}
