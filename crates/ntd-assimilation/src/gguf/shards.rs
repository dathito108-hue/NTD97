#![forbid(unsafe_code)]

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use ntd_capsule::{encode_native_tensor, sha256, Digest, NativeTensor, TensorDescriptor};
use ntd_ir::ValueId;

use super::GgufError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorShardRef {
    pub descriptor: TensorDescriptor,
    pub graph_value: Option<ValueId>,
    pub logical_len: u64,
    pub hash: Digest,
}

pub trait TensorShardSink {
    fn store_tensor(&mut self, tensor: &NativeTensor) -> Result<TensorShardRef, GgufError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileTensorShardStore {
    root: PathBuf,
}

impl FileTensorShardStore {
    pub fn open(root: impl AsRef<Path>) -> Result<Self, GgufError> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root).map_err(|error| GgufError::Io(error.to_string()))?;
        if !root.is_dir() {
            return Err(GgufError::Io("tensor shard root is not a directory".into()));
        }
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn shard_path(&self, hash: &Digest) -> PathBuf {
        self.root.join(format!("{}.ntp97", digest_hex(hash)))
    }

    pub fn read_verified(&self, reference: &TensorShardRef) -> Result<Vec<u8>, GgufError> {
        let bytes = fs::read(self.shard_path(&reference.hash))
            .map_err(|error| GgufError::Io(error.to_string()))?;
        let actual_len = u64::try_from(bytes.len()).map_err(|_| GgufError::LimitExceeded)?;
        if actual_len != reference.logical_len || sha256(&bytes) != reference.hash {
            return Err(GgufError::InvalidTensor);
        }
        Ok(bytes)
    }

    pub fn read_hash_verified(
        &self,
        hash: &Digest,
        logical_len: u64,
    ) -> Result<Vec<u8>, GgufError> {
        let bytes = fs::read(self.shard_path(hash))
            .map_err(|error| GgufError::Io(error.to_string()))?;
        let actual_len = u64::try_from(bytes.len()).map_err(|_| GgufError::LimitExceeded)?;
        if actual_len != logical_len || sha256(&bytes) != *hash {
            return Err(GgufError::InvalidTensor);
        }
        Ok(bytes)
    }

    fn persist_content_addressed(&self, hash: Digest, bytes: &[u8]) -> Result<PathBuf, GgufError> {
        let destination = self.shard_path(&hash);
        if destination.exists() {
            let existing =
                fs::read(&destination).map_err(|error| GgufError::Io(error.to_string()))?;
            if existing.len() != bytes.len() || sha256(&existing) != hash {
                return Err(GgufError::InvalidTensor);
            }
            return Ok(destination);
        }

        let temp = self
            .root
            .join(format!(".{}.{}.tmp", digest_hex(&hash), std::process::id()));
        let mut file = match OpenOptions::new().write(true).create_new(true).open(&temp) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                fs::remove_file(&temp).map_err(|remove| GgufError::Io(remove.to_string()))?;
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&temp)
                    .map_err(|open| GgufError::Io(open.to_string()))?
            }
            Err(error) => return Err(GgufError::Io(error.to_string())),
        };

        let write_result = (|| -> Result<(), GgufError> {
            file.write_all(bytes)
                .map_err(|error| GgufError::Io(error.to_string()))?;
            file.sync_all()
                .map_err(|error| GgufError::Io(error.to_string()))?;
            Ok(())
        })();
        if let Err(error) = write_result {
            let _ = fs::remove_file(&temp);
            return Err(error);
        }
        drop(file);

        match fs::rename(&temp, &destination) {
            Ok(()) => Ok(destination),
            Err(error) if destination.exists() => {
                let _ = fs::remove_file(&temp);
                let existing =
                    fs::read(&destination).map_err(|read| GgufError::Io(read.to_string()))?;
                if existing.len() == bytes.len() && sha256(&existing) == hash {
                    Ok(destination)
                } else {
                    Err(GgufError::Io(error.to_string()))
                }
            }
            Err(error) => {
                let _ = fs::remove_file(&temp);
                Err(GgufError::Io(error.to_string()))
            }
        }
    }
}

impl TensorShardSink for FileTensorShardStore {
    fn store_tensor(&mut self, tensor: &NativeTensor) -> Result<TensorShardRef, GgufError> {
        let encoded = encode_native_tensor(tensor)
            .map_err(|error| GgufError::NativeLowering(format!("NTP97 shard: {error:?}")))?;
        let hash = sha256(&encoded);
        self.persist_content_addressed(hash, &encoded)?;
        Ok(TensorShardRef {
            descriptor: tensor.descriptor.clone(),
            graph_value: tensor.graph_value,
            logical_len: u64::try_from(encoded.len()).map_err(|_| GgufError::LimitExceeded)?,
            hash,
        })
    }
}

fn digest_hex(digest: &Digest) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use ntd_capsule::{decode_native_tensor, QuantizationMetadata};
    use ntd_ir::DType;

    use super::*;

    static NEXT_DIR_ID: AtomicU64 = AtomicU64::new(1);

    fn temp_root() -> PathBuf {
        let id = NEXT_DIR_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("ntd97-shards-{}-{id}", std::process::id()))
    }

    fn sample_tensor() -> NativeTensor {
        NativeTensor {
            descriptor: TensorDescriptor {
                id: 7,
                dtype: DType::F32,
                shape: vec![2],
                byte_len: 8,
            },
            graph_value: Some(ValueId(3)),
            quantization: QuantizationMetadata::None,
            payload: [1.0f32, -2.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        }
    }

    #[test]
    fn file_store_emits_content_addressed_ntp97_and_deduplicates() {
        let root = temp_root();
        let mut store = FileTensorShardStore::open(&root).expect("store");
        let tensor = sample_tensor();

        let first = store.store_tensor(&tensor).expect("first");
        let second = store.store_tensor(&tensor).expect("second");
        assert_eq!(first, second);

        let stored = store.read_verified(&first).expect("stored");
        assert_eq!(decode_native_tensor(&stored).expect("decode"), tensor);
        assert_eq!(
            fs::read_dir(&root).expect("list").count(),
            1,
            "identical shards must deduplicate"
        );

        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn file_store_detects_existing_content_tampering() {
        let root = temp_root();
        let mut store = FileTensorShardStore::open(&root).expect("store");
        let tensor = sample_tensor();
        let reference = store.store_tensor(&tensor).expect("store tensor");

        fs::write(store.shard_path(&reference.hash), b"tampered").expect("tamper");
        assert_eq!(store.store_tensor(&tensor), Err(GgufError::InvalidTensor));

        fs::remove_dir_all(root).expect("cleanup");
    }
}
