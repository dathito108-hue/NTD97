#![forbid(unsafe_code)]

use std::cell::RefCell;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use super::GgufError;

pub trait GgufByteSource {
    fn byte_len(&self) -> Result<u64, GgufError>;
    fn read_exact_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, GgufError>;
}

#[derive(Debug, Clone, Copy)]
pub struct SliceGgufSource<'a> {
    bytes: &'a [u8],
}

impl<'a> SliceGgufSource<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }
}

impl GgufByteSource for SliceGgufSource<'_> {
    fn byte_len(&self) -> Result<u64, GgufError> {
        u64::try_from(self.bytes.len()).map_err(|_| GgufError::LimitExceeded)
    }

    fn read_exact_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, GgufError> {
        let start = usize::try_from(offset).map_err(|_| GgufError::LimitExceeded)?;
        let end = start.checked_add(len).ok_or(GgufError::Overflow)?;
        self.bytes
            .get(start..end)
            .map(<[u8]>::to_vec)
            .ok_or(GgufError::Truncated)
    }
}

#[derive(Debug)]
pub struct FileGgufSource {
    file: RefCell<File>,
    byte_len: u64,
}

impl FileGgufSource {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let file = File::open(path).map_err(|error| GgufError::Io(error.to_string()))?;
        Self::from_file(file)
    }

    pub fn from_file(file: File) -> Result<Self, GgufError> {
        let byte_len = file
            .metadata()
            .map_err(|error| GgufError::Io(error.to_string()))?
            .len();
        Ok(Self {
            file: RefCell::new(file),
            byte_len,
        })
    }
}

impl GgufByteSource for FileGgufSource {
    fn byte_len(&self) -> Result<u64, GgufError> {
        Ok(self.byte_len)
    }

    fn read_exact_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, GgufError> {
        let len_u64 = u64::try_from(len).map_err(|_| GgufError::LimitExceeded)?;
        let end = offset.checked_add(len_u64).ok_or(GgufError::Overflow)?;
        if end > self.byte_len {
            return Err(GgufError::Truncated);
        }

        let mut bytes = vec![0u8; len];
        let mut file = self.file.borrow_mut();
        file.seek(SeekFrom::Start(offset))
            .map_err(|error| GgufError::Io(error.to_string()))?;
        file.read_exact(&mut bytes)
            .map_err(|error| GgufError::Io(error.to_string()))?;
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static NEXT_FILE_ID: AtomicU64 = AtomicU64::new(1);

    #[test]
    fn slice_source_reads_bounded_ranges() {
        let source = SliceGgufSource::new(b"NTD97");
        assert_eq!(source.byte_len(), Ok(5));
        assert_eq!(source.read_exact_at(1, 3), Ok(b"TD9".to_vec()));
        assert_eq!(source.read_exact_at(4, 2), Err(GgufError::Truncated));
    }

    #[test]
    fn file_source_reads_ranges_without_materializing_the_file() {
        let id = NEXT_FILE_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "ntd97-gguf-source-{}-{id}.bin",
            std::process::id()
        ));
        std::fs::write(&path, b"0123456789").expect("write fixture");

        let source = FileGgufSource::open(&path).expect("open source");
        assert_eq!(source.byte_len(), Ok(10));
        assert_eq!(source.read_exact_at(3, 4), Ok(b"3456".to_vec()));
        assert_eq!(source.read_exact_at(9, 2), Err(GgufError::Truncated));

        std::fs::remove_file(path).expect("remove fixture");
    }
}
