#![forbid(unsafe_code)]

use ntd_ir::ValueId;

use crate::{CapsuleBuilder, SectionKind};

pub const NATIVE_GENERATIVE_MANIFEST_MAGIC: [u8; 6] = *b"NGM97\0";
pub const NATIVE_GENERATIVE_MANIFEST_HEADER_LEN: usize = 24;
pub const NATIVE_GENERATIVE_MANIFEST_MAJOR: u16 = 0;
pub const NATIVE_GENERATIVE_MANIFEST_MINOR: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeGenerativeManifest {
    pub token_input: ValueId,
    pub distribution_output: u32,
    pub vocabulary_size: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeGenerativeManifestError {
    Truncated,
    InvalidMagic,
    InvalidHeader,
    UnsupportedVersion { major: u16, minor: u16 },
    InvalidVocabulary,
    NonCanonicalEncoding,
}

pub fn encode_native_generative_manifest(
    manifest: NativeGenerativeManifest,
) -> Result<Vec<u8>, NativeGenerativeManifestError> {
    if manifest.vocabulary_size == 0 {
        return Err(NativeGenerativeManifestError::InvalidVocabulary);
    }

    let mut out = Vec::with_capacity(NATIVE_GENERATIVE_MANIFEST_HEADER_LEN);
    out.extend_from_slice(&NATIVE_GENERATIVE_MANIFEST_MAGIC);
    push_u16(&mut out, NATIVE_GENERATIVE_MANIFEST_HEADER_LEN as u16);
    push_u16(&mut out, NATIVE_GENERATIVE_MANIFEST_MAJOR);
    push_u16(&mut out, NATIVE_GENERATIVE_MANIFEST_MINOR);
    push_u32(&mut out, manifest.token_input.0);
    push_u32(&mut out, manifest.distribution_output);
    push_u32(&mut out, manifest.vocabulary_size);
    push_u16(&mut out, 0);
    Ok(out)
}

pub fn decode_native_generative_manifest(
    bytes: &[u8],
) -> Result<NativeGenerativeManifest, NativeGenerativeManifestError> {
    let mut cursor = Cursor::new(bytes);
    if cursor.take(6)? != NATIVE_GENERATIVE_MANIFEST_MAGIC.as_slice() {
        return Err(NativeGenerativeManifestError::InvalidMagic);
    }
    if usize::from(cursor.u16()?) != NATIVE_GENERATIVE_MANIFEST_HEADER_LEN {
        return Err(NativeGenerativeManifestError::InvalidHeader);
    }

    let major = cursor.u16()?;
    let minor = cursor.u16()?;
    if major != NATIVE_GENERATIVE_MANIFEST_MAJOR || minor > NATIVE_GENERATIVE_MANIFEST_MINOR {
        return Err(NativeGenerativeManifestError::UnsupportedVersion { major, minor });
    }

    let manifest = NativeGenerativeManifest {
        token_input: ValueId(cursor.u32()?),
        distribution_output: cursor.u32()?,
        vocabulary_size: cursor.u32()?,
    };
    if manifest.vocabulary_size == 0 {
        return Err(NativeGenerativeManifestError::InvalidVocabulary);
    }
    if cursor.u16()? != 0 || !cursor.is_finished() {
        return Err(NativeGenerativeManifestError::NonCanonicalEncoding);
    }
    Ok(manifest)
}

pub fn push_native_generative_manifest_section(
    builder: &mut CapsuleBuilder,
    manifest: NativeGenerativeManifest,
) -> Result<(), NativeGenerativeManifestError> {
    builder.push_embedded(
        SectionKind::GenerativeManifest,
        encode_native_generative_manifest(manifest)?,
    );
    Ok(())
}

fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], NativeGenerativeManifestError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(NativeGenerativeManifestError::Truncated)?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or(NativeGenerativeManifestError::Truncated)?;
        self.offset = end;
        Ok(bytes)
    }

    fn u16(&mut self) -> Result<u16, NativeGenerativeManifestError> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> Result<u32, NativeGenerativeManifestError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn is_finished(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_round_trips_canonically() {
        let manifest = NativeGenerativeManifest {
            token_input: ValueId(7),
            distribution_output: 1,
            vocabulary_size: 32_000,
        };
        let encoded = encode_native_generative_manifest(manifest).expect("encode");
        assert_eq!(encoded.len(), NATIVE_GENERATIVE_MANIFEST_HEADER_LEN);
        assert_eq!(
            decode_native_generative_manifest(&encoded).expect("decode"),
            manifest
        );
    }

    #[test]
    fn zero_vocabulary_fails_closed() {
        assert_eq!(
            encode_native_generative_manifest(NativeGenerativeManifest {
                token_input: ValueId(0),
                distribution_output: 0,
                vocabulary_size: 0,
            }),
            Err(NativeGenerativeManifestError::InvalidVocabulary)
        );
    }
}
