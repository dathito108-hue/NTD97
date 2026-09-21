#![forbid(unsafe_code)]

use crate::hash::{sha256, Digest};
use crate::{CapsuleVersion, NativeIntelligenceContract, SectionKind, NCC97_MAGIC};
use ntd_ir::IrVersion;

pub const HEADER_LEN: usize = 112;
pub const MANIFEST_LEN: usize = 56;
pub const INDEX_ENTRY_LEN: usize = 56;
const ALIGNMENT: usize = 8;
const CHUNK_FLAG_EXTERNAL: u16 = 0x0001;
const MANIFEST_FLAG_HAS_BASE: u8 = 0x01;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CapsuleKind {
    Full = 1,
    Thin = 2,
    State = 3,
}

impl TryFrom<u8> for CapsuleKind {
    type Error = CapsuleError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Full),
            2 => Ok(Self::Thin),
            3 => Ok(Self::State),
            other => Err(CapsuleError::InvalidCapsuleKind(other)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkSource {
    Embedded(Vec<u8>),
    External {
        logical_len: u64,
        hash: Digest,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkSpec {
    pub kind: SectionKind,
    pub source: ChunkSource,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapsuleBuilder {
    pub kind: CapsuleKind,
    pub capsule_id: [u8; 16],
    pub base_root: Option<Digest>,
    pub chunks: Vec<ChunkSpec>,
}

impl CapsuleBuilder {
    pub fn new(kind: CapsuleKind, capsule_id: [u8; 16]) -> Self {
        Self {
            kind,
            capsule_id,
            base_root: None,
            chunks: Vec::new(),
        }
    }

    pub fn with_base_root(mut self, base_root: Digest) -> Self {
        self.base_root = Some(base_root);
        self
    }

    pub fn push_embedded(&mut self, kind: SectionKind, bytes: Vec<u8>) {
        self.chunks.push(ChunkSpec {
            kind,
            source: ChunkSource::Embedded(bytes),
        });
    }

    pub fn push_external(&mut self, kind: SectionKind, logical_len: u64, hash: Digest) {
        self.chunks.push(ChunkSpec {
            kind,
            source: ChunkSource::External { logical_len, hash },
        });
    }

    pub fn write(&self) -> Result<Vec<u8>, CapsuleError> {
        if self.kind == CapsuleKind::Full
            && self
                .chunks
                .iter()
                .any(|chunk| matches!(chunk.source, ChunkSource::External { .. }))
        {
            return Err(CapsuleError::ExternalChunkInFullCapsule);
        }

        let chunk_count = u32::try_from(self.chunks.len()).map_err(|_| CapsuleError::Overflow)?;
        let manifest_offset = HEADER_LEN;
        let index_offset = manifest_offset
            .checked_add(MANIFEST_LEN)
            .ok_or(CapsuleError::Overflow)?;
        let index_len = self
            .chunks
            .len()
            .checked_mul(INDEX_ENTRY_LEN)
            .ok_or(CapsuleError::Overflow)?;
        let payload_offset = align_up(
            index_offset
                .checked_add(index_len)
                .ok_or(CapsuleError::Overflow)?,
            ALIGNMENT,
        )?;

        let manifest = encode_manifest(self.kind, self.capsule_id, self.base_root);

        let mut entries = Vec::with_capacity(self.chunks.len());
        let mut cursor = payload_offset;
        let mut file_len = payload_offset;

        for chunk in &self.chunks {
            match &chunk.source {
                ChunkSource::Embedded(bytes) => {
                    let logical_len =
                        u64::try_from(bytes.len()).map_err(|_| CapsuleError::Overflow)?;
                    entries.push(IndexEntry {
                        kind: chunk.kind,
                        flags: 0,
                        logical_len,
                        offset: u64::try_from(cursor).map_err(|_| CapsuleError::Overflow)?,
                        hash: sha256(bytes),
                    });
                    let end = cursor
                        .checked_add(bytes.len())
                        .ok_or(CapsuleError::Overflow)?;
                    file_len = file_len.max(end);
                    cursor = align_up(end, ALIGNMENT)?;
                }
                ChunkSource::External { logical_len, hash } => {
                    entries.push(IndexEntry {
                        kind: chunk.kind,
                        flags: CHUNK_FLAG_EXTERNAL,
                        logical_len: *logical_len,
                        offset: 0,
                        hash: *hash,
                    });
                }
            }
        }

        let index = encode_index(&entries);
        let root_hash = metadata_root(&manifest, &index);

        let mut out = vec![0u8; file_len];
        encode_header(
            &mut out[..HEADER_LEN],
            HeaderFields {
                contract: NativeIntelligenceContract::CURRENT,
                manifest_offset: u64::try_from(manifest_offset)
                    .map_err(|_| CapsuleError::Overflow)?,
                manifest_len: u64::try_from(MANIFEST_LEN).map_err(|_| CapsuleError::Overflow)?,
                index_offset: u64::try_from(index_offset).map_err(|_| CapsuleError::Overflow)?,
                index_len: u64::try_from(index.len()).map_err(|_| CapsuleError::Overflow)?,
                payload_offset: u64::try_from(payload_offset)
                    .map_err(|_| CapsuleError::Overflow)?,
                chunk_count,
                root_hash,
            },
        );

        out[manifest_offset..manifest_offset + MANIFEST_LEN].copy_from_slice(&manifest);
        out[index_offset..index_offset + index.len()].copy_from_slice(&index);

        for (spec, entry) in self.chunks.iter().zip(entries.iter()) {
            if let ChunkSource::Embedded(bytes) = &spec.source {
                let start = usize::try_from(entry.offset).map_err(|_| CapsuleError::Overflow)?;
                let end = start
                    .checked_add(bytes.len())
                    .ok_or(CapsuleError::Overflow)?;
                out[start..end].copy_from_slice(bytes);
            }
        }

        Ok(out)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkStorageView<'a> {
    Embedded(&'a [u8]),
    External,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkView<'a> {
    pub kind: SectionKind,
    pub logical_len: u64,
    pub hash: Digest,
    pub storage: ChunkStorageView<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapsuleView<'a> {
    pub contract: NativeIntelligenceContract,
    pub kind: CapsuleKind,
    pub capsule_id: [u8; 16],
    pub base_root: Option<Digest>,
    pub root_hash: Digest,
    pub chunks: Vec<ChunkView<'a>>,
}

impl<'a> CapsuleView<'a> {
    pub fn read(bytes: &'a [u8]) -> Result<Self, CapsuleError> {
        let header = decode_header(bytes)?;

        if !NativeIntelligenceContract::CURRENT.can_read(header.contract) {
            return Err(CapsuleError::UnsupportedContract(header.contract));
        }

        let expected_index_len = usize::try_from(header.chunk_count)
            .map_err(|_| CapsuleError::Overflow)?
            .checked_mul(INDEX_ENTRY_LEN)
            .ok_or(CapsuleError::Overflow)?;
        let expected_index_offset = HEADER_LEN
            .checked_add(MANIFEST_LEN)
            .ok_or(CapsuleError::Overflow)?;
        let expected_payload_offset = align_up(
            expected_index_offset
                .checked_add(expected_index_len)
                .ok_or(CapsuleError::Overflow)?,
            ALIGNMENT,
        )?;

        if header.manifest_offset != u64::try_from(HEADER_LEN).map_err(|_| CapsuleError::Overflow)?
            || header.manifest_len
                != u64::try_from(MANIFEST_LEN).map_err(|_| CapsuleError::Overflow)?
            || header.index_offset
                != u64::try_from(expected_index_offset).map_err(|_| CapsuleError::Overflow)?
            || header.index_len
                != u64::try_from(expected_index_len).map_err(|_| CapsuleError::Overflow)?
            || header.payload_offset
                != u64::try_from(expected_payload_offset).map_err(|_| CapsuleError::Overflow)?
        {
            return Err(CapsuleError::NonCanonicalLayout);
        }

        let manifest_range = checked_range(
            bytes.len(),
            header.manifest_offset,
            header.manifest_len,
        )?;
        let index_range = checked_range(bytes.len(), header.index_offset, header.index_len)?;

        let manifest_bytes = &bytes[manifest_range.clone()];
        let index_bytes = &bytes[index_range.clone()];

        if metadata_root(manifest_bytes, index_bytes) != header.root_hash {
            return Err(CapsuleError::MetadataIntegrityMismatch);
        }

        let manifest = decode_manifest(manifest_bytes)?;
        let entries = decode_index(index_bytes, header.chunk_count)?;

        let mut chunks = Vec::with_capacity(entries.len());
        let mut expected_embedded_offset = expected_payload_offset;
        let mut expected_file_len = expected_payload_offset;

        for entry in entries {
            if entry.flags & !CHUNK_FLAG_EXTERNAL != 0 {
                return Err(CapsuleError::InvalidChunkFlags(entry.flags));
            }

            if entry.flags & CHUNK_FLAG_EXTERNAL != 0 {
                if entry.offset != 0 {
                    return Err(CapsuleError::InvalidIndex);
                }

                if manifest.kind == CapsuleKind::Full {
                    return Err(CapsuleError::ExternalChunkInFullCapsule);
                }

                chunks.push(ChunkView {
                    kind: entry.kind,
                    logical_len: entry.logical_len,
                    hash: entry.hash,
                    storage: ChunkStorageView::External,
                });
                continue;
            }

            let offset = usize::try_from(entry.offset).map_err(|_| CapsuleError::Overflow)?;
            if offset != expected_embedded_offset {
                return Err(CapsuleError::NonCanonicalLayout);
            }

            let payload_range = checked_range(bytes.len(), entry.offset, entry.logical_len)?;
            expected_file_len = payload_range.end;
            expected_embedded_offset = align_up(payload_range.end, ALIGNMENT)?;

            if sha256(&bytes[payload_range.clone()]) != entry.hash {
                return Err(CapsuleError::ChunkIntegrityMismatch(entry.kind));
            }

            chunks.push(ChunkView {
                kind: entry.kind,
                logical_len: entry.logical_len,
                hash: entry.hash,
                storage: ChunkStorageView::Embedded(&bytes[payload_range]),
            });
        }

        if bytes.len() != expected_file_len {
            return Err(CapsuleError::NonCanonicalLayout);
        }

        Ok(Self {
            contract: header.contract,
            kind: manifest.kind,
            capsule_id: manifest.capsule_id,
            base_root: manifest.base_root,
            root_hash: header.root_hash,
            chunks,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HeaderFields {
    contract: NativeIntelligenceContract,
    manifest_offset: u64,
    manifest_len: u64,
    index_offset: u64,
    index_len: u64,
    payload_offset: u64,
    chunk_count: u32,
    root_hash: Digest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ManifestFields {
    kind: CapsuleKind,
    capsule_id: [u8; 16],
    base_root: Option<Digest>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct IndexEntry {
    kind: SectionKind,
    flags: u16,
    logical_len: u64,
    offset: u64,
    hash: Digest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapsuleError {
    Truncated,
    InvalidMagic,
    InvalidHeader,
    InvalidManifest,
    InvalidIndex,
    InvalidCapsuleKind(u8),
    InvalidSectionKind(u16),
    InvalidChunkFlags(u16),
    UnsupportedContract(NativeIntelligenceContract),
    MetadataIntegrityMismatch,
    ChunkIntegrityMismatch(SectionKind),
    ExternalChunkInFullCapsule,
    NonCanonicalLayout,
    Overflow,
}

fn encode_header(dst: &mut [u8], fields: HeaderFields) {
    dst.fill(0);
    dst[0..6].copy_from_slice(&NCC97_MAGIC);
    put_u16(dst, 6, HEADER_LEN as u16);
    put_u16(dst, 8, fields.contract.capsule.major);
    put_u16(dst, 10, fields.contract.capsule.minor);
    put_u16(dst, 12, fields.contract.ir.major);
    put_u16(dst, 14, fields.contract.ir.minor);
    put_u32(dst, 16, 0);
    put_u64(dst, 20, fields.manifest_offset);
    put_u64(dst, 28, fields.manifest_len);
    put_u64(dst, 36, fields.index_offset);
    put_u64(dst, 44, fields.index_len);
    put_u64(dst, 52, fields.payload_offset);
    put_u32(dst, 60, fields.chunk_count);
    put_u32(dst, 64, 0);
    dst[68..100].copy_from_slice(&fields.root_hash);
}

fn decode_header(bytes: &[u8]) -> Result<HeaderFields, CapsuleError> {
    if bytes.len() < HEADER_LEN {
        return Err(CapsuleError::Truncated);
    }

    if bytes.get(0..6) != Some(NCC97_MAGIC.as_slice()) {
        return Err(CapsuleError::InvalidMagic);
    }

    let header_len = read_u16(bytes, 6)?;
    if usize::from(header_len) != HEADER_LEN {
        return Err(CapsuleError::InvalidHeader);
    }

    let contract = NativeIntelligenceContract {
        capsule: CapsuleVersion {
            major: read_u16(bytes, 8)?,
            minor: read_u16(bytes, 10)?,
        },
        ir: IrVersion {
            major: read_u16(bytes, 12)?,
            minor: read_u16(bytes, 14)?,
        },
    };

    let flags = read_u32(bytes, 16)?;
    let reserved = read_u32(bytes, 64)?;
    if flags != 0 || reserved != 0 || bytes[100..HEADER_LEN].iter().any(|byte| *byte != 0) {
        return Err(CapsuleError::InvalidHeader);
    }

    let mut root_hash = [0u8; 32];
    root_hash.copy_from_slice(&bytes[68..100]);

    Ok(HeaderFields {
        contract,
        manifest_offset: read_u64(bytes, 20)?,
        manifest_len: read_u64(bytes, 28)?,
        index_offset: read_u64(bytes, 36)?,
        index_len: read_u64(bytes, 44)?,
        payload_offset: read_u64(bytes, 52)?,
        chunk_count: read_u32(bytes, 60)?,
        root_hash,
    })
}

fn encode_manifest(
    kind: CapsuleKind,
    capsule_id: [u8; 16],
    base_root: Option<Digest>,
) -> [u8; MANIFEST_LEN] {
    let mut out = [0u8; MANIFEST_LEN];
    out[0] = kind as u8;
    if base_root.is_some() {
        out[1] = MANIFEST_FLAG_HAS_BASE;
    }
    out[8..24].copy_from_slice(&capsule_id);
    if let Some(root) = base_root {
        out[24..56].copy_from_slice(&root);
    }
    out
}

fn decode_manifest(bytes: &[u8]) -> Result<ManifestFields, CapsuleError> {
    if bytes.len() != MANIFEST_LEN {
        return Err(CapsuleError::InvalidManifest);
    }

    let kind = CapsuleKind::try_from(bytes[0])?;
    let flags = bytes[1];

    if flags & !MANIFEST_FLAG_HAS_BASE != 0 || bytes[2..8].iter().any(|byte| *byte != 0) {
        return Err(CapsuleError::InvalidManifest);
    }

    let mut capsule_id = [0u8; 16];
    capsule_id.copy_from_slice(&bytes[8..24]);

    let mut root = [0u8; 32];
    root.copy_from_slice(&bytes[24..56]);

    let base_root = if flags & MANIFEST_FLAG_HAS_BASE != 0 {
        Some(root)
    } else {
        if root.iter().any(|byte| *byte != 0) {
            return Err(CapsuleError::InvalidManifest);
        }
        None
    };

    Ok(ManifestFields {
        kind,
        capsule_id,
        base_root,
    })
}

fn encode_index(entries: &[IndexEntry]) -> Vec<u8> {
    let mut out = vec![0u8; entries.len() * INDEX_ENTRY_LEN];

    for (i, entry) in entries.iter().enumerate() {
        let base = i * INDEX_ENTRY_LEN;
        put_u16(&mut out, base, entry.kind as u16);
        put_u16(&mut out, base + 2, entry.flags);
        put_u32(&mut out, base + 4, 0);
        put_u64(&mut out, base + 8, entry.logical_len);
        put_u64(&mut out, base + 16, entry.offset);
        out[base + 24..base + 56].copy_from_slice(&entry.hash);
    }

    out
}

fn decode_index(bytes: &[u8], count: u32) -> Result<Vec<IndexEntry>, CapsuleError> {
    let count = usize::try_from(count).map_err(|_| CapsuleError::Overflow)?;
    if bytes.len() != count.checked_mul(INDEX_ENTRY_LEN).ok_or(CapsuleError::Overflow)? {
        return Err(CapsuleError::InvalidIndex);
    }

    let mut entries = Vec::with_capacity(count);

    for i in 0..count {
        let base = i * INDEX_ENTRY_LEN;
        let raw_kind = read_u16(bytes, base)?;
        let kind = SectionKind::try_from(raw_kind)?;
        let flags = read_u16(bytes, base + 2)?;

        if read_u32(bytes, base + 4)? != 0 {
            return Err(CapsuleError::InvalidIndex);
        }

        let mut hash = [0u8; 32];
        hash.copy_from_slice(&bytes[base + 24..base + 56]);

        entries.push(IndexEntry {
            kind,
            flags,
            logical_len: read_u64(bytes, base + 8)?,
            offset: read_u64(bytes, base + 16)?,
            hash,
        });
    }

    Ok(entries)
}

fn metadata_root(manifest: &[u8], index: &[u8]) -> Digest {
    let mut bytes = Vec::with_capacity(manifest.len() + index.len());
    bytes.extend_from_slice(manifest);
    bytes.extend_from_slice(index);
    sha256(&bytes)
}

fn checked_range(
    total_len: usize,
    offset: u64,
    len: u64,
) -> Result<std::ops::Range<usize>, CapsuleError> {
    let start = usize::try_from(offset).map_err(|_| CapsuleError::Overflow)?;
    let len = usize::try_from(len).map_err(|_| CapsuleError::Overflow)?;
    let end = start.checked_add(len).ok_or(CapsuleError::Overflow)?;

    if end > total_len {
        return Err(CapsuleError::Truncated);
    }

    Ok(start..end)
}

fn align_up(value: usize, alignment: usize) -> Result<usize, CapsuleError> {
    let add = alignment.checked_sub(1).ok_or(CapsuleError::Overflow)?;
    let value = value.checked_add(add).ok_or(CapsuleError::Overflow)?;
    Ok(value / alignment * alignment)
}

fn put_u16(dst: &mut [u8], offset: usize, value: u16) {
    dst[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(dst: &mut [u8], offset: usize, value: u32) {
    dst[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(dst: &mut [u8], offset: usize, value: u64) {
    dst[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn read_u16(src: &[u8], offset: usize) -> Result<u16, CapsuleError> {
    let bytes = src.get(offset..offset + 2).ok_or(CapsuleError::Truncated)?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn read_u32(src: &[u8], offset: usize) -> Result<u32, CapsuleError> {
    let bytes = src.get(offset..offset + 4).ok_or(CapsuleError::Truncated)?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn read_u64(src: &[u8], offset: usize) -> Result<u64, CapsuleError> {
    let bytes = src.get(offset..offset + 8).ok_or(CapsuleError::Truncated)?;
    Ok(u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5], bytes[6], bytes[7],
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::sha256;

    fn id() -> [u8; 16] {
        *b"NTD97-CAPSULE-01"
    }

    #[test]
    fn full_capsule_round_trips_deterministically() {
        let mut builder = CapsuleBuilder::new(CapsuleKind::Full, id());
        builder.push_embedded(SectionKind::Graph, b"ir-graph".to_vec());
        builder.push_embedded(SectionKind::Tokenizer, b"tokenizer".to_vec());

        let first = builder.write().expect("write first");
        let second = builder.write().expect("write second");

        assert_eq!(first, second);

        let view = CapsuleView::read(&first).expect("read");
        assert_eq!(view.kind, CapsuleKind::Full);
        assert_eq!(view.capsule_id, id());
        assert_eq!(view.chunks.len(), 2);

        assert_eq!(
            view.chunks[0].storage,
            ChunkStorageView::Embedded(b"ir-graph")
        );
        assert_eq!(
            view.chunks[1].storage,
            ChunkStorageView::Embedded(b"tokenizer")
        );
    }

    #[test]
    fn thin_capsule_can_reference_external_chunk() {
        let external_hash = sha256(b"immutable tensor shard");
        let mut builder = CapsuleBuilder::new(CapsuleKind::Thin, id());
        builder.push_external(SectionKind::Tensors, 22, external_hash);

        let bytes = builder.write().expect("write");
        let view = CapsuleView::read(&bytes).expect("read");

        assert_eq!(view.chunks.len(), 1);
        assert_eq!(view.chunks[0].hash, external_hash);
        assert_eq!(view.chunks[0].logical_len, 22);
        assert_eq!(view.chunks[0].storage, ChunkStorageView::External);
    }

    #[test]
    fn state_capsule_records_base_root() {
        let base_root = sha256(b"base capsule");
        let builder = CapsuleBuilder::new(CapsuleKind::State, id()).with_base_root(base_root);

        let bytes = builder.write().expect("write");
        let view = CapsuleView::read(&bytes).expect("read");

        assert_eq!(view.base_root, Some(base_root));
    }

    #[test]
    fn full_capsule_rejects_external_chunks() {
        let mut builder = CapsuleBuilder::new(CapsuleKind::Full, id());
        builder.push_external(SectionKind::Tensors, 100, sha256(b"external"));

        assert_eq!(
            builder.write(),
            Err(CapsuleError::ExternalChunkInFullCapsule)
        );
    }

    #[test]
    fn detects_payload_tampering() {
        let mut builder = CapsuleBuilder::new(CapsuleKind::Full, id());
        builder.push_embedded(SectionKind::Graph, b"graph".to_vec());

        let mut bytes = builder.write().expect("write");
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;

        assert_eq!(
            CapsuleView::read(&bytes),
            Err(CapsuleError::ChunkIntegrityMismatch(SectionKind::Graph))
        );
    }

    #[test]
    fn detects_metadata_tampering() {
        let mut builder = CapsuleBuilder::new(CapsuleKind::Full, id());
        builder.push_embedded(SectionKind::Graph, b"graph".to_vec());

        let mut bytes = builder.write().expect("write");
        bytes[HEADER_LEN + 8] ^= 0x01;

        assert_eq!(
            CapsuleView::read(&bytes),
            Err(CapsuleError::MetadataIntegrityMismatch)
        );
    }

    #[test]
    fn rejects_trailing_bytes_as_noncanonical() {
        let mut builder = CapsuleBuilder::new(CapsuleKind::Full, id());
        builder.push_embedded(SectionKind::Graph, b"graph".to_vec());
        let mut bytes = builder.write().expect("write");
        bytes.push(0);

        assert_eq!(
            CapsuleView::read(&bytes),
            Err(CapsuleError::NonCanonicalLayout)
        );
    }

    #[test]
    fn rejects_truncation() {
        let mut builder = CapsuleBuilder::new(CapsuleKind::Full, id());
        builder.push_embedded(SectionKind::Graph, b"graph".to_vec());
        let mut bytes = builder.write().expect("write");
        bytes.pop();

        assert_eq!(CapsuleView::read(&bytes), Err(CapsuleError::Truncated));
    }
}
