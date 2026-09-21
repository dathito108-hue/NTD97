#![forbid(unsafe_code)]

use sha2::{Digest, Sha256};

use crate::PcFabricError;

pub const DEFAULT_ARTIFACT_CHUNK: u32 = 64 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactDescriptor {
    pub transfer_id: u64,
    pub name: String,
    pub length: u64,
    pub sha256: [u8; 32],
    pub chunk_size: u32,
}

impl ArtifactDescriptor {
    pub fn validate(&self) -> Result<(), PcFabricError> {
        if self.transfer_id == 0
            || self.name.trim().is_empty()
            || self.chunk_size == 0
            || self.chunk_size > 4 * 1024 * 1024
        {
            return Err(PcFabricError::InvalidArtifact);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactChunk {
    pub transfer_id: u64,
    pub offset: u64,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactSender {
    descriptor: ArtifactDescriptor,
    bytes: Vec<u8>,
}

impl ArtifactSender {
    pub fn new(
        transfer_id: u64,
        name: impl Into<String>,
        bytes: Vec<u8>,
        chunk_size: u32,
    ) -> Result<Self, PcFabricError> {
        let name = name.into();
        let length = u64::try_from(bytes.len()).map_err(|_| PcFabricError::Overflow)?;
        let descriptor = ArtifactDescriptor {
            transfer_id,
            name,
            length,
            sha256: Sha256::digest(&bytes).into(),
            chunk_size,
        };
        descriptor.validate()?;
        Ok(Self { descriptor, bytes })
    }

    pub fn descriptor(&self) -> &ArtifactDescriptor {
        &self.descriptor
    }

    pub fn chunk_at(&self, offset: u64) -> Result<ArtifactChunk, PcFabricError> {
        if offset > self.descriptor.length {
            return Err(PcFabricError::InvalidArtifact);
        }
        let offset_usize = usize::try_from(offset).map_err(|_| PcFabricError::Overflow)?;
        let chunk_size =
            usize::try_from(self.descriptor.chunk_size).map_err(|_| PcFabricError::Overflow)?;
        let end = offset_usize
            .saturating_add(chunk_size)
            .min(self.bytes.len());
        if offset_usize > self.bytes.len() {
            return Err(PcFabricError::InvalidArtifact);
        }
        Ok(ArtifactChunk {
            transfer_id: self.descriptor.transfer_id,
            offset,
            data: self.bytes[offset_usize..end].to_vec(),
        })
    }

    pub fn chunks(&self) -> Result<Vec<ArtifactChunk>, PcFabricError> {
        let chunk_size =
            usize::try_from(self.descriptor.chunk_size).map_err(|_| PcFabricError::Overflow)?;
        let mut chunks = Vec::new();
        let mut offset = 0usize;
        while offset < self.bytes.len() {
            let end = offset.saturating_add(chunk_size).min(self.bytes.len());
            chunks.push(ArtifactChunk {
                transfer_id: self.descriptor.transfer_id,
                offset: u64::try_from(offset).map_err(|_| PcFabricError::Overflow)?,
                data: self.bytes[offset..end].to_vec(),
            });
            offset = end;
        }
        Ok(chunks)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactAssembler {
    descriptor: ArtifactDescriptor,
    bytes: Vec<u8>,
    next_offset: u64,
}

impl ArtifactAssembler {
    pub fn new(descriptor: ArtifactDescriptor) -> Result<Self, PcFabricError> {
        descriptor.validate()?;
        let capacity = usize::try_from(descriptor.length).map_err(|_| PcFabricError::Overflow)?;
        Ok(Self {
            descriptor,
            bytes: Vec::with_capacity(capacity),
            next_offset: 0,
        })
    }

    pub fn descriptor(&self) -> &ArtifactDescriptor {
        &self.descriptor
    }

    pub fn next_offset(&self) -> u64 {
        self.next_offset
    }

    pub fn push(&mut self, chunk: ArtifactChunk) -> Result<(), PcFabricError> {
        if chunk.transfer_id != self.descriptor.transfer_id || chunk.offset != self.next_offset {
            return Err(PcFabricError::ArtifactOutOfOrder);
        }
        if chunk.data.len()
            > usize::try_from(self.descriptor.chunk_size).map_err(|_| PcFabricError::Overflow)?
        {
            return Err(PcFabricError::InvalidArtifact);
        }

        let chunk_len = u64::try_from(chunk.data.len()).map_err(|_| PcFabricError::Overflow)?;
        let end = self
            .next_offset
            .checked_add(chunk_len)
            .ok_or(PcFabricError::Overflow)?;
        if end > self.descriptor.length {
            return Err(PcFabricError::InvalidArtifact);
        }

        self.bytes.extend_from_slice(&chunk.data);
        self.next_offset = end;
        Ok(())
    }

    pub fn finish(self) -> Result<Vec<u8>, PcFabricError> {
        if self.next_offset != self.descriptor.length {
            return Err(PcFabricError::InvalidArtifact);
        }
        let actual: [u8; 32] = Sha256::digest(&self.bytes).into();
        if actual != self.descriptor.sha256 {
            return Err(PcFabricError::ArtifactHashMismatch);
        }
        Ok(self.bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunked_artifact_round_trips_with_final_hash() {
        let bytes = (0..200_000u32)
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        let sender = ArtifactSender::new(7, "build.bin", bytes.clone(), 32 * 1024).expect("sender");
        let mut receiver = ArtifactAssembler::new(sender.descriptor().clone()).expect("receiver");

        for chunk in sender.chunks().expect("chunks") {
            receiver.push(chunk).expect("push");
        }

        assert_eq!(receiver.finish().expect("finish"), bytes);
    }

    #[test]
    fn tampered_artifact_is_rejected() {
        let sender = ArtifactSender::new(8, "result.txt", b"trusted".to_vec(), 4).expect("sender");
        let mut receiver = ArtifactAssembler::new(sender.descriptor().clone()).expect("receiver");
        let mut chunks = sender.chunks().expect("chunks");
        chunks[0].data[0] ^= 1;

        for chunk in chunks {
            receiver.push(chunk).expect("push");
        }

        assert_eq!(receiver.finish(), Err(PcFabricError::ArtifactHashMismatch));
    }
}
