#![forbid(unsafe_code)]

use ntd_ir::IrVersion;

pub const NCC97_MAGIC: [u8; 6] = *b"NCC97\0";
pub const NCC97_MAJOR: u16 = 0;
pub const NCC97_MINOR: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum SectionKind {
    Graph = 1,
    Tensors = 2,
    Tokenizer = 3,
    Codecs = 4,
    Router = 5,
    Adapters = 6,
    Capabilities = 7,
    MemorySchema = 8,
    MemoryState = 9,
    DeviceProfiles = 10,
    Provenance = 11,
    Signatures = 12,
    AssimilationLog = 13,
    WorldStateSchema = 14,
    ContinuityState = 15,
    EmbodimentProfile = 16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapsuleVersion {
    pub major: u16,
    pub minor: u16,
}

impl CapsuleVersion {
    pub const CURRENT: Self = Self {
        major: NCC97_MAJOR,
        minor: NCC97_MINOR,
    };

    pub fn can_read(self, other: Self) -> bool {
        self.major == other.major && other.minor <= self.minor
    }
}

/// Version contract recorded by an NCC97 manifest.
///
/// Phase 003A deliberately freezes the semantic contract only. The exact binary
/// header layout, offsets, encoding and reader/writer implementation belong to
/// Phase 003B.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeIntelligenceContract {
    pub capsule: CapsuleVersion,
    pub ir: IrVersion,
}

impl NativeIntelligenceContract {
    pub const CURRENT: Self = Self {
        capsule: CapsuleVersion::CURRENT,
        ir: IrVersion::CURRENT,
    };

    pub fn can_read(self, other: Self) -> bool {
        self.capsule.can_read(other.capsule) && self.ir.can_read(other.ir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unknown_capsule_major_version() {
        let current = CapsuleVersion::CURRENT;
        assert!(!current.can_read(CapsuleVersion { major: 1, minor: 0 }));
    }

    #[test]
    fn rejects_unknown_ir_major_version() {
        let current = NativeIntelligenceContract::CURRENT;
        assert!(!current.can_read(NativeIntelligenceContract {
            capsule: CapsuleVersion::CURRENT,
            ir: IrVersion { major: 1, minor: 0 },
        }));
    }
}
