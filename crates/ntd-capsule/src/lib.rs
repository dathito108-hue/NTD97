#![forbid(unsafe_code)]

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unknown_major_version() {
        let current = CapsuleVersion::CURRENT;
        assert!(!current.can_read(CapsuleVersion { major: 1, minor: 0 }));
    }
}
