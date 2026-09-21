#![forbid(unsafe_code)]

mod binary;
mod descriptors;
mod hash;
mod ir_codec;
mod native_tensor;
mod tokenizer;

pub use binary::{
    CapsuleBuilder, CapsuleError, CapsuleKind, CapsuleView, ChunkSource, ChunkSpec,
    ChunkStorageView, ChunkView, HEADER_LEN, INDEX_ENTRY_LEN, MANIFEST_LEN,
};
pub use descriptors::{
    decode_descriptor_frame, decode_tensor_descriptors, encode_descriptor_frame,
    encode_tensor_descriptors, DescriptorError, DescriptorFrame, DescriptorFrameKind,
    TensorDescriptor, DESCRIPTOR_FRAME_HEADER_LEN, DESCRIPTOR_FRAME_MAGIC, DESCRIPTOR_MAJOR,
    DESCRIPTOR_MINOR, TENSOR_DESCRIPTOR_HEADER_LEN, TENSOR_DESCRIPTOR_MAGIC,
    TENSOR_RECORD_HEADER_LEN,
};
pub use hash::{sha256, Digest};
pub use ir_codec::{
    decode_graph, decode_graph_section, encode_graph, push_graph_section, GraphSectionError,
    IrCodecError, IR_GRAPH_HEADER_LEN, IR_GRAPH_MAGIC, NODE_HEADER_LEN, VALUE_DECL_LEN,
};
pub use native_tensor::{
    decode_native_tensor, encode_native_tensor, load_native_program, push_native_tensor_shard,
    push_tensor_descriptor_section, ContentStore, MemoryContentStore, NativeProgram, NativeTensor,
    NativeTensorError, QuantizationMetadata, NATIVE_TENSOR_HEADER_LEN, NATIVE_TENSOR_MAGIC,
    NATIVE_TENSOR_MAJOR, NATIVE_TENSOR_MINOR, NO_GRAPH_BINDING,
};
pub use tokenizer::{
    decode_native_tokenizer, encode_native_tokenizer, load_native_generative_program,
    push_native_tokenizer_section, NativeGenerativeError, NativeGenerativeProgram,
    NativeGpt2PreTokenizer, NativeTokenizerDescriptor, NativeTokenizerError, NativeTokenizerModel,
    LEGACY_NATIVE_TOKENIZER_FORMAT, NATIVE_TOKENIZER_FORMAT, NATIVE_TOKENIZER_HEADER_LEN,
    NATIVE_TOKENIZER_MAGIC, NATIVE_TOKENIZER_MAJOR, NATIVE_TOKENIZER_MINOR,
    PREVIOUS_NATIVE_TOKENIZER_FORMAT, NO_SPECIAL_TOKEN,
};

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

impl TryFrom<u16> for SectionKind {
    type Error = CapsuleError;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Graph),
            2 => Ok(Self::Tensors),
            3 => Ok(Self::Tokenizer),
            4 => Ok(Self::Codecs),
            5 => Ok(Self::Router),
            6 => Ok(Self::Adapters),
            7 => Ok(Self::Capabilities),
            8 => Ok(Self::MemorySchema),
            9 => Ok(Self::MemoryState),
            10 => Ok(Self::DeviceProfiles),
            11 => Ok(Self::Provenance),
            12 => Ok(Self::Signatures),
            13 => Ok(Self::AssimilationLog),
            14 => Ok(Self::WorldStateSchema),
            15 => Ok(Self::ContinuityState),
            16 => Ok(Self::EmbodimentProfile),
            other => Err(CapsuleError::InvalidSectionKind(other)),
        }
    }
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

    #[test]
    fn section_kind_is_strictly_decoded() {
        assert_eq!(SectionKind::try_from(1), Ok(SectionKind::Graph));
        assert_eq!(
            SectionKind::try_from(999),
            Err(CapsuleError::InvalidSectionKind(999))
        );
    }
}
