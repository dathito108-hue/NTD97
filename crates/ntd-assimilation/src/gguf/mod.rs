#![forbid(unsafe_code)]

mod lower;
mod reader;
mod shards;
mod storage;
mod transcode;

use std::collections::BTreeMap;

pub use lower::{
    lower_llama_model, lower_llama_model_from_source, lower_llama_model_to_shards,
    lowered_llama_candidate, streamed_llama_thin_capsule, LlamaConfig, LlamaTensorBinding,
    LoweredLlamaModel, StreamedLoweredLlamaModel,
};
pub use reader::{parse_gguf, parse_gguf_source};
pub use shards::{FileTensorShardStore, TensorShardRef, TensorShardSink};
pub use storage::{FileGgufSource, GgufByteSource, SliceGgufSource};
pub use transcode::{
    ggml_tensor_byte_len, ggml_type_supported, gguf_tensor_bytes, gguf_tensor_bytes_from_source,
    transcode_tensor, transcode_tensor_from_source, TranscodedTensor,
};

pub const GGUF_MAGIC: [u8; 4] = *b"GGUF";
pub const GGUF_VERSION: u32 = 3;
pub const GGUF_DEFAULT_ALIGNMENT: u32 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl TryFrom<u32> for GgufValueType {
    type Error = GgufError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Uint8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::Uint16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::Uint32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::Uint64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            other => Err(GgufError::UnsupportedValueType(other)),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array {
        element_type: GgufValueType,
        values: Vec<GgufValue>,
    },
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GgufValue {
    pub fn as_string(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::Uint8(value) => Some(u32::from(*value)),
            Self::Uint16(value) => Some(u32::from(*value)),
            Self::Uint32(value) => Some(*value),
            Self::Uint64(value) => u32::try_from(*value).ok(),
            Self::Int8(value) => u32::try_from(*value).ok(),
            Self::Int16(value) => u32::try_from(*value).ok(),
            Self::Int32(value) => u32::try_from(*value).ok(),
            Self::Int64(value) => u32::try_from(*value).ok(),
            _ => None,
        }
    }

    pub fn as_string_array(&self) -> Option<Vec<&str>> {
        let Self::Array {
            element_type: GgufValueType::String,
            values,
        } = self
        else {
            return None;
        };
        values
            .iter()
            .map(GgufValue::as_string)
            .collect::<Option<Vec<_>>>()
    }

    pub fn as_f32_array(&self) -> Option<Vec<f32>> {
        let Self::Array {
            element_type: GgufValueType::Float32,
            values,
        } = self
        else {
            return None;
        };
        values
            .iter()
            .map(|value| match value {
                GgufValue::Float32(value) => Some(*value),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
    }

    pub fn as_i32_array(&self) -> Option<Vec<i32>> {
        let Self::Array {
            element_type: GgufValueType::Int32,
            values,
        } = self
        else {
            return None;
        };
        values
            .iter()
            .map(|value| match value {
                GgufValue::Int32(value) => Some(*value),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(value) => Some(*value),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_type: u32,
    pub data_offset: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgufTokenizer {
    pub model: String,
    pub tokens: Vec<Vec<u8>>,
    pub scores: Option<Vec<f32>>,
    pub token_types: Option<Vec<i32>>,
    pub merges: Vec<String>,
    pub pre_tokenizer: Option<String>,
    pub add_space_prefix: Option<bool>,
    pub remove_extra_whitespaces: Option<bool>,
    pub normalizer_lowercase: Option<bool>,
    pub normalizer_strip_accents: Option<bool>,
    pub has_precompiled_charsmap: bool,
    pub add_bos_token: Option<bool>,
    pub add_eos_token: Option<bool>,
    pub bos_token: Option<u32>,
    pub eos_token: Option<u32>,
    pub unknown_token: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlamaSpmPolicy {
    pub add_space_prefix: bool,
    pub add_bos_token: bool,
    pub add_eos_token: bool,
    pub inherited_defaults: bool,
}

impl GgufTokenizer {
    pub fn resolved_llama_spm_policy(&self) -> Option<LlamaSpmPolicy> {
        if self.model != "llama" {
            return None;
        }
        Some(LlamaSpmPolicy {
            add_space_prefix: self.add_space_prefix.unwrap_or(true),
            add_bos_token: self.add_bos_token.unwrap_or(true),
            add_eos_token: self.add_eos_token.unwrap_or(false),
            inherited_defaults: self.add_space_prefix.is_none()
                || self.add_bos_token.is_none()
                || self.add_eos_token.is_none(),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgufModel {
    pub version: u32,
    pub alignment: u32,
    pub metadata: BTreeMap<String, GgufValue>,
    pub tensors: Vec<GgufTensorInfo>,
    pub data_offset: u64,
    pub file_len: u64,
}

impl GgufModel {
    pub fn parse(bytes: &[u8]) -> Result<Self, GgufError> {
        parse_gguf(bytes)
    }

    pub fn parse_source(source: &dyn GgufByteSource) -> Result<Self, GgufError> {
        parse_gguf_source(source)
    }

    pub fn architecture(&self) -> Result<&str, GgufError> {
        self.metadata
            .get("general.architecture")
            .and_then(GgufValue::as_string)
            .filter(|value| !value.trim().is_empty())
            .ok_or(GgufError::MissingMetadata("general.architecture"))
    }

    pub fn model_name(&self) -> Option<&str> {
        self.metadata
            .get("general.name")
            .and_then(GgufValue::as_string)
            .filter(|value| !value.trim().is_empty())
    }

    pub fn tokenizer(&self) -> Result<GgufTokenizer, GgufError> {
        let model = self
            .metadata
            .get("tokenizer.ggml.model")
            .and_then(GgufValue::as_string)
            .filter(|value| !value.trim().is_empty())
            .ok_or(GgufError::MissingMetadata("tokenizer.ggml.model"))?
            .to_owned();
        let tokens = self
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(GgufValue::as_string_array)
            .ok_or(GgufError::MissingMetadata("tokenizer.ggml.tokens"))?
            .into_iter()
            .map(|token| token.as_bytes().to_vec())
            .collect::<Vec<_>>();
        if tokens.is_empty() {
            return Err(GgufError::InvalidTokenizer);
        }

        let optional_u32 = |key: &'static str| -> Result<Option<u32>, GgufError> {
            match self.metadata.get(key) {
                Some(value) => value
                    .as_u32()
                    .map(Some)
                    .ok_or(GgufError::InvalidMetadataType(key)),
                None => Ok(None),
            }
        };
        let optional_bool = |key: &'static str| -> Result<Option<bool>, GgufError> {
            match self.metadata.get(key) {
                Some(value) => value
                    .as_bool()
                    .map(Some)
                    .ok_or(GgufError::InvalidMetadataType(key)),
                None => Ok(None),
            }
        };
        let optional_f32_array = |key: &'static str| -> Result<Option<Vec<f32>>, GgufError> {
            match self.metadata.get(key) {
                Some(value) => value
                    .as_f32_array()
                    .map(Some)
                    .ok_or(GgufError::InvalidMetadataType(key)),
                None => Ok(None),
            }
        };
        let optional_i32_array = |key: &'static str| -> Result<Option<Vec<i32>>, GgufError> {
            match self.metadata.get(key) {
                Some(value) => value
                    .as_i32_array()
                    .map(Some)
                    .ok_or(GgufError::InvalidMetadataType(key)),
                None => Ok(None),
            }
        };

        let scores = optional_f32_array("tokenizer.ggml.scores")?;
        if scores.as_ref().is_some_and(|scores| {
            scores.len() != tokens.len() || scores.iter().any(|score| !score.is_finite())
        }) {
            return Err(GgufError::InvalidTokenizer);
        }
        let token_types = optional_i32_array("tokenizer.ggml.token_type")?;
        if token_types
            .as_ref()
            .is_some_and(|types| types.len() != tokens.len())
        {
            return Err(GgufError::InvalidTokenizer);
        }
        let merges = match self.metadata.get("tokenizer.ggml.merges") {
            Some(value) => value
                .as_string_array()
                .ok_or(GgufError::InvalidMetadataType("tokenizer.ggml.merges"))?
                .into_iter()
                .map(ToOwned::to_owned)
                .collect(),
            None => Vec::new(),
        };
        let pre_tokenizer = match self.metadata.get("tokenizer.ggml.pre") {
            Some(value) => Some(
                value
                    .as_string()
                    .ok_or(GgufError::InvalidMetadataType("tokenizer.ggml.pre"))?
                    .to_owned(),
            ),
            None => None,
        };

        let tokenizer = GgufTokenizer {
            model,
            tokens,
            scores,
            token_types,
            merges,
            pre_tokenizer,
            add_space_prefix: optional_bool("tokenizer.ggml.add_space_prefix")?,
            remove_extra_whitespaces: optional_bool("tokenizer.ggml.remove_extra_whitespaces")?,
            normalizer_lowercase: optional_bool("tokenizer.ggml.normalizer.lowercase")?,
            normalizer_strip_accents: optional_bool("tokenizer.ggml.normalizer.strip_accents")?,
            has_precompiled_charsmap: self
                .metadata
                .contains_key("tokenizer.ggml.precompiled_charsmap"),
            add_bos_token: optional_bool("tokenizer.ggml.add_bos_token")?,
            add_eos_token: optional_bool("tokenizer.ggml.add_eos_token")?,
            bos_token: optional_u32("tokenizer.ggml.bos_token_id")?,
            eos_token: optional_u32("tokenizer.ggml.eos_token_id")?,
            unknown_token: optional_u32("tokenizer.ggml.unknown_token_id")?,
        };
        let vocab = u32::try_from(tokenizer.tokens.len()).map_err(|_| GgufError::LimitExceeded)?;
        for token in [
            tokenizer.bos_token,
            tokenizer.eos_token,
            tokenizer.unknown_token,
        ]
        .into_iter()
        .flatten()
        {
            if token >= vocab {
                return Err(GgufError::InvalidTokenizer);
            }
        }
        Ok(tokenizer)
    }
}

fn llama_spm_blocker(tokenizer: &GgufTokenizer) -> Option<String> {
    if tokenizer.scores.is_none() || tokenizer.token_types.is_none() {
        return Some(
            "llama tokenizer metadata is missing scores/token types required for native SentencePiece BPE semantics"
                .into(),
        );
    }
    if tokenizer
        .pre_tokenizer
        .as_deref()
        .is_some_and(|pre| pre != "default")
    {
        return Some(format!(
            "llama tokenizer pre-tokenizer '{}' is not supported by native SPM execution",
            tokenizer.pre_tokenizer.as_deref().unwrap_or_default()
        ));
    }
    if tokenizer.remove_extra_whitespaces == Some(true)
        || tokenizer.normalizer_lowercase == Some(true)
        || tokenizer.normalizer_strip_accents == Some(true)
        || tokenizer.has_precompiled_charsmap
    {
        return Some(
            "llama tokenizer requires normalization semantics not yet represented by the native SPM contract"
                .into(),
        );
    }
    None
}

fn gpt2_bpe_blocker(tokenizer: &GgufTokenizer) -> Option<String> {
    if tokenizer.merges.is_empty() {
        return Some(
            "gpt2 tokenizer metadata is missing merge ranks required for native BPE semantics"
                .into(),
        );
    }
    match tokenizer.pre_tokenizer.as_deref() {
        Some("gpt-2") => {}
        Some(pre) => {
            return Some(format!(
                "gpt2 pre-tokenizer '{pre}' is not the canonical GPT-2 regex supported by NTD97"
            ));
        }
        None => {
            return Some(
                "gpt2 tokenizer is missing tokenizer.ggml.pre required to select native pre-tokenizer semantics"
                    .into(),
            );
        }
    }
    if tokenizer.add_bos_token.is_none() || tokenizer.add_eos_token.is_none() {
        return Some("gpt2 tokenizer is missing explicit add-BOS/add-EOS policy metadata".into());
    }
    if tokenizer.add_space_prefix == Some(true)
        || tokenizer.remove_extra_whitespaces == Some(true)
        || tokenizer.normalizer_lowercase == Some(true)
        || tokenizer.normalizer_strip_accents == Some(true)
        || tokenizer.has_precompiled_charsmap
    {
        return Some(
            "gpt2 tokenizer requires normalization semantics outside canonical native GPT-2 BPE"
                .into(),
        );
    }
    None
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufTensorDisposition {
    NativeF32,
    NativeF16,
    NativeBf16,
    SupportedTranscode(u32),
    Unsupported(u32),
}

pub fn tensor_disposition(ggml_type: u32) -> GgufTensorDisposition {
    match ggml_type {
        0 => GgufTensorDisposition::NativeF32,
        1 => GgufTensorDisposition::NativeF16,
        30 => GgufTensorDisposition::NativeBf16,
        2 | 8 | 12 | 13 | 14 => GgufTensorDisposition::SupportedTranscode(ggml_type),
        other => GgufTensorDisposition::Unsupported(other),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufConversionPlan {
    pub architecture: String,
    pub model_name: Option<String>,
    pub tokenizer_model: String,
    pub vocabulary_size: usize,
    pub tensor_count: usize,
    pub direct_tensor_count: usize,
    pub transcode_tensor_count: usize,
    pub unsupported_tensor_count: usize,
    pub alignment: u32,
    pub blockers: Vec<String>,
}

impl GgufConversionPlan {
    pub fn from_model(model: &GgufModel) -> Result<Self, GgufError> {
        let tokenizer = model.tokenizer()?;
        let architecture = model.architecture()?.to_owned();
        let direct_tensor_count = model
            .tensors
            .iter()
            .filter(|tensor| {
                matches!(
                    tensor_disposition(tensor.ggml_type),
                    GgufTensorDisposition::NativeF32
                        | GgufTensorDisposition::NativeF16
                        | GgufTensorDisposition::NativeBf16
                )
            })
            .count();
        let transcode_tensor_count = model
            .tensors
            .iter()
            .filter(|tensor| {
                matches!(
                    tensor_disposition(tensor.ggml_type),
                    GgufTensorDisposition::SupportedTranscode(_)
                )
            })
            .count();
        let unsupported_types = model
            .tensors
            .iter()
            .filter_map(|tensor| match tensor_disposition(tensor.ggml_type) {
                GgufTensorDisposition::Unsupported(kind) => Some(kind),
                _ => None,
            })
            .collect::<std::collections::BTreeSet<_>>();
        let unsupported_tensor_count = model
            .tensors
            .iter()
            .filter(|tensor| {
                matches!(
                    tensor_disposition(tensor.ggml_type),
                    GgufTensorDisposition::Unsupported(_)
                )
            })
            .count();

        let mut blockers = Vec::new();
        if architecture != "llama" {
            blockers.push(format!(
                "architecture '{architecture}' has no canonical NTD97 lowering yet"
            ));
        }
        match tokenizer.model.as_str() {
            "llama" => {
                if let Some(blocker) = llama_spm_blocker(&tokenizer) {
                    blockers.push(blocker);
                }
            }
            "gpt2" => {
                if let Some(blocker) = gpt2_bpe_blocker(&tokenizer) {
                    blockers.push(blocker);
                }
            }
            _ => blockers.push(format!(
                "tokenizer '{}' has no canonical NTD97 tokenizer lowering yet",
                tokenizer.model
            )),
        }
        if unsupported_tensor_count > 0 {
            blockers.push(format!(
                "{unsupported_tensor_count} tensors use unsupported GGML types {:?}",
                unsupported_types
            ));
        }
        blockers.push(
            "representative real-model source-vs-NIR97 semantic equivalence is required before activation"
                .into(),
        );

        Ok(Self {
            architecture,
            model_name: model.model_name().map(ToOwned::to_owned),
            tokenizer_model: tokenizer.model,
            vocabulary_size: tokenizer.tokens.len(),
            tensor_count: model.tensors.len(),
            direct_tensor_count,
            transcode_tensor_count,
            unsupported_tensor_count,
            alignment: model.alignment,
            blockers,
        })
    }

    pub fn activation_ready(&self) -> bool {
        self.blockers.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GgufError {
    Truncated,
    InvalidMagic,
    UnsupportedVersion(u32),
    UnsupportedValueType(u32),
    UnsupportedTensorType(u32),
    UnsupportedArchitecture(String),
    UnsupportedModelFeature(String),
    MissingTensor(String),
    InvalidModelConfig(&'static str),
    NativeLowering(String),
    InvalidBool(u8),
    InvalidUtf8,
    Io(String),
    InvalidArrayType,
    InvalidAlignment,
    InvalidTensorAlignment,
    InvalidTensor,
    InvalidTokenizer,
    DuplicateMetadata(String),
    DuplicateTensor(String),
    MissingMetadata(&'static str),
    InvalidMetadataType(&'static str),
    LimitExceeded,
    Overflow,
}
