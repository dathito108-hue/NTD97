#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use crate::{
    decode_descriptor_frame, encode_descriptor_frame, load_native_program, sha256, CapsuleBuilder,
    CapsuleView, ChunkStorageView, ContentStore, DescriptorError, DescriptorFrame,
    DescriptorFrameKind, Digest, NativeProgram, NativeTensorError, SectionKind,
};

pub const NATIVE_TOKENIZER_FORMAT: &str = "ntd97.tokenizer.v2";
pub const LEGACY_NATIVE_TOKENIZER_FORMAT: &str = "ntd97.tokenizer.vocab.v1";
pub const NATIVE_TOKENIZER_MAGIC: [u8; 6] = *b"NTK97\0";
pub const NATIVE_TOKENIZER_HEADER_LEN: usize = 32;
pub const NATIVE_TOKENIZER_MAJOR: u16 = 0;
pub const NATIVE_TOKENIZER_MINOR: u16 = 2;
pub const NO_SPECIAL_TOKEN: u32 = u32::MAX;

const TOKENIZER_MODEL_VOCABULARY: u32 = 0;
const TOKENIZER_MODEL_LLAMA_SPM: u32 = 1;
const TOKENIZER_FLAG_ADD_SPACE_PREFIX: u32 = 1 << 0;
const TOKENIZER_FLAG_ADD_BOS: u32 = 1 << 1;
const TOKENIZER_FLAG_ADD_EOS: u32 = 1 << 2;
const TOKENIZER_KNOWN_FLAGS: u32 =
    TOKENIZER_FLAG_ADD_SPACE_PREFIX | TOKENIZER_FLAG_ADD_BOS | TOKENIZER_FLAG_ADD_EOS;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeTokenizerModel {
    Vocabulary,
    LlamaSpm {
        score_bits: Vec<u32>,
        token_types: Vec<i32>,
        add_space_prefix: bool,
        add_bos_token: bool,
        add_eos_token: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeTokenizerDescriptor {
    pub tokens: Vec<Vec<u8>>,
    pub bos_token: Option<u32>,
    pub eos_token: Option<u32>,
    pub unknown_token: Option<u32>,
    pub model: NativeTokenizerModel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeGenerativeProgram {
    pub program: NativeProgram,
    pub tokenizer: NativeTokenizerDescriptor,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeTokenizerError {
    Descriptor(DescriptorError),
    WrongDescriptorKind(DescriptorFrameKind),
    WrongFormat(String),
    Truncated,
    InvalidMagic,
    InvalidHeader,
    UnsupportedVersion { major: u16, minor: u16 },
    EmptyVocabulary,
    EmptyToken(u32),
    DuplicateToken { first: u32, second: u32 },
    InvalidSpecialToken(u32),
    InvalidTokenizerModel(u32),
    InvalidScoreCount { expected: usize, actual: usize },
    InvalidTokenTypeCount { expected: usize, actual: usize },
    InvalidTokenType { token: u32, token_type: i32 },
    NonFiniteScore(u32),
    NonCanonicalEncoding,
    Overflow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeGenerativeError {
    Program(NativeTensorError),
    Tokenizer(NativeTokenizerError),
    MissingTokenizer,
    MultipleTokenizers,
    MissingExternalContent(Digest),
    ExternalLengthMismatch { expected: u64, actual: u64 },
    ExternalIntegrityMismatch(Digest),
    Overflow,
}

pub fn encode_native_tokenizer(
    tokenizer: &NativeTokenizerDescriptor,
) -> Result<Vec<u8>, NativeTokenizerError> {
    validate_tokenizer(tokenizer)?;

    let token_count =
        u32::try_from(tokenizer.tokens.len()).map_err(|_| NativeTokenizerError::Overflow)?;
    let mut payload = Vec::new();
    payload.extend_from_slice(&NATIVE_TOKENIZER_MAGIC);
    push_u16(&mut payload, NATIVE_TOKENIZER_HEADER_LEN as u16);
    push_u16(&mut payload, NATIVE_TOKENIZER_MAJOR);
    push_u16(&mut payload, NATIVE_TOKENIZER_MINOR);
    push_u32(&mut payload, token_count);
    push_u32(
        &mut payload,
        tokenizer.bos_token.unwrap_or(NO_SPECIAL_TOKEN),
    );
    push_u32(
        &mut payload,
        tokenizer.eos_token.unwrap_or(NO_SPECIAL_TOKEN),
    );
    push_u32(
        &mut payload,
        tokenizer.unknown_token.unwrap_or(NO_SPECIAL_TOKEN),
    );
    let model_tag = match tokenizer.model {
        NativeTokenizerModel::Vocabulary => TOKENIZER_MODEL_VOCABULARY,
        NativeTokenizerModel::LlamaSpm { .. } => TOKENIZER_MODEL_LLAMA_SPM,
    };
    push_u32(&mut payload, model_tag);

    for token in &tokenizer.tokens {
        let len = u32::try_from(token.len()).map_err(|_| NativeTokenizerError::Overflow)?;
        push_u32(&mut payload, len);
        payload.extend_from_slice(token);
    }

    if let NativeTokenizerModel::LlamaSpm {
        score_bits,
        token_types,
        add_space_prefix,
        add_bos_token,
        add_eos_token,
    } = &tokenizer.model
    {
        let score_count =
            u32::try_from(score_bits.len()).map_err(|_| NativeTokenizerError::Overflow)?;
        push_u32(&mut payload, score_count);
        for score in score_bits {
            push_u32(&mut payload, *score);
        }

        let type_count =
            u32::try_from(token_types.len()).map_err(|_| NativeTokenizerError::Overflow)?;
        push_u32(&mut payload, type_count);
        for token_type in token_types {
            push_i32(&mut payload, *token_type);
        }

        let mut flags = 0u32;
        if *add_space_prefix {
            flags |= TOKENIZER_FLAG_ADD_SPACE_PREFIX;
        }
        if *add_bos_token {
            flags |= TOKENIZER_FLAG_ADD_BOS;
        }
        if *add_eos_token {
            flags |= TOKENIZER_FLAG_ADD_EOS;
        }
        push_u32(&mut payload, flags);
    }

    encode_descriptor_frame(&DescriptorFrame {
        kind: DescriptorFrameKind::Tokenizer,
        format: NATIVE_TOKENIZER_FORMAT.to_owned(),
        payload,
    })
    .map_err(NativeTokenizerError::Descriptor)
}

pub fn decode_native_tokenizer(
    bytes: &[u8],
) -> Result<NativeTokenizerDescriptor, NativeTokenizerError> {
    let frame = decode_descriptor_frame(bytes).map_err(NativeTokenizerError::Descriptor)?;
    if frame.kind != DescriptorFrameKind::Tokenizer {
        return Err(NativeTokenizerError::WrongDescriptorKind(frame.kind));
    }
    if frame.format != NATIVE_TOKENIZER_FORMAT && frame.format != LEGACY_NATIVE_TOKENIZER_FORMAT {
        return Err(NativeTokenizerError::WrongFormat(frame.format));
    }

    let mut cursor = Cursor::new(&frame.payload);
    if cursor.take(6)? != NATIVE_TOKENIZER_MAGIC.as_slice() {
        return Err(NativeTokenizerError::InvalidMagic);
    }
    if usize::from(cursor.u16()?) != NATIVE_TOKENIZER_HEADER_LEN {
        return Err(NativeTokenizerError::InvalidHeader);
    }

    let major = cursor.u16()?;
    let minor = cursor.u16()?;
    if major != NATIVE_TOKENIZER_MAJOR || minor > NATIVE_TOKENIZER_MINOR {
        return Err(NativeTokenizerError::UnsupportedVersion { major, minor });
    }

    let token_count = cursor.u32()?;
    let bos_token = decode_special(cursor.u32()?);
    let eos_token = decode_special(cursor.u32()?);
    let unknown_token = decode_special(cursor.u32()?);
    let model_tag = cursor.u32()?;

    if minor <= 1 && model_tag != TOKENIZER_MODEL_VOCABULARY {
        return Err(NativeTokenizerError::NonCanonicalEncoding);
    }

    let capacity = usize::try_from(token_count).map_err(|_| NativeTokenizerError::Overflow)?;
    let mut tokens = Vec::with_capacity(capacity);
    for _ in 0..token_count {
        let len = usize::try_from(cursor.u32()?).map_err(|_| NativeTokenizerError::Overflow)?;
        tokens.push(cursor.take(len)?.to_vec());
    }

    let model = if minor <= 1 {
        NativeTokenizerModel::Vocabulary
    } else {
        match model_tag {
            TOKENIZER_MODEL_VOCABULARY => NativeTokenizerModel::Vocabulary,
            TOKENIZER_MODEL_LLAMA_SPM => {
                let score_count =
                    usize::try_from(cursor.u32()?).map_err(|_| NativeTokenizerError::Overflow)?;
                let mut score_bits = Vec::with_capacity(score_count);
                for _ in 0..score_count {
                    score_bits.push(cursor.u32()?);
                }

                let type_count =
                    usize::try_from(cursor.u32()?).map_err(|_| NativeTokenizerError::Overflow)?;
                let mut token_types = Vec::with_capacity(type_count);
                for _ in 0..type_count {
                    token_types.push(cursor.i32()?);
                }

                let flags = cursor.u32()?;
                if flags & !TOKENIZER_KNOWN_FLAGS != 0 {
                    return Err(NativeTokenizerError::NonCanonicalEncoding);
                }

                NativeTokenizerModel::LlamaSpm {
                    score_bits,
                    token_types,
                    add_space_prefix: flags & TOKENIZER_FLAG_ADD_SPACE_PREFIX != 0,
                    add_bos_token: flags & TOKENIZER_FLAG_ADD_BOS != 0,
                    add_eos_token: flags & TOKENIZER_FLAG_ADD_EOS != 0,
                }
            }
            other => return Err(NativeTokenizerError::InvalidTokenizerModel(other)),
        }
    };

    if !cursor.is_finished() {
        return Err(NativeTokenizerError::NonCanonicalEncoding);
    }

    let tokenizer = NativeTokenizerDescriptor {
        tokens,
        bos_token,
        eos_token,
        unknown_token,
        model,
    };
    validate_tokenizer(&tokenizer)?;
    Ok(tokenizer)
}

pub fn push_native_tokenizer_section(
    builder: &mut CapsuleBuilder,
    tokenizer: &NativeTokenizerDescriptor,
) -> Result<(), NativeTokenizerError> {
    builder.push_embedded(SectionKind::Tokenizer, encode_native_tokenizer(tokenizer)?);
    Ok(())
}

pub fn load_native_generative_program<S: ContentStore>(
    capsule: &CapsuleView<'_>,
    store: &S,
) -> Result<NativeGenerativeProgram, NativeGenerativeError> {
    let program = load_native_program(capsule, store).map_err(NativeGenerativeError::Program)?;

    let tokenizer_chunks = capsule
        .chunks
        .iter()
        .filter(|chunk| chunk.kind == SectionKind::Tokenizer)
        .collect::<Vec<_>>();
    let chunk = match tokenizer_chunks.as_slice() {
        [] => return Err(NativeGenerativeError::MissingTokenizer),
        [chunk] => *chunk,
        _ => return Err(NativeGenerativeError::MultipleTokenizers),
    };

    let bytes = match chunk.storage {
        ChunkStorageView::Embedded(bytes) => bytes,
        ChunkStorageView::External => {
            let bytes = store
                .get(&chunk.hash)
                .ok_or(NativeGenerativeError::MissingExternalContent(chunk.hash))?;
            let actual = u64::try_from(bytes.len()).map_err(|_| NativeGenerativeError::Overflow)?;
            if actual != chunk.logical_len {
                return Err(NativeGenerativeError::ExternalLengthMismatch {
                    expected: chunk.logical_len,
                    actual,
                });
            }
            if sha256(bytes) != chunk.hash {
                return Err(NativeGenerativeError::ExternalIntegrityMismatch(chunk.hash));
            }
            bytes
        }
    };

    let tokenizer = decode_native_tokenizer(bytes).map_err(NativeGenerativeError::Tokenizer)?;

    Ok(NativeGenerativeProgram { program, tokenizer })
}

fn validate_tokenizer(tokenizer: &NativeTokenizerDescriptor) -> Result<(), NativeTokenizerError> {
    if tokenizer.tokens.is_empty() {
        return Err(NativeTokenizerError::EmptyVocabulary);
    }

    let mut seen = BTreeMap::new();
    for (index, token) in tokenizer.tokens.iter().enumerate() {
        let id = u32::try_from(index).map_err(|_| NativeTokenizerError::Overflow)?;
        if token.is_empty() {
            return Err(NativeTokenizerError::EmptyToken(id));
        }
        if let Some(first) = seen.insert(token.clone(), id) {
            return Err(NativeTokenizerError::DuplicateToken { first, second: id });
        }
    }

    let token_count =
        u32::try_from(tokenizer.tokens.len()).map_err(|_| NativeTokenizerError::Overflow)?;
    for special in [
        tokenizer.bos_token,
        tokenizer.eos_token,
        tokenizer.unknown_token,
    ]
    .into_iter()
    .flatten()
    {
        if special >= token_count {
            return Err(NativeTokenizerError::InvalidSpecialToken(special));
        }
    }
    Ok(())
}

fn decode_special(value: u32) -> Option<u32> {
    (value != NO_SPECIAL_TOKEN).then_some(value)
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

    fn take(&mut self, len: usize) -> Result<&'a [u8], NativeTokenizerError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(NativeTokenizerError::Overflow)?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or(NativeTokenizerError::Truncated)?;
        self.offset = end;
        Ok(bytes)
    }

    fn u16(&mut self) -> Result<u16, NativeTokenizerError> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> Result<u32, NativeTokenizerError> {
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

    fn sample_tokenizer() -> NativeTokenizerDescriptor {
        NativeTokenizerDescriptor {
            tokens: vec![
                b"a".to_vec(),
                b"b".to_vec(),
                b"<bos>".to_vec(),
                b"<eos>".to_vec(),
            ],
            bos_token: Some(2),
            eos_token: Some(3),
            unknown_token: None,
        }
    }

    #[test]
    fn tokenizer_descriptor_round_trips_deterministically() {
        let tokenizer = sample_tokenizer();
        let first = encode_native_tokenizer(&tokenizer).expect("encode first");
        let second = encode_native_tokenizer(&tokenizer).expect("encode second");
        assert_eq!(first, second);
        assert_eq!(decode_native_tokenizer(&first).expect("decode"), tokenizer);
    }

    #[test]
    fn invalid_special_token_is_rejected() {
        let mut tokenizer = sample_tokenizer();
        tokenizer.eos_token = Some(99);
        assert_eq!(
            encode_native_tokenizer(&tokenizer),
            Err(NativeTokenizerError::InvalidSpecialToken(99))
        );
    }
}
