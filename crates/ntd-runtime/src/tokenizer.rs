#![forbid(unsafe_code)]

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};

pub const TOKEN_TYPE_NORMAL: i32 = 1;
pub const TOKEN_TYPE_UNKNOWN: i32 = 2;
pub const TOKEN_TYPE_CONTROL: i32 = 3;
pub const TOKEN_TYPE_USER_DEFINED: i32 = 4;
pub const TOKEN_TYPE_UNUSED: i32 = 5;
pub const TOKEN_TYPE_BYTE: i32 = 6;

const SPM_SPACE: &[u8; 3] = b"\xE2\x96\x81";

pub trait TextTokenizer {
    fn vocab_size(&self) -> usize;
    fn eos_token(&self) -> Option<u32>;
    fn encode_text(&self, text: &str, add_special_tokens: bool)
        -> Result<Vec<u32>, TokenizerError>;
    fn decode_text(&self, token_ids: &[u32], skip_special: bool) -> Result<String, TokenizerError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VocabularyTokenizer {
    tokens: Vec<Vec<u8>>,
    bos_token: Option<u32>,
    eos_token: Option<u32>,
    unknown_token: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlamaSpmConfig {
    pub bos_token: Option<u32>,
    pub eos_token: Option<u32>,
    pub unknown_token: Option<u32>,
    pub add_space_prefix: bool,
    pub add_bos_token: bool,
    pub add_eos_token: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlamaSpmTokenizer {
    tokens: Vec<Vec<u8>>,
    score_bits: Vec<u32>,
    token_types: Vec<i32>,
    bos_token: Option<u32>,
    eos_token: Option<u32>,
    unknown_token: Option<u32>,
    add_space_prefix: bool,
    add_bos_token: bool,
    add_eos_token: bool,
    token_to_id: BTreeMap<Vec<u8>, u32>,
    byte_tokens: [Option<u32>; 256],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerError {
    EmptyVocabulary,
    EmptyToken(u32),
    DuplicateToken { first: u32, second: u32 },
    InvalidSpecialToken(u32),
    InvalidScoreCount { expected: usize, actual: usize },
    InvalidTokenTypeCount { expected: usize, actual: usize },
    InvalidTokenType { token: u32, token_type: i32 },
    NonFiniteScore(u32),
    MissingByteToken(u8),
    InvalidBpeMerge(String),
    DuplicateBpeMerge(String),
    MissingTokenPiece(Vec<u8>),
    InvalidGpt2ByteChar(char),
    UnmatchedInput(usize),
    InvalidTokenId(u32),
    InvalidUtf8,
    Overflow,
}

impl VocabularyTokenizer {
    pub fn new(
        tokens: Vec<Vec<u8>>,
        bos_token: Option<u32>,
        eos_token: Option<u32>,
        unknown_token: Option<u32>,
    ) -> Result<Self, TokenizerError> {
        validate_vocabulary(&tokens)?;
        validate_specials(&tokens, bos_token, eos_token, unknown_token)?;

        Ok(Self {
            tokens,
            bos_token,
            eos_token,
            unknown_token,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn bos_token(&self) -> Option<u32> {
        self.bos_token
    }

    pub fn eos_token(&self) -> Option<u32> {
        self.eos_token
    }

    pub fn unknown_token(&self) -> Option<u32> {
        self.unknown_token
    }

    pub fn tokens(&self) -> &[Vec<u8>] {
        &self.tokens
    }

    pub fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>, TokenizerError> {
        let bytes = text.as_bytes();
        let mut output = Vec::new();

        if add_bos {
            output.extend(self.bos_token);
        }

        let mut offset = 0usize;
        while offset < bytes.len() {
            let mut best: Option<(usize, u32)> = None;

            for (index, token) in self.tokens.iter().enumerate() {
                let id = u32::try_from(index).map_err(|_| TokenizerError::Overflow)?;
                if self.is_special(id) {
                    continue;
                }
                if !bytes[offset..].starts_with(token) {
                    continue;
                }

                match best {
                    Some((best_len, best_id))
                        if best_len > token.len() || (best_len == token.len() && best_id < id) => {}
                    _ => best = Some((token.len(), id)),
                }
            }

            if let Some((matched_len, id)) = best {
                output.push(id);
                offset = offset
                    .checked_add(matched_len)
                    .ok_or(TokenizerError::Overflow)?;
                continue;
            }

            if let Some(unknown) = self.unknown_token {
                output.push(unknown);
                offset = offset.checked_add(1).ok_or(TokenizerError::Overflow)?;
                continue;
            }

            return Err(TokenizerError::UnmatchedInput(offset));
        }

        Ok(output)
    }

    pub fn decode(&self, token_ids: &[u32], skip_special: bool) -> Result<String, TokenizerError> {
        let mut bytes = Vec::new();

        for id in token_ids {
            if skip_special && self.is_special(*id) {
                continue;
            }

            let index = usize::try_from(*id).map_err(|_| TokenizerError::InvalidTokenId(*id))?;
            let token = self
                .tokens
                .get(index)
                .ok_or(TokenizerError::InvalidTokenId(*id))?;
            bytes.extend_from_slice(token);
        }

        String::from_utf8(bytes).map_err(|_| TokenizerError::InvalidUtf8)
    }

    fn is_special(&self, id: u32) -> bool {
        self.bos_token == Some(id) || self.eos_token == Some(id) || self.unknown_token == Some(id)
    }
}

impl TextTokenizer for VocabularyTokenizer {
    fn vocab_size(&self) -> usize {
        VocabularyTokenizer::vocab_size(self)
    }

    fn eos_token(&self) -> Option<u32> {
        VocabularyTokenizer::eos_token(self)
    }

    fn encode_text(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Vec<u32>, TokenizerError> {
        self.encode(text, add_special_tokens)
    }

    fn decode_text(&self, token_ids: &[u32], skip_special: bool) -> Result<String, TokenizerError> {
        self.decode(token_ids, skip_special)
    }
}

impl LlamaSpmTokenizer {
    pub fn new(
        tokens: Vec<Vec<u8>>,
        score_bits: Vec<u32>,
        token_types: Vec<i32>,
        config: LlamaSpmConfig,
    ) -> Result<Self, TokenizerError> {
        validate_vocabulary(&tokens)?;
        validate_specials(
            &tokens,
            config.bos_token,
            config.eos_token,
            config.unknown_token,
        )?;

        if score_bits.len() != tokens.len() {
            return Err(TokenizerError::InvalidScoreCount {
                expected: tokens.len(),
                actual: score_bits.len(),
            });
        }
        if token_types.len() != tokens.len() {
            return Err(TokenizerError::InvalidTokenTypeCount {
                expected: tokens.len(),
                actual: token_types.len(),
            });
        }

        let mut token_to_id = BTreeMap::new();
        let mut byte_tokens = [None; 256];
        for (index, token) in tokens.iter().enumerate() {
            let id = u32::try_from(index).map_err(|_| TokenizerError::Overflow)?;
            if !f32::from_bits(score_bits[index]).is_finite() {
                return Err(TokenizerError::NonFiniteScore(id));
            }
            let token_type = token_types[index];
            if !(TOKEN_TYPE_NORMAL..=TOKEN_TYPE_BYTE).contains(&token_type) {
                return Err(TokenizerError::InvalidTokenType {
                    token: id,
                    token_type,
                });
            }
            token_to_id.insert(token.clone(), id);
            if token_type == TOKEN_TYPE_BYTE {
                if let Some(byte) = parse_byte_piece(token) {
                    byte_tokens[usize::from(byte)] = Some(id);
                }
            }
        }

        Ok(Self {
            tokens,
            score_bits,
            token_types,
            bos_token: config.bos_token,
            eos_token: config.eos_token,
            unknown_token: config.unknown_token,
            add_space_prefix: config.add_space_prefix,
            add_bos_token: config.add_bos_token,
            add_eos_token: config.add_eos_token,
            token_to_id,
            byte_tokens,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn bos_token(&self) -> Option<u32> {
        self.bos_token
    }

    pub fn eos_token(&self) -> Option<u32> {
        self.eos_token
    }

    pub fn unknown_token(&self) -> Option<u32> {
        self.unknown_token
    }

    pub fn add_space_prefix(&self) -> bool {
        self.add_space_prefix
    }

    pub fn add_bos_token(&self) -> bool {
        self.add_bos_token
    }

    pub fn add_eos_token(&self) -> bool {
        self.add_eos_token
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, TokenizerError> {
        let normalized = normalize_spm(text, self.add_space_prefix);
        let mut output = Vec::new();

        if add_special_tokens && self.add_bos_token {
            output.extend(self.bos_token);
        }

        if !normalized.is_empty() {
            self.encode_normalized(&normalized, &mut output)?;
        }

        if add_special_tokens && self.add_eos_token {
            output.extend(self.eos_token);
        }

        Ok(output)
    }

    pub fn decode(&self, token_ids: &[u32], skip_special: bool) -> Result<String, TokenizerError> {
        self.decode_after(None, token_ids, skip_special)
    }

    pub fn decode_after(
        &self,
        previous_token: Option<u32>,
        token_ids: &[u32],
        skip_special: bool,
    ) -> Result<String, TokenizerError> {
        let mut bytes = Vec::new();
        let mut strip_dummy_prefix = self.add_space_prefix && previous_token == self.bos_token;

        for id in token_ids {
            let index = usize::try_from(*id).map_err(|_| TokenizerError::InvalidTokenId(*id))?;
            let token = self
                .tokens
                .get(index)
                .ok_or(TokenizerError::InvalidTokenId(*id))?;
            let token_type = self.token_types[index];

            if skip_special
                && (self.bos_token == Some(*id)
                    || self.eos_token == Some(*id)
                    || self.unknown_token == Some(*id)
                    || token_type == TOKEN_TYPE_CONTROL
                    || token_type == TOKEN_TYPE_UNUSED)
            {
                strip_dummy_prefix = self.add_space_prefix && self.bos_token == Some(*id);
                continue;
            }

            let before = bytes.len();
            if token_type == TOKEN_TYPE_BYTE {
                let byte = parse_byte_piece(token).ok_or(TokenizerError::InvalidTokenType {
                    token: *id,
                    token_type,
                })?;
                bytes.push(byte);
            } else {
                append_unescaped_spm(token, &mut bytes);
            }

            if strip_dummy_prefix && bytes.get(before) == Some(&b' ') {
                bytes.remove(before);
            }
            strip_dummy_prefix = false;
        }

        String::from_utf8(bytes).map_err(|_| TokenizerError::InvalidUtf8)
    }

    fn encode_normalized(
        &self,
        normalized: &[u8],
        output: &mut Vec<u32>,
    ) -> Result<(), TokenizerError> {
        let mut symbols = split_utf8_symbols(normalized);
        let mut queue = BinaryHeap::new();
        let mut reverse_merges = BTreeMap::<Vec<u8>, (Vec<u8>, Vec<u8>)>::new();

        for right in 1..symbols.len() {
            self.try_add_bigram(
                normalized,
                &symbols,
                right - 1,
                right,
                &mut queue,
                &mut reverse_merges,
            )?;
        }

        while let Some(bigram) = queue.pop() {
            if symbols[bigram.left].len == 0 || symbols[bigram.right].len == 0 {
                continue;
            }
            if symbols[bigram.left]
                .len
                .checked_add(symbols[bigram.right].len)
                .ok_or(TokenizerError::Overflow)?
                != bigram.size
            {
                continue;
            }
            if symbols[bigram.left].next != Some(bigram.right) {
                continue;
            }

            let right_next = symbols[bigram.right].next;
            symbols[bigram.left].len = bigram.size;
            symbols[bigram.left].next = right_next;
            symbols[bigram.right].len = 0;
            if let Some(next) = right_next {
                symbols[next].prev = Some(bigram.left);
            }

            if let Some(prev) = symbols[bigram.left].prev {
                self.try_add_bigram(
                    normalized,
                    &symbols,
                    prev,
                    bigram.left,
                    &mut queue,
                    &mut reverse_merges,
                )?;
            }
            if let Some(next) = symbols[bigram.left].next {
                self.try_add_bigram(
                    normalized,
                    &symbols,
                    bigram.left,
                    next,
                    &mut queue,
                    &mut reverse_merges,
                )?;
            }
        }

        if symbols.is_empty() {
            return Ok(());
        }

        let mut current = Some(0usize);
        while let Some(index) = current {
            let symbol = &symbols[index];
            if symbol.len > 0 {
                let end = symbol
                    .start
                    .checked_add(symbol.len)
                    .ok_or(TokenizerError::Overflow)?;
                self.emit_piece(&normalized[symbol.start..end], &reverse_merges, output)?;
            }
            current = symbol.next;
        }
        Ok(())
    }

    fn try_add_bigram(
        &self,
        normalized: &[u8],
        symbols: &[SpmSymbol],
        left: usize,
        right: usize,
        queue: &mut BinaryHeap<SpmBigram>,
        reverse_merges: &mut BTreeMap<Vec<u8>, (Vec<u8>, Vec<u8>)>,
    ) -> Result<(), TokenizerError> {
        if symbols[left].len == 0 || symbols[right].len == 0 {
            return Ok(());
        }

        let left_end = symbols[left]
            .start
            .checked_add(symbols[left].len)
            .ok_or(TokenizerError::Overflow)?;
        if left_end != symbols[right].start {
            return Ok(());
        }
        let end = left_end
            .checked_add(symbols[right].len)
            .ok_or(TokenizerError::Overflow)?;
        let text = normalized
            .get(symbols[left].start..end)
            .ok_or(TokenizerError::Overflow)?
            .to_vec();
        let Some(token_id) = self.token_to_id.get(&text).copied() else {
            return Ok(());
        };
        let token_index =
            usize::try_from(token_id).map_err(|_| TokenizerError::InvalidTokenId(token_id))?;
        if !is_merge_token_type(self.token_types[token_index]) {
            return Ok(());
        }

        let score = f32::from_bits(self.score_bits[token_index]);
        queue.push(SpmBigram {
            left,
            right,
            score,
            size: text.len(),
        });

        let left_text = normalized[symbols[left].start..left_end].to_vec();
        let right_text = normalized[left_end..end].to_vec();
        reverse_merges.insert(text, (left_text, right_text));
        Ok(())
    }

    fn emit_piece(
        &self,
        text: &[u8],
        reverse_merges: &BTreeMap<Vec<u8>, (Vec<u8>, Vec<u8>)>,
        output: &mut Vec<u32>,
    ) -> Result<(), TokenizerError> {
        if let Some(token_id) = self.token_to_id.get(text).copied() {
            let index =
                usize::try_from(token_id).map_err(|_| TokenizerError::InvalidTokenId(token_id))?;
            if is_output_token_type(self.token_types[index]) {
                output.push(token_id);
                return Ok(());
            }
        }

        if let Some((left, right)) = reverse_merges.get(text) {
            self.emit_piece(left, reverse_merges, output)?;
            self.emit_piece(right, reverse_merges, output)?;
            return Ok(());
        }

        for byte in text {
            let token = self.byte_tokens[usize::from(*byte)]
                .ok_or(TokenizerError::MissingByteToken(*byte))?;
            output.push(token);
        }
        Ok(())
    }
}

impl TextTokenizer for LlamaSpmTokenizer {
    fn vocab_size(&self) -> usize {
        LlamaSpmTokenizer::vocab_size(self)
    }

    fn eos_token(&self) -> Option<u32> {
        LlamaSpmTokenizer::eos_token(self)
    }

    fn encode_text(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Vec<u32>, TokenizerError> {
        self.encode(text, add_special_tokens)
    }

    fn decode_text(&self, token_ids: &[u32], skip_special: bool) -> Result<String, TokenizerError> {
        self.decode(token_ids, skip_special)
    }
}

#[derive(Debug, Clone)]
struct SpmSymbol {
    start: usize,
    len: usize,
    prev: Option<usize>,
    next: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
struct SpmBigram {
    left: usize,
    right: usize,
    score: f32,
    size: usize,
}

impl PartialEq for SpmBigram {
    fn eq(&self, other: &Self) -> bool {
        self.left == other.left
            && self.right == other.right
            && self.score.to_bits() == other.score.to_bits()
            && self.size == other.size
    }
}

impl Eq for SpmBigram {}

impl PartialOrd for SpmBigram {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SpmBigram {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.left.cmp(&self.left))
            .then_with(|| other.right.cmp(&self.right))
    }
}

pub(crate) fn validate_vocabulary(tokens: &[Vec<u8>]) -> Result<(), TokenizerError> {
    if tokens.is_empty() {
        return Err(TokenizerError::EmptyVocabulary);
    }

    let mut seen = BTreeMap::new();
    for (index, token) in tokens.iter().enumerate() {
        let id = u32::try_from(index).map_err(|_| TokenizerError::Overflow)?;
        if token.is_empty() {
            return Err(TokenizerError::EmptyToken(id));
        }
        if let Some(first) = seen.insert(token.clone(), id) {
            return Err(TokenizerError::DuplicateToken { first, second: id });
        }
    }
    Ok(())
}

pub(crate) fn validate_specials(
    tokens: &[Vec<u8>],
    bos_token: Option<u32>,
    eos_token: Option<u32>,
    unknown_token: Option<u32>,
) -> Result<(), TokenizerError> {
    for id in [bos_token, eos_token, unknown_token].into_iter().flatten() {
        let index = usize::try_from(id).map_err(|_| TokenizerError::InvalidSpecialToken(id))?;
        if index >= tokens.len() {
            return Err(TokenizerError::InvalidSpecialToken(id));
        }
    }
    Ok(())
}

fn normalize_spm(text: &str, add_space_prefix: bool) -> Vec<u8> {
    let mut output = Vec::with_capacity(text.len().saturating_add(3));
    if add_space_prefix && !text.is_empty() {
        output.extend_from_slice(SPM_SPACE);
    }
    for character in text.chars() {
        if character == ' ' {
            output.extend_from_slice(SPM_SPACE);
        } else {
            let mut encoded = [0u8; 4];
            output.extend_from_slice(character.encode_utf8(&mut encoded).as_bytes());
        }
    }
    output
}

fn split_utf8_symbols(text: &[u8]) -> Vec<SpmSymbol> {
    let mut symbols = Vec::new();
    let mut offset = 0usize;

    while offset < text.len() {
        let len = utf8_char_len(text[offset]).min(text.len() - offset);
        let index = symbols.len();
        symbols.push(SpmSymbol {
            start: offset,
            len,
            prev: index.checked_sub(1),
            next: None,
        });
        if index > 0 {
            symbols[index - 1].next = Some(index);
        }
        offset += len;
    }

    symbols
}

fn utf8_char_len(first: u8) -> usize {
    if first & 0b1000_0000 == 0 {
        1
    } else if first & 0b1110_0000 == 0b1100_0000 {
        2
    } else if first & 0b1111_0000 == 0b1110_0000 {
        3
    } else if first & 0b1111_1000 == 0b1111_0000 {
        4
    } else {
        1
    }
}

fn is_merge_token_type(token_type: i32) -> bool {
    matches!(token_type, TOKEN_TYPE_NORMAL | TOKEN_TYPE_USER_DEFINED)
}

fn is_output_token_type(token_type: i32) -> bool {
    matches!(
        token_type,
        TOKEN_TYPE_NORMAL | TOKEN_TYPE_UNKNOWN | TOKEN_TYPE_USER_DEFINED
    )
}

fn parse_byte_piece(token: &[u8]) -> Option<u8> {
    if token.len() == 1 {
        return Some(token[0]);
    }
    if token.len() != 6
        || token[0] != b'<'
        || token[1] != b'0'
        || token[2] != b'x'
        || token[5] != b'>'
    {
        return None;
    }
    let high = hex_nibble(token[3])?;
    let low = hex_nibble(token[4])?;
    Some((high << 4) | low)
}

fn hex_nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

fn append_unescaped_spm(token: &[u8], output: &mut Vec<u8>) {
    let mut offset = 0usize;
    while offset < token.len() {
        if token[offset..].starts_with(SPM_SPACE) {
            output.push(b' ');
            offset += SPM_SPACE.len();
        } else {
            output.push(token[offset]);
            offset += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer() -> VocabularyTokenizer {
        VocabularyTokenizer::new(
            vec![
                b"a".to_vec(),
                b"ab".to_vec(),
                b"b".to_vec(),
                b"<bos>".to_vec(),
                b"<eos>".to_vec(),
                b"<unk>".to_vec(),
            ],
            Some(3),
            Some(4),
            Some(5),
        )
        .expect("tokenizer")
    }

    fn spm_tokenizer() -> LlamaSpmTokenizer {
        let mut tokens = vec![
            b"<unk>".to_vec(),
            b"<s>".to_vec(),
            b"</s>".to_vec(),
            SPM_SPACE.to_vec(),
            b"h".to_vec(),
            b"e".to_vec(),
            b"l".to_vec(),
            b"o".to_vec(),
            b"he".to_vec(),
            b"ll".to_vec(),
            b"lo".to_vec(),
            [&SPM_SPACE[..], b"he"].concat(),
        ];
        let mut scores = vec![
            -1000.0, -1000.0, -1000.0, -10.0, -5.0, -5.0, -5.0, -5.0, 2.0, 1.0, 1.5, 3.0,
        ];
        let mut types = vec![
            TOKEN_TYPE_UNKNOWN,
            TOKEN_TYPE_CONTROL,
            TOKEN_TYPE_CONTROL,
            TOKEN_TYPE_NORMAL,
            TOKEN_TYPE_NORMAL,
            TOKEN_TYPE_NORMAL,
            TOKEN_TYPE_NORMAL,
            TOKEN_TYPE_NORMAL,
            TOKEN_TYPE_NORMAL,
            TOKEN_TYPE_NORMAL,
            TOKEN_TYPE_NORMAL,
            TOKEN_TYPE_NORMAL,
        ];
        for byte in 0u8..=u8::MAX {
            tokens.push(format!("<0x{byte:02X}>").into_bytes());
            scores.push(-1000.0);
            types.push(TOKEN_TYPE_BYTE);
        }

        LlamaSpmTokenizer::new(
            tokens,
            scores.into_iter().map(f32::to_bits).collect(),
            types,
            LlamaSpmConfig {
                bos_token: Some(1),
                eos_token: Some(2),
                unknown_token: Some(0),
                add_space_prefix: true,
                add_bos_token: true,
                add_eos_token: false,
            },
        )
        .expect("spm tokenizer")
    }

    #[test]
    fn longest_match_is_deterministic() {
        assert_eq!(
            tokenizer().encode("aba", false).expect("encode"),
            vec![1, 0]
        );
    }

    #[test]
    fn special_tokens_are_not_matched_from_user_text() {
        let tokenizer = tokenizer();
        let encoded = tokenizer.encode("<bos>", false).expect("encode");
        assert_eq!(encoded, vec![5, 2, 5, 5, 5]);
    }

    #[test]
    fn round_trip_skips_special_tokens() {
        let tokenizer = tokenizer();
        assert_eq!(tokenizer.decode(&[3, 0, 2, 4], true).expect("decode"), "ab");
    }

    #[test]
    fn llama_spm_uses_score_ordered_merges_and_byte_fallback() {
        let tokenizer = spm_tokenizer();
        let tokens = tokenizer.encode("hello!", true).expect("encode");
        let byte_exclamation = 12 + usize::from(b'!');

        assert_eq!(
            tokens,
            vec![
                1,
                11,
                6,
                10,
                u32::try_from(byte_exclamation).expect("byte id")
            ]
        );
        assert_eq!(tokenizer.decode(&tokens, true).expect("decode"), " hello!");
    }

    #[test]
    fn llama_spm_decode_after_bos_strips_source_dummy_prefix() {
        let tokenizer = spm_tokenizer();
        let tokens = tokenizer.encode("hello!", true).expect("encode");
        assert_eq!(
            tokenizer
                .decode_after(Some(1), &tokens[1..], true)
                .expect("decode after bos"),
            "hello!"
        );
    }

    #[test]
    fn llama_spm_honors_source_bos_and_eos_policy() {
        let tokenizer = spm_tokenizer();
        assert_eq!(tokenizer.encode("", true).expect("empty"), vec![1]);
        assert_eq!(
            tokenizer.encode("", false).expect("plain empty"),
            Vec::<u32>::new()
        );
    }

    #[test]
    fn llama_spm_rejects_missing_byte_fallback_instead_of_guessing() {
        let tokenizer = LlamaSpmTokenizer::new(
            vec![b"<unk>".to_vec(), b"a".to_vec()],
            vec![(-1000.0f32).to_bits(), 0.0f32.to_bits()],
            vec![TOKEN_TYPE_UNKNOWN, TOKEN_TYPE_NORMAL],
            LlamaSpmConfig {
                bos_token: None,
                eos_token: None,
                unknown_token: Some(0),
                add_space_prefix: false,
                add_bos_token: false,
                add_eos_token: false,
            },
        )
        .expect("tokenizer");

        assert_eq!(
            tokenizer.encode("!", false),
            Err(TokenizerError::MissingByteToken(b'!'))
        );
    }
}
