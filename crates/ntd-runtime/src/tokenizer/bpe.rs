#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use super::{
    validate_specials, validate_vocabulary, TextTokenizer, TokenizerError, TOKEN_TYPE_CONTROL,
    TOKEN_TYPE_UNUSED,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gpt2PreTokenizer {
    Gpt2,
    Llama3,
}

impl Gpt2PreTokenizer {
    pub fn source_name(self) -> &'static str {
        match self {
            Self::Gpt2 => "gpt-2",
            Self::Llama3 => "llama-bpe",
        }
    }

    pub fn from_source_name(value: &str) -> Option<Self> {
        match value {
            "gpt-2" | "gpt2" => Some(Self::Gpt2),
            "llama3" | "llama-v3" | "llama-bpe" | "falcon3" => Some(Self::Llama3),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gpt2BpeConfig {
    pub pre_tokenizer: Gpt2PreTokenizer,
    pub bos_token: Option<u32>,
    pub eos_token: Option<u32>,
    pub unknown_token: Option<u32>,
    pub add_bos_token: bool,
    pub add_eos_token: bool,
    pub ignore_merges: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gpt2BpeTokenizer {
    tokens: Vec<Vec<u8>>,
    token_types: Vec<i32>,
    merge_ranks: BTreeMap<(Vec<u8>, Vec<u8>), usize>,
    token_to_id: BTreeMap<Vec<u8>, u32>,
    config: Gpt2BpeConfig,
    byte_decoder: BTreeMap<char, u8>,
}

impl Gpt2BpeTokenizer {
    pub fn new(
        tokens: Vec<Vec<u8>>,
        token_types: Vec<i32>,
        merges: Vec<String>,
        config: Gpt2BpeConfig,
    ) -> Result<Self, TokenizerError> {
        validate_vocabulary(&tokens)?;
        validate_specials(
            &tokens,
            config.bos_token,
            config.eos_token,
            config.unknown_token,
        )?;
        if token_types.len() != tokens.len() {
            return Err(TokenizerError::InvalidTokenTypeCount {
                expected: tokens.len(),
                actual: token_types.len(),
            });
        }

        let mut token_to_id = BTreeMap::new();
        for (index, token) in tokens.iter().enumerate() {
            let id = u32::try_from(index).map_err(|_| TokenizerError::Overflow)?;
            token_to_id.insert(token.clone(), id);
        }

        let mut merge_ranks = BTreeMap::new();
        for (rank, merge) in merges.iter().enumerate() {
            let (left, right) = split_merge(merge).ok_or(TokenizerError::InvalidMerge(rank))?;
            let key = (left.as_bytes().to_vec(), right.as_bytes().to_vec());
            if let Some(first) = merge_ranks.insert(key, rank) {
                return Err(TokenizerError::DuplicateMerge {
                    first,
                    second: rank,
                });
            }
        }

        let mut byte_decoder = BTreeMap::new();
        for byte in 0u8..=u8::MAX {
            byte_decoder.insert(gpt2_byte_char(byte), byte);
        }

        Ok(Self {
            tokens,
            token_types,
            merge_ranks,
            token_to_id,
            config,
            byte_decoder,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn eos_token(&self) -> Option<u32> {
        self.config.eos_token
    }

    pub fn pre_tokenizer(&self) -> Gpt2PreTokenizer {
        self.config.pre_tokenizer
    }

    pub fn encode(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Vec<u32>, TokenizerError> {
        let mut output = Vec::new();
        if add_special_tokens && self.config.add_bos_token {
            output.extend(self.config.bos_token);
        }

        let spans = match self.config.pre_tokenizer {
            Gpt2PreTokenizer::Gpt2 => split_gpt2(text),
            Gpt2PreTokenizer::Llama3 => split_llama3(text),
        };
        for span in spans {
            self.encode_span(span.as_bytes(), &mut output)?;
        }

        if add_special_tokens && self.config.add_eos_token {
            output.extend(self.config.eos_token);
        }
        Ok(output)
    }

    pub fn decode(&self, token_ids: &[u32], skip_special: bool) -> Result<String, TokenizerError> {
        let mut bytes = Vec::new();
        for id in token_ids {
            let index = usize::try_from(*id).map_err(|_| TokenizerError::InvalidTokenId(*id))?;
            let token = self
                .tokens
                .get(index)
                .ok_or(TokenizerError::InvalidTokenId(*id))?;
            let token_type = self.token_types[index];
            let is_special = self.config.bos_token == Some(*id)
                || self.config.eos_token == Some(*id)
                || self.config.unknown_token == Some(*id)
                || token_type == TOKEN_TYPE_CONTROL
                || token_type == TOKEN_TYPE_UNUSED;
            if skip_special && is_special {
                continue;
            }
            if is_special {
                bytes.extend_from_slice(token);
                continue;
            }

            let piece = std::str::from_utf8(token).map_err(|_| TokenizerError::InvalidUtf8)?;
            for character in piece.chars() {
                let byte = self
                    .byte_decoder
                    .get(&character)
                    .copied()
                    .ok_or(TokenizerError::InvalidByteEncoding(character as u32))?;
                bytes.push(byte);
            }
        }
        String::from_utf8(bytes).map_err(|_| TokenizerError::InvalidUtf8)
    }

    fn encode_span(&self, bytes: &[u8], output: &mut Vec<u32>) -> Result<(), TokenizerError> {
        if bytes.is_empty() {
            return Ok(());
        }

        let encoded = gpt2_encode_bytes(bytes);
        if self.config.ignore_merges {
            if let Some(id) = self.token_to_id.get(&encoded).copied() {
                output.push(id);
                return Ok(());
            }
        }

        let encoded_text =
            std::str::from_utf8(&encoded).map_err(|_| TokenizerError::InvalidUtf8)?;
        let mut symbols = encoded_text
            .chars()
            .map(|character| character.to_string().into_bytes())
            .collect::<Vec<_>>();

        while symbols.len() > 1 {
            let mut best: Option<(usize, usize)> = None;
            for index in 0..symbols.len() - 1 {
                let key = (symbols[index].clone(), symbols[index + 1].clone());
                let Some(rank) = self.merge_ranks.get(&key).copied() else {
                    continue;
                };
                match best {
                    Some((best_rank, best_index))
                        if best_rank < rank || (best_rank == rank && best_index < index) => {}
                    _ => best = Some((rank, index)),
                }
            }

            let Some((_, index)) = best else {
                break;
            };
            let right = symbols.remove(index + 1);
            symbols[index].extend_from_slice(&right);
        }

        for symbol in symbols {
            let Some(id) = self.token_to_id.get(&symbol).copied() else {
                return Err(TokenizerError::MissingBpeToken(symbol));
            };
            output.push(id);
        }
        Ok(())
    }
}

impl TextTokenizer for Gpt2BpeTokenizer {
    fn vocab_size(&self) -> usize {
        Gpt2BpeTokenizer::vocab_size(self)
    }

    fn eos_token(&self) -> Option<u32> {
        Gpt2BpeTokenizer::eos_token(self)
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

fn split_merge(merge: &str) -> Option<(&str, &str)> {
    let mut parts = merge.split(' ');
    let left = parts.next()?;
    let right = parts.next()?;
    if left.is_empty() || right.is_empty() || parts.next().is_some() {
        return None;
    }
    Some((left, right))
}

fn gpt2_encode_bytes(bytes: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        let mut encoded = [0u8; 4];
        output.extend_from_slice(gpt2_byte_char(*byte).encode_utf8(&mut encoded).as_bytes());
    }
    output
}

fn gpt2_byte_char(byte: u8) -> char {
    if (b'!'..=b'~').contains(&byte)
        || (0xA1..=0xAC).contains(&byte)
        || (0xAE..=0xFF).contains(&byte)
    {
        char::from_u32(u32::from(byte)).expect("byte codepoint")
    } else {
        let mut extra = 0u32;
        for candidate in 0u16..=u16::from(byte) {
            let candidate = candidate as u8;
            let visible = (b'!'..=b'~').contains(&candidate)
                || (0xA1..=0xAC).contains(&candidate)
                || (0xAE..=0xFF).contains(&candidate);
            if !visible {
                if candidate == byte {
                    return char::from_u32(256 + extra).expect("GPT-2 byte codepoint");
                }
                extra += 1;
            }
        }
        unreachable!("target byte must be visited")
    }
}

fn split_gpt2(text: &str) -> Vec<&str> {
    split_profile(text, false)
}

fn split_llama3(text: &str) -> Vec<&str> {
    split_profile(text, true)
}

fn split_profile(text: &str, llama3: bool) -> Vec<&str> {
    let chars = text.char_indices().collect::<Vec<_>>();
    let mut spans = Vec::new();
    let mut cursor = 0usize;

    while cursor < chars.len() {
        let start = chars[cursor].0;
        if let Some(end) = contraction_end(&chars, cursor, llama3) {
            spans.push(&text[start..byte_index(text, &chars, end)]);
            cursor = end;
            continue;
        }

        let current = chars[cursor].1;
        if current == ' ' && cursor + 1 < chars.len() {
            let next = chars[cursor + 1].1;
            if next.is_alphabetic() {
                let end = consume_while(&chars, cursor + 1, |c| c.is_alphabetic());
                spans.push(&text[start..byte_index(text, &chars, end)]);
                cursor = end;
                continue;
            }
            if !llama3 && next.is_numeric() {
                let end = consume_while(&chars, cursor + 1, |c| c.is_numeric());
                spans.push(&text[start..byte_index(text, &chars, end)]);
                cursor = end;
                continue;
            }
            if !next.is_whitespace() && !next.is_alphanumeric() {
                let mut end =
                    consume_while(&chars, cursor + 1, |c| !c.is_whitespace() && !c.is_alphanumeric());
                if llama3 {
                    while end < chars.len() && matches!(chars[end].1, '\r' | '\n') {
                        end += 1;
                    }
                }
                spans.push(&text[start..byte_index(text, &chars, end)]);
                cursor = end;
                continue;
            }
        }

        if llama3
            && current != '\r'
            && current != '\n'
            && !current.is_alphanumeric()
            && cursor + 1 < chars.len()
            && chars[cursor + 1].1.is_alphabetic()
        {
            let end = consume_while(&chars, cursor + 1, |c| c.is_alphabetic());
            spans.push(&text[start..byte_index(text, &chars, end)]);
            cursor = end;
            continue;
        }

        if current.is_alphabetic() {
            let end = consume_while(&chars, cursor, |c| c.is_alphabetic());
            spans.push(&text[start..byte_index(text, &chars, end)]);
            cursor = end;
            continue;
        }

        if current.is_numeric() {
            let mut end = cursor;
            let limit = if llama3 { 3 } else { usize::MAX };
            let mut count = 0usize;
            while end < chars.len() && chars[end].1.is_numeric() && count < limit {
                end += 1;
                count += 1;
            }
            spans.push(&text[start..byte_index(text, &chars, end)]);
            cursor = end;
            continue;
        }

        if current.is_whitespace() {
            let end = consume_while(&chars, cursor, |c| c.is_whitespace());
            spans.push(&text[start..byte_index(text, &chars, end)]);
            cursor = end;
            continue;
        }

        let mut end =
            consume_while(&chars, cursor, |c| !c.is_whitespace() && !c.is_alphanumeric());
        if llama3 {
            while end < chars.len() && matches!(chars[end].1, '\r' | '\n') {
                end += 1;
            }
        }
        spans.push(&text[start..byte_index(text, &chars, end)]);
        cursor = end;
    }

    spans
}

fn contraction_end(chars: &[(usize, char)], cursor: usize, case_insensitive: bool) -> Option<usize> {
    if chars[cursor].1 != '\'' {
        return None;
    }
    const SUFFIXES: [&str; 7] = ["s", "t", "re", "ve", "m", "ll", "d"];
    for suffix in SUFFIXES {
        let suffix_chars = suffix.chars().collect::<Vec<_>>();
        if cursor + 1 + suffix_chars.len() > chars.len() {
            continue;
        }
        let matches = suffix_chars.iter().enumerate().all(|(offset, expected)| {
            let actual = chars[cursor + 1 + offset].1;
            if case_insensitive {
                actual.eq_ignore_ascii_case(expected)
            } else {
                actual == *expected
            }
        });
        if matches {
            return Some(cursor + 1 + suffix_chars.len());
        }
    }
    None
}

fn consume_while<F>(chars: &[(usize, char)], mut cursor: usize, predicate: F) -> usize
where
    F: Fn(char) -> bool,
{
    while cursor < chars.len() && predicate(chars[cursor].1) {
        cursor += 1;
    }
    cursor
}

fn byte_index(text: &str, chars: &[(usize, char)], char_index: usize) -> usize {
    chars.get(char_index).map(|entry| entry.0).unwrap_or(text.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TOKEN_TYPE_CONTROL, TOKEN_TYPE_NORMAL};

    fn encode_piece(value: &str) -> Vec<u8> {
        gpt2_encode_bytes(value.as_bytes())
    }

    fn tokenizer(profile: Gpt2PreTokenizer) -> Gpt2BpeTokenizer {
        let raw_tokens = ["h", "e", "l", "o", "he", "ll", "lo", " hello", "!", "<bos>", "<eos>"];
        let mut tokens = raw_tokens
            .iter()
            .take(9)
            .map(|value| encode_piece(value))
            .collect::<Vec<_>>();
        tokens.push(b"<bos>".to_vec());
        tokens.push(b"<eos>".to_vec());
        let token_types = vec![
            TOKEN_TYPE_NORMAL,
            TOKEN_TYPE_NORMAL,
            TOKEN_TYPE_NORMAL,
            TOKEN_TYPE_NORMAL,
            TOKEN_TYPE_NORMAL,
            TOKEN_TYPE_NORMAL,
            TOKEN_TYPE_NORMAL,
            TOKEN_TYPE_NORMAL,
            TOKEN_TYPE_NORMAL,
            TOKEN_TYPE_CONTROL,
            TOKEN_TYPE_CONTROL,
        ];
        let merges = vec![
            "h e".to_owned(),
            "l l".to_owned(),
            "l o".to_owned(),
            format!("{} {}", gpt2_byte_char(b' '), "he"),
            format!("{} {}", [gpt2_byte_char(b' '), 'h'].iter().collect::<String>(), "e"),
        ];

        Gpt2BpeTokenizer::new(
            tokens,
            token_types,
            merges,
            Gpt2BpeConfig {
                pre_tokenizer: profile,
                bos_token: Some(9),
                eos_token: Some(10),
                unknown_token: None,
                add_bos_token: true,
                add_eos_token: false,
                ignore_merges: profile == Gpt2PreTokenizer::Llama3,
            },
        )
        .expect("tokenizer")
    }

    #[test]
    fn gpt2_byte_mapping_round_trips_non_ascii_bytes() {
        for byte in 0u8..=u8::MAX {
            let character = gpt2_byte_char(byte);
            let tokenizer = tokenizer(Gpt2PreTokenizer::Gpt2);
            assert_eq!(tokenizer.byte_decoder.get(&character), Some(&byte));
        }
    }

    #[test]
    fn classic_gpt2_profile_merges_and_decodes() {
        let tokenizer = tokenizer(Gpt2PreTokenizer::Gpt2);
        let ids = tokenizer.encode("hello!", true).expect("encode");
        assert_eq!(ids.first(), Some(&9));
        assert_eq!(tokenizer.decode(&ids, true).expect("decode"), "hello!");
    }

    #[test]
    fn llama3_profile_uses_direct_vocab_piece_when_merges_are_ignored() {
        let tokenizer = tokenizer(Gpt2PreTokenizer::Llama3);
        let ids = tokenizer.encode(" hello!", false).expect("encode");
        assert_eq!(ids, vec![7, 8]);
        assert_eq!(tokenizer.decode(&ids, true).expect("decode"), " hello!");
    }

    #[test]
    fn llama3_profile_splits_long_digit_runs_in_groups_of_three() {
        assert_eq!(split_llama3("1234567"), vec!["123", "456", "7"]);
    }

    #[test]
    fn gpt2_profile_keeps_digit_run_together() {
        assert_eq!(split_gpt2("1234567"), vec!["1234567"]);
    }
}
