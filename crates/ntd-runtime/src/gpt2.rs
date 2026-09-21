#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use unicode_general_category::{get_general_category, GeneralCategory};

use crate::tokenizer::{validate_specials, validate_vocabulary, TextTokenizer, TokenizerError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gpt2BpeConfig {
    pub bos_token: Option<u32>,
    pub eos_token: Option<u32>,
    pub unknown_token: Option<u32>,
    pub add_bos_token: bool,
    pub add_eos_token: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gpt2BpeTokenizer {
    tokens: Vec<Vec<u8>>,
    bos_token: Option<u32>,
    eos_token: Option<u32>,
    unknown_token: Option<u32>,
    add_bos_token: bool,
    add_eos_token: bool,
    token_to_id: BTreeMap<Vec<u8>, u32>,
    merge_ranks: BTreeMap<(Vec<u8>, Vec<u8>), usize>,
    byte_encoder: [char; 256],
    byte_decoder: BTreeMap<char, u8>,
}

impl Gpt2BpeTokenizer {
    pub fn new(
        tokens: Vec<Vec<u8>>,
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

        let token_to_id = tokens
            .iter()
            .enumerate()
            .map(|(index, token)| {
                u32::try_from(index)
                    .map(|id| (token.clone(), id))
                    .map_err(|_| TokenizerError::Overflow)
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;

        let mut merge_ranks = BTreeMap::new();
        for (rank, merge) in merges.into_iter().enumerate() {
            let Some((left, right)) = parse_merge(&merge) else {
                return Err(TokenizerError::InvalidBpeMerge(merge));
            };
            let key = (left.as_bytes().to_vec(), right.as_bytes().to_vec());
            if merge_ranks.insert(key.clone(), rank).is_some() {
                return Err(TokenizerError::DuplicateBpeMerge(merge));
            }

            let mut merged = key.0.clone();
            merged.extend_from_slice(&key.1);
            if !token_to_id.contains_key(&merged) {
                return Err(TokenizerError::InvalidBpeMerge(merge));
            }
        }

        let byte_encoder = gpt2_byte_encoder();
        let byte_decoder = byte_encoder
            .iter()
            .enumerate()
            .map(|(byte, character)| {
                u8::try_from(byte)
                    .map(|byte| (*character, byte))
                    .map_err(|_| TokenizerError::Overflow)
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;

        for character in byte_encoder {
            let encoded = character.to_string().into_bytes();
            if !token_to_id.contains_key(&encoded) {
                return Err(TokenizerError::MissingTokenPiece(encoded));
            }
        }

        Ok(Self {
            tokens,
            bos_token: config.bos_token,
            eos_token: config.eos_token,
            unknown_token: config.unknown_token,
            add_bos_token: config.add_bos_token,
            add_eos_token: config.add_eos_token,
            token_to_id,
            merge_ranks,
            byte_encoder,
            byte_decoder,
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

    pub fn add_bos_token(&self) -> bool {
        self.add_bos_token
    }

    pub fn add_eos_token(&self) -> bool {
        self.add_eos_token
    }

    pub fn encode(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Vec<u32>, TokenizerError> {
        let mut output = Vec::new();
        if add_special_tokens && self.add_bos_token {
            output.extend(self.bos_token);
        }

        for word in gpt2_pretokenize(text) {
            let encoded = self.byte_encode(&word);
            self.encode_word(&encoded, &mut output)?;
        }

        if add_special_tokens && self.add_eos_token {
            output.extend(self.eos_token);
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
            let token_text = std::str::from_utf8(token).map_err(|_| TokenizerError::InvalidUtf8)?;

            for character in token_text.chars() {
                let byte = self
                    .byte_decoder
                    .get(&character)
                    .copied()
                    .ok_or(TokenizerError::InvalidGpt2ByteChar(character))?;
                bytes.push(byte);
            }
        }

        String::from_utf8(bytes).map_err(|_| TokenizerError::InvalidUtf8)
    }

    fn is_special(&self, id: u32) -> bool {
        self.bos_token == Some(id) || self.eos_token == Some(id) || self.unknown_token == Some(id)
    }

    fn byte_encode(&self, text: &str) -> Vec<u8> {
        let mut encoded = String::new();
        for byte in text.as_bytes() {
            encoded.push(self.byte_encoder[usize::from(*byte)]);
        }
        encoded.into_bytes()
    }

    fn encode_word(&self, encoded: &[u8], output: &mut Vec<u32>) -> Result<(), TokenizerError> {
        let encoded_text = std::str::from_utf8(encoded).map_err(|_| TokenizerError::InvalidUtf8)?;
        let mut symbols = encoded_text
            .chars()
            .map(|character| character.to_string().into_bytes())
            .collect::<Vec<_>>();

        loop {
            let mut best: Option<(usize, usize)> = None;
            for index in 0..symbols.len().saturating_sub(1) {
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
            let token = self
                .token_to_id
                .get(&symbol)
                .copied()
                .ok_or_else(|| TokenizerError::MissingTokenPiece(symbol.clone()))?;
            output.push(token);
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

fn parse_merge(merge: &str) -> Option<(&str, &str)> {
    let (left, right) = merge.split_once(' ')?;
    if left.is_empty() || right.is_empty() || right.contains(' ') {
        return None;
    }
    Some((left, right))
}

fn gpt2_pretokenize(text: &str) -> Vec<String> {
    let characters = text.chars().collect::<Vec<_>>();
    let mut output = Vec::new();
    let mut position = 0usize;

    while position < characters.len() {
        let start = position;
        let current = characters[position];

        if current == '\'' && position + 1 < characters.len() {
            let next = characters[position + 1];
            if matches!(next, 's' | 't' | 'm' | 'd') {
                position += 2;
                output.push(characters[start..position].iter().collect());
                continue;
            }
            if position + 2 < characters.len() {
                let third = characters[position + 2];
                if matches!((next, third), ('r', 'e') | ('v', 'e') | ('l', 'l')) {
                    position += 3;
                    output.push(characters[start..position].iter().collect());
                    continue;
                }
            }
        }

        let classified = if current == ' ' {
            position.checked_add(1)
        } else {
            Some(position)
        };

        if classified
            .and_then(|index| characters.get(index))
            .is_some_and(|character| is_letter(*character))
        {
            if current == ' ' {
                position += 1;
            }
            while position < characters.len() && is_letter(characters[position]) {
                position += 1;
            }
            output.push(characters[start..position].iter().collect());
            continue;
        }

        if classified
            .and_then(|index| characters.get(index))
            .is_some_and(|character| is_number(*character))
        {
            if current == ' ' {
                position += 1;
            }
            while position < characters.len() && is_number(characters[position]) {
                position += 1;
            }
            output.push(characters[start..position].iter().collect());
            continue;
        }

        if classified
            .and_then(|index| characters.get(index))
            .is_some_and(|character| is_other(*character))
        {
            if current == ' ' {
                position += 1;
            }
            while position < characters.len() && is_other(characters[position]) {
                position += 1;
            }
            output.push(characters[start..position].iter().collect());
            continue;
        }

        let mut whitespace_count = 0usize;
        while position + whitespace_count < characters.len()
            && characters[position + whitespace_count].is_whitespace()
        {
            whitespace_count += 1;
        }

        if whitespace_count > 1 && position + whitespace_count < characters.len() {
            position += whitespace_count - 1;
            output.push(characters[start..position].iter().collect());
            continue;
        }

        if whitespace_count > 0 {
            position += whitespace_count;
            output.push(characters[start..position].iter().collect());
            continue;
        }

        position += 1;
        output.push(characters[start..position].iter().collect());
    }

    output
}

fn is_letter(character: char) -> bool {
    matches!(
        get_general_category(character),
        GeneralCategory::UppercaseLetter
            | GeneralCategory::LowercaseLetter
            | GeneralCategory::TitlecaseLetter
            | GeneralCategory::ModifierLetter
            | GeneralCategory::OtherLetter
    )
}

fn is_number(character: char) -> bool {
    matches!(
        get_general_category(character),
        GeneralCategory::DecimalNumber
            | GeneralCategory::LetterNumber
            | GeneralCategory::OtherNumber
    )
}

fn is_other(character: char) -> bool {
    !character.is_whitespace() && !is_letter(character) && !is_number(character)
}

fn gpt2_byte_encoder() -> [char; 256] {
    let mut output = ['\0'; 256];
    let mut assigned = [false; 256];

    for byte in 0x21u16..=0x7Eu16 {
        let index = usize::from(byte);
        output[index] = char::from_u32(u32::from(byte)).expect("ASCII byte");
        assigned[index] = true;
    }
    for byte in 0xA1u16..=0xACu16 {
        let index = usize::from(byte);
        output[index] = char::from_u32(u32::from(byte)).expect("Latin-1 byte");
        assigned[index] = true;
    }
    for byte in 0xAEu16..=0xFFu16 {
        let index = usize::from(byte);
        output[index] = char::from_u32(u32::from(byte)).expect("Latin-1 byte");
        assigned[index] = true;
    }

    let mut fallback = 0u32;
    for byte in 0u16..=0xFFu16 {
        let index = usize::from(byte);
        if assigned[index] {
            continue;
        }
        output[index] = char::from_u32(256 + fallback).expect("GPT-2 byte codepoint");
        fallback += 1;
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_tokenizer() -> Gpt2BpeTokenizer {
        let encoder = gpt2_byte_encoder();
        let mut tokens = Vec::new();
        for character in encoder {
            tokens.push(character.to_string().into_bytes());
        }

        let byte = |value: u8| encoder[usize::from(value)].to_string();
        let space = byte(b' ');
        let h = byte(b'h');
        let e = byte(b'e');
        let l = byte(b'l');
        let o = byte(b'o');

        let he = format!("{h}{e}");
        let hel = format!("{he}{l}");
        let hell = format!("{hel}{l}");
        let hello = format!("{hell}{o}");
        let spaced_hello = format!("{space}{hello}");

        tokens.extend([
            he.as_bytes().to_vec(),
            hel.as_bytes().to_vec(),
            hell.as_bytes().to_vec(),
            hello.as_bytes().to_vec(),
            spaced_hello.as_bytes().to_vec(),
            b"<|endoftext|>".to_vec(),
        ]);

        let merges = vec![
            format!("{h} {e}"),
            format!("{he} {l}"),
            format!("{hel} {l}"),
            format!("{hell} {o}"),
            format!("{space} {hello}"),
        ];

        let special = u32::try_from(tokens.len() - 1).expect("special id");
        Gpt2BpeTokenizer::new(
            tokens,
            merges,
            Gpt2BpeConfig {
                bos_token: Some(special),
                eos_token: Some(special),
                unknown_token: None,
                add_bos_token: false,
                add_eos_token: false,
            },
        )
        .expect("gpt2 tokenizer")
    }

    #[test]
    fn canonical_gpt2_regex_splits_contractions_categories_and_whitespace() {
        assert_eq!(
            gpt2_pretokenize(" hello! it's 42\n\nnext"),
            vec![" hello", "!", " it", "'s", " 42", "\n", "\n", "next"]
        );
        assert_eq!(
            gpt2_pretokenize(" αβ１２"),
            vec![" αβ", "１２"]
        );
    }

    #[test]
    fn byte_encoder_matches_known_gpt2_space_mapping() {
        let encoder = gpt2_byte_encoder();
        assert_eq!(encoder[usize::from(b' ')], 'Ġ');
        assert_eq!(encoder[usize::from(b'!')], '!');
        assert_eq!(encoder[0], 'Ā');
    }

    #[test]
    fn gpt2_bpe_uses_merge_ranks_and_round_trips_utf8() {
        let tokenizer = toy_tokenizer();
        let encoded = tokenizer.encode(" hello!", false).expect("encode");
        assert_eq!(encoded.len(), 2);
        assert_eq!(tokenizer.decode(&encoded, false).expect("decode"), " hello!");

        let unicode = " hé🙂";
        let encoded = tokenizer.encode(unicode, false).expect("unicode encode");
        assert_eq!(tokenizer.decode(&encoded, false).expect("unicode decode"), unicode);
    }

    #[test]
    fn gpt2_bpe_honors_source_special_token_policy() {
        let mut tokenizer = toy_tokenizer();
        tokenizer.add_bos_token = true;
        tokenizer.add_eos_token = true;
        let encoded = tokenizer.encode("a", true).expect("encode");
        assert_eq!(encoded.first().copied(), tokenizer.bos_token());
        assert_eq!(encoded.last().copied(), tokenizer.eos_token());
    }

    #[test]
    fn malformed_or_duplicate_merges_fail_closed() {
        let encoder = gpt2_byte_encoder();
        let tokens = encoder
            .iter()
            .map(|character| character.to_string().into_bytes())
            .collect::<Vec<_>>();
        let config = Gpt2BpeConfig {
            bos_token: None,
            eos_token: None,
            unknown_token: None,
            add_bos_token: false,
            add_eos_token: false,
        };

        assert!(matches!(
            Gpt2BpeTokenizer::new(tokens.clone(), vec!["bad".into()], config),
            Err(TokenizerError::InvalidBpeMerge(_))
        ));
        assert!(matches!(
            Gpt2BpeTokenizer::new(tokens, vec!["a b".into(), "a b".into()], config),
            Err(TokenizerError::InvalidBpeMerge(_)) | Err(TokenizerError::DuplicateBpeMerge(_))
        ));
    }
}
