#![forbid(unsafe_code)]

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VocabularyTokenizer {
    tokens: Vec<Vec<u8>>,
    bos_token: Option<u32>,
    eos_token: Option<u32>,
    unknown_token: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerError {
    EmptyVocabulary,
    EmptyToken(u32),
    DuplicateToken { first: u32, second: u32 },
    InvalidSpecialToken(u32),
    UnmatchedInput(usize),
    InvalidTokenId(u32),
    InvalidUtf8,
}

impl VocabularyTokenizer {
    pub fn new(
        tokens: Vec<Vec<u8>>,
        bos_token: Option<u32>,
        eos_token: Option<u32>,
        unknown_token: Option<u32>,
    ) -> Result<Self, TokenizerError> {
        if tokens.is_empty() {
            return Err(TokenizerError::EmptyVocabulary);
        }

        for (index, token) in tokens.iter().enumerate() {
            let id = u32::try_from(index).map_err(|_| TokenizerError::InvalidTokenId(u32::MAX))?;
            if token.is_empty() {
                return Err(TokenizerError::EmptyToken(id));
            }

            for (prior_index, prior) in tokens[..index].iter().enumerate() {
                if prior == token {
                    return Err(TokenizerError::DuplicateToken {
                        first: u32::try_from(prior_index)
                            .map_err(|_| TokenizerError::InvalidTokenId(u32::MAX))?,
                        second: id,
                    });
                }
            }
        }

        let tokenizer = Self {
            tokens,
            bos_token,
            eos_token,
            unknown_token,
        };
        tokenizer.validate_specials()?;
        Ok(tokenizer)
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
            if let Some(bos) = self.bos_token {
                output.push(bos);
            }
        }

        let mut offset = 0usize;
        while offset < bytes.len() {
            let mut best: Option<(usize, u32)> = None;

            for (index, token) in self.tokens.iter().enumerate() {
                let id =
                    u32::try_from(index).map_err(|_| TokenizerError::InvalidTokenId(u32::MAX))?;
                if self.is_special(id) {
                    continue;
                }
                if !bytes[offset..].starts_with(token) {
                    continue;
                }

                match best {
                    Some((best_len, best_id))
                        if best_len > token.len()
                            || (best_len == token.len() && best_id < id) => {}
                    _ => best = Some((token.len(), id)),
                }
            }

            if let Some((matched_len, id)) = best {
                output.push(id);
                offset = offset
                    .checked_add(matched_len)
                    .ok_or(TokenizerError::UnmatchedInput(offset))?;
                continue;
            }

            if let Some(unknown) = self.unknown_token {
                output.push(unknown);
                offset = offset
                    .checked_add(1)
                    .ok_or(TokenizerError::UnmatchedInput(offset))?;
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

            let index =
                usize::try_from(*id).map_err(|_| TokenizerError::InvalidTokenId(*id))?;
            let token = self
                .tokens
                .get(index)
                .ok_or(TokenizerError::InvalidTokenId(*id))?;
            bytes.extend_from_slice(token);
        }

        String::from_utf8(bytes).map_err(|_| TokenizerError::InvalidUtf8)
    }

    fn validate_specials(&self) -> Result<(), TokenizerError> {
        for id in [self.bos_token, self.eos_token, self.unknown_token]
            .into_iter()
            .flatten()
        {
            let index = usize::try_from(id).map_err(|_| TokenizerError::InvalidSpecialToken(id))?;
            if index >= self.tokens.len() {
                return Err(TokenizerError::InvalidSpecialToken(id));
            }
        }
        Ok(())
    }

    fn is_special(&self, id: u32) -> bool {
        self.bos_token == Some(id)
            || self.eos_token == Some(id)
            || self.unknown_token == Some(id)
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

    #[test]
    fn longest_match_is_deterministic() {
        assert_eq!(tokenizer().encode("aba", false).expect("encode"), vec![1, 0]);
    }

    #[test]
    fn special_tokens_are_not_matched_from_user_text() {
        let tokenizer = tokenizer();
        let encoded = tokenizer.encode("<bos>", false).expect("encode");
        assert_eq!(encoded, vec![5, 5, 2, 5, 5]);
    }

    #[test]
    fn round_trip_skips_special_tokens() {
        let tokenizer = tokenizer();
        assert_eq!(
            tokenizer
                .decode(&[3, 0, 2, 4], true)
                .expect("decode"),
            "ab"
        );
    }
}
