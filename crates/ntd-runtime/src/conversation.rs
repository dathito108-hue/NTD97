#![forbid(unsafe_code)]

use crate::{TextTokenizer, TokenizerError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversationRole {
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversationTurn {
    pub role: ConversationRole,
    pub content: String,
}

impl ConversationTurn {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ConversationRole::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: ConversationRole::Assistant,
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledChatPrompt {
    pub text: String,
    pub token_ids: Vec<u32>,
    pub retained_history_turns: usize,
    pub dropped_history_turns: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatPromptError {
    EmptyUserMessage,
    EmptyTurn(usize),
    InvalidTokenBudget,
    ContextOverflow { required: usize, limit: usize },
    Tokenizer(TokenizerError),
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct NativeChatPromptCompiler;

impl NativeChatPromptCompiler {
    pub fn compile<T: TextTokenizer>(
        &self,
        history: &[ConversationTurn],
        user_message: &str,
        tokenizer: &T,
        max_prompt_tokens: usize,
    ) -> Result<CompiledChatPrompt, ChatPromptError> {
        if user_message.trim().is_empty() {
            return Err(ChatPromptError::EmptyUserMessage);
        }
        if max_prompt_tokens == 0 {
            return Err(ChatPromptError::InvalidTokenBudget);
        }
        for (index, turn) in history.iter().enumerate() {
            if turn.content.trim().is_empty() {
                return Err(ChatPromptError::EmptyTurn(index));
            }
        }

        for dropped in 0..=history.len() {
            let retained = &history[dropped..];
            let text = render_prompt(retained, user_message);
            let token_ids = tokenizer
                .encode_text(&text, true)
                .map_err(ChatPromptError::Tokenizer)?;
            if token_ids.len() <= max_prompt_tokens {
                return Ok(CompiledChatPrompt {
                    text,
                    token_ids,
                    retained_history_turns: retained.len(),
                    dropped_history_turns: dropped,
                });
            }
        }

        let text = render_prompt(&[], user_message);
        let required = tokenizer
            .encode_text(&text, true)
            .map_err(ChatPromptError::Tokenizer)?
            .len();
        Err(ChatPromptError::ContextOverflow {
            required,
            limit: max_prompt_tokens,
        })
    }
}

fn render_prompt(history: &[ConversationTurn], user_message: &str) -> String {
    let mut text = String::new();
    for turn in history {
        match turn.role {
            ConversationRole::User => text.push_str("User: "),
            ConversationRole::Assistant => text.push_str("Assistant: "),
        }
        text.push_str(&turn.content);
        text.push('\n');
    }
    text.push_str("User: ");
    text.push_str(user_message);
    text.push_str("\nAssistant:");
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VocabularyTokenizer;

    fn byte_tokenizer() -> VocabularyTokenizer {
        VocabularyTokenizer::new(
            (0u8..=u8::MAX).map(|byte| vec![byte]).collect(),
            None,
            None,
            None,
        )
        .expect("byte tokenizer")
    }

    #[test]
    fn compiles_deterministic_chat_template() {
        let compiler = NativeChatPromptCompiler;
        let history = vec![
            ConversationTurn::user("hello"),
            ConversationTurn::assistant("hi"),
        ];

        let compiled = compiler
            .compile(&history, "status?", &byte_tokenizer(), 128)
            .expect("compile");

        assert_eq!(
            compiled.text,
            "User: hello\nAssistant: hi\nUser: status?\nAssistant:"
        );
        assert_eq!(compiled.retained_history_turns, 2);
        assert_eq!(compiled.dropped_history_turns, 0);
        assert_eq!(compiled.token_ids.len(), compiled.text.len());
    }

    #[test]
    fn drops_oldest_complete_turns_to_fit_token_budget() {
        let compiler = NativeChatPromptCompiler;
        let history = vec![
            ConversationTurn::user("first message"),
            ConversationTurn::assistant("first answer"),
            ConversationTurn::user("recent"),
        ];
        let tokenizer = byte_tokenizer();
        let no_history_len = tokenizer
            .encode_text("User: now\nAssistant:", true)
            .expect("encode")
            .len();
        let recent_len = tokenizer
            .encode_text("User: recent\nUser: now\nAssistant:", true)
            .expect("encode")
            .len();

        let compiled = compiler
            .compile(&history, "now", &tokenizer, recent_len)
            .expect("compile");

        assert_eq!(compiled.retained_history_turns, 1);
        assert_eq!(compiled.dropped_history_turns, 2);
        assert_eq!(compiled.text, "User: recent\nUser: now\nAssistant:");
        assert!(compiled.token_ids.len() > no_history_len);
    }

    #[test]
    fn rejects_user_message_that_cannot_fit_without_history() {
        let compiler = NativeChatPromptCompiler;
        let error = compiler
            .compile(&[], "this message is too large", &byte_tokenizer(), 4)
            .expect_err("overflow");

        assert!(matches!(
            error,
            ChatPromptError::ContextOverflow {
                required,
                limit: 4
            } if required > 4
        ));
    }
}
