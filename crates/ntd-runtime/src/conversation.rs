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
    pub retained_memory_items: usize,
    pub dropped_memory_items: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatPromptError {
    EmptyUserMessage,
    EmptyTurn(usize),
    InvalidHistoryOrder(usize),
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
        self.compile_with_memory(history, &[], user_message, tokenizer, max_prompt_tokens)
    }

    pub fn compile_with_memory<T: TextTokenizer>(
        &self,
        history: &[ConversationTurn],
        memory: &[String],
        user_message: &str,
        tokenizer: &T,
        max_prompt_tokens: usize,
    ) -> Result<CompiledChatPrompt, ChatPromptError> {
        validate_inputs(history, memory, user_message, max_prompt_tokens)?;

        for dropped_history in (0..=history.len()).step_by(2) {
            for retained_memory in (0..=memory.len()).rev() {
                let retained_history = &history[dropped_history..];
                let retained_memory_items = &memory[..retained_memory];
                let text = render_prompt(retained_history, retained_memory_items, user_message);
                let token_ids = tokenizer
                    .encode_text(&text, true)
                    .map_err(ChatPromptError::Tokenizer)?;
                if token_ids.len() <= max_prompt_tokens {
                    return Ok(CompiledChatPrompt {
                        text,
                        token_ids,
                        retained_history_turns: retained_history.len(),
                        dropped_history_turns: dropped_history,
                        retained_memory_items: retained_memory,
                        dropped_memory_items: memory.len() - retained_memory,
                    });
                }
            }
        }

        let text = render_prompt(&[], &[], user_message);
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

fn validate_inputs(
    history: &[ConversationTurn],
    memory: &[String],
    user_message: &str,
    max_prompt_tokens: usize,
) -> Result<(), ChatPromptError> {
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
        let expected = if index % 2 == 0 {
            ConversationRole::User
        } else {
            ConversationRole::Assistant
        };
        if turn.role != expected {
            return Err(ChatPromptError::InvalidHistoryOrder(index));
        }
    }
    if history.len() % 2 != 0 {
        return Err(ChatPromptError::InvalidHistoryOrder(history.len() - 1));
    }
    for item in memory {
        if item.trim().is_empty() {
            return Err(ChatPromptError::EmptyTurn(history.len()));
        }
    }
    Ok(())
}

fn render_prompt(history: &[ConversationTurn], memory: &[String], user_message: &str) -> String {
    let mut text = String::new();
    if !memory.is_empty() {
        text.push_str("Relevant memory:\n");
        for item in memory {
            text.push_str("- ");
            text.push_str(item);
            text.push('\n');
        }
        text.push_str("\n");
    }
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
            ConversationTurn::user("recent question"),
            ConversationTurn::assistant("recent answer"),
        ];
        let tokenizer = byte_tokenizer();
        let no_history_len = tokenizer
            .encode_text("User: now\nAssistant:", true)
            .expect("encode")
            .len();
        let recent_len = tokenizer
            .encode_text(
                "User: recent question\nAssistant: recent answer\nUser: now\nAssistant:",
                true,
            )
            .expect("encode")
            .len();

        let compiled = compiler
            .compile(&history, "now", &tokenizer, recent_len)
            .expect("compile");

        assert_eq!(compiled.retained_history_turns, 2);
        assert_eq!(compiled.dropped_history_turns, 2);
        assert_eq!(
            compiled.text,
            "User: recent question\nAssistant: recent answer\nUser: now\nAssistant:"
        );
        assert!(compiled.token_ids.len() > no_history_len);
    }

    #[test]
    fn memory_context_is_ranked_and_trimmed_before_recent_dialogue() {
        let compiler = NativeChatPromptCompiler;
        let history = vec![
            ConversationTurn::user("recent question"),
            ConversationTurn::assistant("recent answer"),
        ];
        let memory = vec!["highest relevance".to_owned(), "lower relevance".to_owned()];
        let tokenizer = byte_tokenizer();
        let one_memory = "Relevant memory:\n- highest relevance\n\nUser: recent question\nAssistant: recent answer\nUser: now\nAssistant:";
        let limit = tokenizer
            .encode_text(one_memory, true)
            .expect("encode")
            .len();

        let compiled = compiler
            .compile_with_memory(&history, &memory, "now", &tokenizer, limit)
            .expect("compile");

        assert_eq!(compiled.retained_history_turns, 2);
        assert_eq!(compiled.dropped_history_turns, 0);
        assert_eq!(compiled.retained_memory_items, 1);
        assert_eq!(compiled.dropped_memory_items, 1);
        assert!(compiled.text.contains("highest relevance"));
        assert!(!compiled.text.contains("lower relevance"));
    }

    #[test]
    fn rejects_non_alternating_history() {
        let compiler = NativeChatPromptCompiler;
        let history = vec![
            ConversationTurn::user("question"),
            ConversationTurn::user("second question"),
        ];

        assert_eq!(
            compiler.compile(&history, "now", &byte_tokenizer(), 128),
            Err(ChatPromptError::InvalidHistoryOrder(1))
        );
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
