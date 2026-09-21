#![forbid(unsafe_code)]

use ntd_core::{Intent, TaskGraph};

use crate::{
    decode_cognitive_checkpoint, encode_cognitive_checkpoint, CheckpointError, CognitiveError,
    CognitiveIdentity, CognitiveRuntime, ConversationRole, ConversationTurn, MemoryError, MemoryKind,
    TaskStatus,
};

pub const NCS97_MAGIC: [u8; 6] = *b"NCS97\0";
pub const NCS97_MAJOR: u16 = 0;
pub const NCS97_MINOR: u16 = 1;
pub const NCS97_HEADER_LEN: usize = 24;

const MAX_TURNS: usize = 4096;
const MAX_TEXT_BYTES: usize = 1024 * 1024;
const MAX_TOKENS: usize = 1_000_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveConversationTurn {
    pub task_id: u64,
    pub user_message: String,
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub generated_text: String,
    pub max_new_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SovereignConversationState {
    cognition: CognitiveRuntime,
    turns: Vec<ConversationTurn>,
    active: Option<ActiveConversationTurn>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConversationStateError {
    EmptyUserMessage,
    EmptyAssistantResponse,
    InvalidTokenBudget,
    ActiveTurnExists,
    MissingActiveTurn,
    TaskMismatch { expected: u64, actual: u64 },
    InvalidHistory,
    InvalidActiveTurn,
    LimitExceeded,
    Truncated,
    InvalidMagic,
    InvalidHeader,
    UnsupportedVersion { major: u16, minor: u16 },
    InvalidUtf8,
    InvalidRole(u8),
    NonCanonicalEncoding,
    Overflow,
    Cognitive(CognitiveError),
    Checkpoint(CheckpointError),
    Memory(MemoryError),
}

impl From<CognitiveError> for ConversationStateError {
    fn from(value: CognitiveError) -> Self {
        Self::Cognitive(value)
    }
}

impl From<CheckpointError> for ConversationStateError {
    fn from(value: CheckpointError) -> Self {
        Self::Checkpoint(value)
    }
}

impl From<MemoryError> for ConversationStateError {
    fn from(value: MemoryError) -> Self {
        Self::Memory(value)
    }
}

impl SovereignConversationState {
    pub fn new(identity: CognitiveIdentity) -> Self {
        Self {
            cognition: CognitiveRuntime::new(identity),
            turns: Vec::new(),
            active: None,
        }
    }

    pub fn from_parts(
        cognition: CognitiveRuntime,
        turns: Vec<ConversationTurn>,
        active: Option<ActiveConversationTurn>,
    ) -> Result<Self, ConversationStateError> {
        validate_turns(&turns)?;
        validate_active(&cognition, active.as_ref())?;
        Ok(Self {
            cognition,
            turns,
            active,
        })
    }

    pub fn cognition(&self) -> &CognitiveRuntime {
        &self.cognition
    }

    pub fn cognition_mut(&mut self) -> &mut CognitiveRuntime {
        &mut self.cognition
    }

    pub fn turns(&self) -> &[ConversationTurn] {
        &self.turns
    }

    pub fn active(&self) -> Option<&ActiveConversationTurn> {
        self.active.as_ref()
    }

    pub fn begin_turn(
        &mut self,
        user_message: impl Into<String>,
        prompt_tokens: Vec<u32>,
        max_new_tokens: usize,
    ) -> Result<u64, ConversationStateError> {
        if self.active.is_some() {
            return Err(ConversationStateError::ActiveTurnExists);
        }
        let user_message = user_message.into();
        if user_message.trim().is_empty() {
            return Err(ConversationStateError::EmptyUserMessage);
        }
        if max_new_tokens == 0 {
            return Err(ConversationStateError::InvalidTokenBudget);
        }
        if prompt_tokens.is_empty() || prompt_tokens.len() > MAX_TOKENS {
            return Err(ConversationStateError::InvalidActiveTurn);
        }

        let task_id = self.cognition.state_mut().submit_task(
            Intent::new(user_message.clone()),
            TaskGraph::default(),
            None,
        )?;
        self.cognition.start_external_task(task_id)?;
        self.active = Some(ActiveConversationTurn {
            task_id,
            user_message,
            prompt_tokens,
            generated_tokens: Vec::with_capacity(max_new_tokens.min(4096)),
            generated_text: String::new(),
            max_new_tokens,
        });
        Ok(task_id)
    }

    pub fn append_generated(
        &mut self,
        task_id: u64,
        token: u32,
        text_piece: &str,
    ) -> Result<(), ConversationStateError> {
        let active = self
            .active
            .as_mut()
            .ok_or(ConversationStateError::MissingActiveTurn)?;
        ensure_task(active.task_id, task_id)?;
        if active.generated_tokens.len() >= active.max_new_tokens {
            return Err(ConversationStateError::InvalidActiveTurn);
        }
        active.generated_tokens.push(token);
        active.generated_text.push_str(text_piece);
        if active.generated_text.len() > MAX_TEXT_BYTES {
            return Err(ConversationStateError::LimitExceeded);
        }
        Ok(())
    }

    pub fn complete_turn(&mut self, task_id: u64) -> Result<(), ConversationStateError> {
        let active = self
            .active
            .as_ref()
            .cloned()
            .ok_or(ConversationStateError::MissingActiveTurn)?;
        ensure_task(active.task_id, task_id)?;
        if active.generated_text.trim().is_empty() {
            return Err(ConversationStateError::EmptyAssistantResponse);
        }

        let mut cognition = self.cognition.clone();
        cognition.complete_external_task(task_id, active.generated_text.clone())?;
        let tick = cognition.state().tick;
        cognition.state_mut().memory.store(
            MemoryKind::Episodic,
            format!(
                "user: {}\nassistant: {}",
                active.user_message, active.generated_text
            ),
            vec!["conversation".into(), "dialogue".into()],
            650,
            tick,
        )?;

        let mut turns = self.turns.clone();
        turns.push(ConversationTurn::user(active.user_message));
        turns.push(ConversationTurn::assistant(active.generated_text));
        validate_turns(&turns)?;

        self.cognition = cognition;
        self.turns = turns;
        self.active = None;
        Ok(())
    }

    pub fn cancel_turn(
        &mut self,
        task_id: u64,
        reason: impl Into<String>,
    ) -> Result<(), ConversationStateError> {
        let active = self
            .active
            .as_ref()
            .ok_or(ConversationStateError::MissingActiveTurn)?;
        ensure_task(active.task_id, task_id)?;

        let mut cognition = self.cognition.clone();
        cognition.pause_external_task(task_id, reason)?;
        self.cognition = cognition;
        self.active = None;
        Ok(())
    }

    pub fn recall_conversation_context(
        &mut self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<String>, ConversationStateError> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let tick = self
            .cognition
            .state()
            .tick
            .checked_add(1)
            .ok_or(ConversationStateError::Overflow)?;
        self.cognition.state_mut().tick = tick;

        let mut memory_query = crate::MemoryQuery::new(query);
        memory_query.kinds.push(MemoryKind::Episodic);
        memory_query.tags.push("conversation".into());
        memory_query.limit = limit;
        Ok(self
            .cognition
            .state_mut()
            .memory
            .retrieve(&memory_query, tick)?
            .into_iter()
            .map(|hit| hit.record.content)
            .collect())
    }

    pub fn all_tokens_for_active(&self) -> Result<Vec<u32>, ConversationStateError> {
        let active = self
            .active
            .as_ref()
            .ok_or(ConversationStateError::MissingActiveTurn)?;
        let mut tokens = Vec::with_capacity(
            active
                .prompt_tokens
                .len()
                .checked_add(active.generated_tokens.len())
                .ok_or(ConversationStateError::Overflow)?,
        );
        tokens.extend_from_slice(&active.prompt_tokens);
        tokens.extend_from_slice(&active.generated_tokens);
        Ok(tokens)
    }
}

pub fn encode_conversation_checkpoint(
    state: &SovereignConversationState,
) -> Result<Vec<u8>, ConversationStateError> {
    validate_turns(&state.turns)?;
    validate_active(&state.cognition, state.active.as_ref())?;
    let cognitive = encode_cognitive_checkpoint(state.cognition.state())?;

    let mut payload = Vec::new();
    push_u64(&mut payload, len_u64(cognitive.len())?);
    payload.extend_from_slice(&cognitive);
    push_u32(&mut payload, len_u32(state.turns.len())?);
    for turn in &state.turns {
        push_u8(
            &mut payload,
            match turn.role {
                ConversationRole::User => 1,
                ConversationRole::Assistant => 2,
            },
        );
        push_string(&mut payload, &turn.content)?;
    }

    match &state.active {
        Some(active) => {
            push_u8(&mut payload, 1);
            push_u64(&mut payload, active.task_id);
            push_string(&mut payload, &active.user_message)?;
            push_u64(&mut payload, len_u64(active.max_new_tokens)?);
            push_tokens(&mut payload, &active.prompt_tokens)?;
            push_tokens(&mut payload, &active.generated_tokens)?;
            push_string(&mut payload, &active.generated_text)?;
        }
        None => push_u8(&mut payload, 0),
    }

    let payload_len = len_u64(payload.len())?;
    let mut out = Vec::with_capacity(
        NCS97_HEADER_LEN
            .checked_add(payload.len())
            .ok_or(ConversationStateError::Overflow)?,
    );
    out.extend_from_slice(&NCS97_MAGIC);
    push_u16(&mut out, NCS97_HEADER_LEN as u16);
    push_u16(&mut out, NCS97_MAJOR);
    push_u16(&mut out, NCS97_MINOR);
    push_u32(&mut out, 0);
    push_u64(&mut out, payload_len);
    out.extend_from_slice(&payload);
    Ok(out)
}

pub fn decode_conversation_checkpoint(
    bytes: &[u8],
) -> Result<SovereignConversationState, ConversationStateError> {
    if bytes.len() < NCS97_HEADER_LEN {
        return Err(ConversationStateError::Truncated);
    }
    if bytes.get(0..6) != Some(NCS97_MAGIC.as_slice()) {
        return Err(ConversationStateError::InvalidMagic);
    }

    let mut header = Cursor::new(bytes);
    header.take(6)?;
    if usize::from(header.u16()?) != NCS97_HEADER_LEN {
        return Err(ConversationStateError::InvalidHeader);
    }
    let major = header.u16()?;
    let minor = header.u16()?;
    if major != NCS97_MAJOR || minor > NCS97_MINOR {
        return Err(ConversationStateError::UnsupportedVersion { major, minor });
    }
    if header.u32()? != 0 {
        return Err(ConversationStateError::InvalidHeader);
    }
    let payload_len =
        usize::try_from(header.u64()?).map_err(|_| ConversationStateError::Overflow)?;
    let expected = NCS97_HEADER_LEN
        .checked_add(payload_len)
        .ok_or(ConversationStateError::Overflow)?;
    if bytes.len() != expected {
        return Err(ConversationStateError::NonCanonicalEncoding);
    }

    let mut cursor = Cursor::new(&bytes[NCS97_HEADER_LEN..]);
    let cognitive_len =
        usize::try_from(cursor.u64()?).map_err(|_| ConversationStateError::Overflow)?;
    let cognitive = decode_cognitive_checkpoint(cursor.take(cognitive_len)?)?;
    let cognition = CognitiveRuntime::from_state(cognitive)?;

    let turn_count =
        usize::try_from(cursor.u32()?).map_err(|_| ConversationStateError::Overflow)?;
    if turn_count > MAX_TURNS {
        return Err(ConversationStateError::LimitExceeded);
    }
    let mut turns = Vec::with_capacity(turn_count);
    for _ in 0..turn_count {
        let role = match cursor.u8()? {
            1 => ConversationRole::User,
            2 => ConversationRole::Assistant,
            other => return Err(ConversationStateError::InvalidRole(other)),
        };
        turns.push(ConversationTurn {
            role,
            content: cursor.string()?,
        });
    }

    let active = match cursor.u8()? {
        0 => None,
        1 => {
            let task_id = cursor.u64()?;
            let user_message = cursor.string()?;
            let max_new_tokens =
                usize::try_from(cursor.u64()?).map_err(|_| ConversationStateError::Overflow)?;
            let prompt_tokens = cursor.tokens()?;
            let generated_tokens = cursor.tokens()?;
            let generated_text = cursor.string()?;
            Some(ActiveConversationTurn {
                task_id,
                user_message,
                prompt_tokens,
                generated_tokens,
                generated_text,
                max_new_tokens,
            })
        }
        _ => return Err(ConversationStateError::NonCanonicalEncoding),
    };

    if !cursor.is_finished() {
        return Err(ConversationStateError::NonCanonicalEncoding);
    }

    SovereignConversationState::from_parts(cognition, turns, active)
}

fn validate_turns(turns: &[ConversationTurn]) -> Result<(), ConversationStateError> {
    if turns.len() > MAX_TURNS || turns.len() % 2 != 0 {
        return Err(ConversationStateError::InvalidHistory);
    }
    for (index, turn) in turns.iter().enumerate() {
        if turn.content.trim().is_empty() || turn.content.len() > MAX_TEXT_BYTES {
            return Err(ConversationStateError::InvalidHistory);
        }
        let expected = if index % 2 == 0 {
            ConversationRole::User
        } else {
            ConversationRole::Assistant
        };
        if turn.role != expected {
            return Err(ConversationStateError::InvalidHistory);
        }
    }
    Ok(())
}

fn validate_active(
    cognition: &CognitiveRuntime,
    active: Option<&ActiveConversationTurn>,
) -> Result<(), ConversationStateError> {
    let Some(active) = active else {
        return Ok(());
    };
    if active.task_id == 0
        || active.user_message.trim().is_empty()
        || active.user_message.len() > MAX_TEXT_BYTES
        || active.prompt_tokens.is_empty()
        || active.prompt_tokens.len() > MAX_TOKENS
        || active.generated_tokens.len() > active.max_new_tokens
        || active.generated_tokens.len() > MAX_TOKENS
        || active.generated_text.len() > MAX_TEXT_BYTES
        || active.max_new_tokens == 0
        || active.max_new_tokens > MAX_TOKENS
    {
        return Err(ConversationStateError::InvalidActiveTurn);
    }
    let task = cognition
        .state()
        .tasks
        .get(&active.task_id)
        .ok_or(ConversationStateError::InvalidActiveTurn)?;
    if matches!(task.status, TaskStatus::Completed | TaskStatus::Failed) {
        return Err(ConversationStateError::InvalidActiveTurn);
    }
    Ok(())
}

fn ensure_task(expected: u64, actual: u64) -> Result<(), ConversationStateError> {
    if expected == actual {
        Ok(())
    } else {
        Err(ConversationStateError::TaskMismatch { expected, actual })
    }
}

fn push_tokens(out: &mut Vec<u8>, tokens: &[u32]) -> Result<(), ConversationStateError> {
    if tokens.len() > MAX_TOKENS {
        return Err(ConversationStateError::LimitExceeded);
    }
    push_u32(out, len_u32(tokens.len())?);
    for token in tokens {
        push_u32(out, *token);
    }
    Ok(())
}

fn push_string(out: &mut Vec<u8>, value: &str) -> Result<(), ConversationStateError> {
    if value.len() > MAX_TEXT_BYTES {
        return Err(ConversationStateError::LimitExceeded);
    }
    push_u32(out, len_u32(value.len())?);
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

fn len_u32(value: usize) -> Result<u32, ConversationStateError> {
    u32::try_from(value).map_err(|_| ConversationStateError::Overflow)
}

fn len_u64(value: usize) -> Result<u64, ConversationStateError> {
    u64::try_from(value).map_err(|_| ConversationStateError::Overflow)
}

fn push_u8(out: &mut Vec<u8>, value: u8) {
    out.push(value);
}

fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
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

    fn take(&mut self, len: usize) -> Result<&'a [u8], ConversationStateError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(ConversationStateError::Overflow)?;
        let slice = self
            .bytes
            .get(self.offset..end)
            .ok_or(ConversationStateError::Truncated)?;
        self.offset = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, ConversationStateError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, ConversationStateError> {
        let mut bytes = [0u8; 2];
        bytes.copy_from_slice(self.take(2)?);
        Ok(u16::from_le_bytes(bytes))
    }

    fn u32(&mut self) -> Result<u32, ConversationStateError> {
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64, ConversationStateError> {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(bytes))
    }

    fn string(&mut self) -> Result<String, ConversationStateError> {
        let len = usize::try_from(self.u32()?).map_err(|_| ConversationStateError::Overflow)?;
        if len > MAX_TEXT_BYTES {
            return Err(ConversationStateError::LimitExceeded);
        }
        String::from_utf8(self.take(len)?.to_vec()).map_err(|_| ConversationStateError::InvalidUtf8)
    }

    fn tokens(&mut self) -> Result<Vec<u32>, ConversationStateError> {
        let len = usize::try_from(self.u32()?).map_err(|_| ConversationStateError::Overflow)?;
        if len > MAX_TOKENS {
            return Err(ConversationStateError::LimitExceeded);
        }
        let mut tokens = Vec::with_capacity(len);
        for _ in 0..len {
            tokens.push(self.u32()?);
        }
        Ok(tokens)
    }

    fn is_finished(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> CognitiveIdentity {
        CognitiveIdentity(*b"NTD97-CHATSTATE1")
    }

    #[test]
    fn completed_turn_becomes_cognitive_task_and_sovereign_memory() {
        let mut state = SovereignConversationState::new(identity());
        let task = state
            .begin_turn("remember this", vec![1, 2, 3], 4)
            .expect("begin");
        state
            .append_generated(task, 7, "remembered")
            .expect("append");
        state.complete_turn(task).expect("complete");

        assert_eq!(state.turns().len(), 2);
        assert_eq!(
            state
                .cognition()
                .state()
                .tasks
                .get(&task)
                .expect("task")
                .status,
            TaskStatus::Completed
        );
        let recalled = state
            .recall_conversation_context("remember", 4)
            .expect("recall");
        assert!(recalled.iter().any(|text| text.contains("remembered")));
    }

    #[test]
    fn cancelled_turn_does_not_commit_partial_conversation() {
        let mut state = SovereignConversationState::new(identity());
        let task = state
            .begin_turn("cancel me", vec![1], 8)
            .expect("begin");
        state.append_generated(task, 2, "partial").expect("append");
        state.cancel_turn(task, "user cancelled").expect("cancel");

        assert!(state.turns().is_empty());
        assert!(state.active().is_none());
        assert_eq!(
            state
                .cognition()
                .state()
                .tasks
                .get(&task)
                .expect("task")
                .status,
            TaskStatus::Paused
        );
    }

    #[test]
    fn checkpoint_round_trip_preserves_active_generation_prefix() {
        let mut state = SovereignConversationState::new(identity());
        let first = state
            .begin_turn("first", vec![10, 11], 4)
            .expect("first");
        state.append_generated(first, 12, "answer").expect("append");
        state.complete_turn(first).expect("complete");

        let active = state
            .begin_turn("second", vec![20, 21], 8)
            .expect("active");
        state.append_generated(active, 22, "par").expect("append");
        state.append_generated(active, 23, "tial").expect("append");

        let encoded = encode_conversation_checkpoint(&state).expect("encode");
        let restored = decode_conversation_checkpoint(&encoded).expect("decode");

        assert_eq!(restored, state);
        assert_eq!(
            restored.all_tokens_for_active().expect("tokens"),
            vec![20, 21, 22, 23]
        );
    }

    #[test]
    fn checkpoint_rejects_noncanonical_trailing_bytes() {
        let state = SovereignConversationState::new(identity());
        let mut encoded = encode_conversation_checkpoint(&state).expect("encode");
        encoded.push(0);
        assert_eq!(
            decode_conversation_checkpoint(&encoded),
            Err(ConversationStateError::NonCanonicalEncoding)
        );
    }
}
