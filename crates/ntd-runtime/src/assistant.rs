#![forbid(unsafe_code)]

use std::collections::BTreeSet;

use ntd_core::{Intent, ReasoningBudget, TaskGraph};

use crate::{
    decode_cognitive_checkpoint, encode_cognitive_checkpoint, CheckpointError,
    AssistantActionPlan, CognitiveCycleReport, CognitiveError, CognitiveIdentity, CognitiveRuntime,
    CognitiveSignals, ConversationRole, ConversationTurn, MemoryError, MemoryKind, TaskStatus,
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
    pub model_asset_id: String,
    pub model_version: u32,
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
    InvalidTaskGraph,
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
        model_asset_id: impl Into<String>,
        model_version: u32,
        user_message: impl Into<String>,
        prompt_tokens: Vec<u32>,
        max_new_tokens: usize,
    ) -> Result<u64, ConversationStateError> {
        if self.active.is_some() {
            return Err(ConversationStateError::ActiveTurnExists);
        }
        let model_asset_id = model_asset_id.into();
        if model_asset_id.trim().is_empty() || model_version == 0 {
            return Err(ConversationStateError::InvalidActiveTurn);
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

        let mut cognition = self.cognition.clone();
        let task_id = cognition.state_mut().submit_task(
            Intent::new(user_message.clone()),
            TaskGraph::default(),
            None,
        )?;
        cognition.start_external_task(task_id)?;
        self.cognition = cognition;
        self.active = Some(ActiveConversationTurn {
            task_id,
            model_asset_id,
            model_version,
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
        let next_text_len = active
            .generated_text
            .len()
            .checked_add(text_piece.len())
            .ok_or(ConversationStateError::Overflow)?;
        if next_text_len > MAX_TEXT_BYTES {
            return Err(ConversationStateError::LimitExceeded);
        }
        active.generated_tokens.push(token);
        active.generated_text.push_str(text_piece);
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

    pub fn prior_conversation_failure(&self) -> bool {
        self.cognition
            .state()
            .tasks
            .values()
            .next_back()
            .is_some_and(|task| matches!(task.status, TaskStatus::Paused | TaskStatus::Failed))
    }

    pub fn install_action_task_graph(
        &mut self,
        task_id: u64,
        graph: TaskGraph,
    ) -> Result<(), ConversationStateError> {
        let active = self
            .active
            .as_ref()
            .ok_or(ConversationStateError::MissingActiveTurn)?;
        ensure_task(active.task_id, task_id)?;
        if graph.actions.is_empty() {
            return Err(ConversationStateError::InvalidTaskGraph);
        }

        let mut seen = BTreeSet::new();
        for node in &graph.actions {
            if node.id == 0
                || !seen.insert(node.id)
                || node.capability.0.trim().is_empty()
            {
                return Err(ConversationStateError::InvalidTaskGraph);
            }
        }

        let mut cognition = self.cognition.clone();
        let task = cognition
            .state_mut()
            .tasks
            .get_mut(&task_id)
            .ok_or(ConversationStateError::InvalidTaskGraph)?;
        if matches!(task.status, TaskStatus::Completed | TaskStatus::Failed) {
            return Err(ConversationStateError::InvalidTaskGraph);
        }
        task.graph = graph;
        let action_count = task.graph.actions.len();
        cognition.state_mut().set_world_fact(
            format!("conversation.task.{task_id}.action_count"),
            action_count.to_string(),
        )?;
        self.cognition = cognition;
        Ok(())
    }

    pub fn install_action_plan(
        &mut self,
        task_id: u64,
        plan: &AssistantActionPlan,
    ) -> Result<(), ConversationStateError> {
        if plan.canonical_text.len() > MAX_TEXT_BYTES {
            return Err(ConversationStateError::LimitExceeded);
        }
        let mut candidate = self.clone();
        candidate.install_action_task_graph(task_id, plan.graph.clone())?;
        candidate.cognition.state_mut().set_world_fact(
            format!("conversation.task.{task_id}.action_protocol"),
            plan.canonical_text.clone(),
        )?;
        *self = candidate;
        Ok(())
    }

    pub fn action_plan_protocol_for_task(&self, task_id: u64) -> Option<&str> {
        let key = format!("conversation.task.{task_id}.action_protocol");
        self.cognition
            .state()
            .world
            .get(&key)
            .map(|fact| fact.value.as_str())
    }

    pub fn action_count_for_task(&self, task_id: u64) -> Option<usize> {
        let key = format!("conversation.task.{task_id}.action_count");
        self.cognition
            .state()
            .world
            .get(&key)?
            .value
            .parse::<usize>()
            .ok()
    }

    pub fn record_action_planner_status(
        &mut self,
        task_id: u64,
        status: &str,
    ) -> Result<(), ConversationStateError> {
        if !matches!(status, "direct" | "actions" | "invalid" | "unsupported") {
            return Err(ConversationStateError::InvalidTaskGraph);
        }
        if !self.cognition.state().tasks.contains_key(&task_id) {
            return Err(ConversationStateError::TaskMismatch {
                expected: task_id,
                actual: 0,
            });
        }
        self.cognition.state_mut().set_world_fact(
            format!("conversation.task.{task_id}.action_planner_status"),
            status,
        )?;
        Ok(())
    }

    pub fn action_planner_status_for_task(&self, task_id: u64) -> Option<&str> {
        let key = format!("conversation.task.{task_id}.action_planner_status");
        self.cognition
            .state()
            .world
            .get(&key)
            .map(|fact| fact.value.as_str())
    }

    pub fn record_verified_action_count(
        &mut self,
        task_id: u64,
        count: usize,
    ) -> Result<(), ConversationStateError> {
        if !self.cognition.state().tasks.contains_key(&task_id) {
            return Err(ConversationStateError::TaskMismatch {
                expected: task_id,
                actual: 0,
            });
        }
        self.cognition.state_mut().set_world_fact(
            format!("conversation.task.{task_id}.verified_action_count"),
            count.to_string(),
        )?;
        Ok(())
    }

    pub fn verified_action_count_for_task(&self, task_id: u64) -> Option<usize> {
        let key = format!("conversation.task.{task_id}.verified_action_count");
        self.cognition
            .state()
            .world
            .get(&key)?
            .value
            .parse::<usize>()
            .ok()
    }

    pub fn replace_active_prompt_tokens(
        &mut self,
        task_id: u64,
        prompt_tokens: Vec<u32>,
    ) -> Result<(), ConversationStateError> {
        if prompt_tokens.is_empty() || prompt_tokens.len() > MAX_TOKENS {
            return Err(ConversationStateError::InvalidActiveTurn);
        }
        let active = self
            .active
            .as_mut()
            .ok_or(ConversationStateError::MissingActiveTurn)?;
        ensure_task(active.task_id, task_id)?;
        if !active.generated_tokens.is_empty() || !active.generated_text.is_empty() {
            return Err(ConversationStateError::InvalidActiveTurn);
        }
        active.prompt_tokens = prompt_tokens;
        Ok(())
    }

    pub fn record_reasoning_profile(
        &mut self,
        task_id: u64,
        budget: ReasoningBudget,
        signals: CognitiveSignals,
        recalled_memory_items: usize,
    ) -> Result<(), ConversationStateError> {
        if !self.cognition.state().tasks.contains_key(&task_id) {
            return Err(ConversationStateError::TaskMismatch {
                expected: task_id,
                actual: 0,
            });
        }
        let prefix = format!("conversation.task.{task_id}");
        self.cognition.state_mut().set_world_fact(
            format!("{prefix}.reasoning_budget"),
            reasoning_budget_name(budget),
        )?;
        self.cognition.state_mut().set_world_fact(
            format!("{prefix}.complexity_milli"),
            signal_milli(signals.complexity).to_string(),
        )?;
        self.cognition.state_mut().set_world_fact(
            format!("{prefix}.uncertainty_milli"),
            signal_milli(signals.uncertainty).to_string(),
        )?;
        self.cognition.state_mut().set_world_fact(
            format!("{prefix}.recalled_memory_items"),
            recalled_memory_items.to_string(),
        )?;
        Ok(())
    }

    pub fn record_reasoning_cycle_report(
        &mut self,
        task_id: u64,
        report: &CognitiveCycleReport,
    ) -> Result<(), ConversationStateError> {
        if report.task_id != task_id {
            return Err(ConversationStateError::TaskMismatch {
                expected: task_id,
                actual: report.task_id,
            });
        }
        if !self.cognition.state().tasks.contains_key(&task_id) {
            return Err(ConversationStateError::TaskMismatch {
                expected: task_id,
                actual: 0,
            });
        }

        let prefix = format!("conversation.task.{task_id}");
        self.cognition.state_mut().set_world_fact(
            format!("{prefix}.reasoning_iterations"),
            report.iterations.to_string(),
        )?;
        self.cognition.state_mut().set_world_fact(
            format!("{prefix}.reasoning_verification_failures"),
            report.verification_failures.to_string(),
        )?;
        Ok(())
    }

    pub fn reasoning_budget_for_task(&self, task_id: u64) -> Option<ReasoningBudget> {
        let key = format!("conversation.task.{task_id}.reasoning_budget");
        let value = self.cognition.state().world.get(&key)?.value.as_str();
        match value {
            "reflex" => Some(ReasoningBudget::Reflex),
            "standard" => Some(ReasoningBudget::Standard),
            "deep" => Some(ReasoningBudget::Deep),
            "recovery" => Some(ReasoningBudget::Recovery),
            _ => None,
        }
    }

    pub fn reasoning_iterations_for_task(&self, task_id: u64) -> Option<u32> {
        let key = format!("conversation.task.{task_id}.reasoning_iterations");
        self.cognition
            .state()
            .world
            .get(&key)?
            .value
            .parse::<u32>()
            .ok()
    }

    pub fn reasoning_verification_failures_for_task(&self, task_id: u64) -> Option<u32> {
        let key = format!("conversation.task.{task_id}.reasoning_verification_failures");
        self.cognition
            .state()
            .world
            .get(&key)?
            .value
            .parse::<u32>()
            .ok()
    }

    pub fn recalled_memory_items_for_task(&self, task_id: u64) -> Option<usize> {
        let key = format!("conversation.task.{task_id}.recalled_memory_items");
        self.cognition
            .state()
            .world
            .get(&key)?
            .value
            .parse::<usize>()
            .ok()
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

pub fn memory_recall_limit_for_budget(budget: ReasoningBudget) -> usize {
    match budget {
        ReasoningBudget::Reflex => 1,
        ReasoningBudget::Standard => 2,
        ReasoningBudget::Deep => 4,
        ReasoningBudget::Recovery => 6,
    }
}

fn reasoning_budget_name(budget: ReasoningBudget) -> &'static str {
    match budget {
        ReasoningBudget::Reflex => "reflex",
        ReasoningBudget::Standard => "standard",
        ReasoningBudget::Deep => "deep",
        ReasoningBudget::Recovery => "recovery",
    }
}

fn signal_milli(value: f32) -> u16 {
    (value.clamp(0.0, 1.0) * 1000.0).round() as u16
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
            push_string(&mut payload, &active.model_asset_id)?;
            push_u32(&mut payload, active.model_version);
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
            let model_asset_id = cursor.string()?;
            let model_version = cursor.u32()?;
            let user_message = cursor.string()?;
            let max_new_tokens =
                usize::try_from(cursor.u64()?).map_err(|_| ConversationStateError::Overflow)?;
            let prompt_tokens = cursor.tokens()?;
            let generated_tokens = cursor.tokens()?;
            let generated_text = cursor.string()?;
            Some(ActiveConversationTurn {
                task_id,
                model_asset_id,
                model_version,
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
        || active.model_asset_id.trim().is_empty()
        || active.model_asset_id.len() > MAX_TEXT_BYTES
        || active.model_version == 0
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
    if task.intent.objective != active.user_message
        || matches!(task.status, TaskStatus::Completed | TaskStatus::Failed)
    {
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
            .begin_turn("model.test", 1, "remember this", vec![1, 2, 3], 4)
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
            .begin_turn("model.test", 1, "cancel me", vec![1], 8)
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
    fn reasoning_profile_is_sovereign_and_survives_checkpoint() {
        let mut state = SovereignConversationState::new(identity());
        let task = state
            .begin_turn("model.test", 1, "analyze this", vec![1, 2], 8)
            .expect("begin");
        state
            .record_reasoning_profile(
                task,
                ReasoningBudget::Deep,
                CognitiveSignals {
                    complexity: 0.72,
                    uncertainty: 0.81,
                    prior_failure: false,
                },
                4,
            )
            .expect("profile");

        let encoded = encode_conversation_checkpoint(&state).expect("encode");
        let restored = decode_conversation_checkpoint(&encoded).expect("decode");

        assert_eq!(
            restored.reasoning_budget_for_task(task),
            Some(ReasoningBudget::Deep)
        );
        assert_eq!(restored.recalled_memory_items_for_task(task), Some(4));
    }

    #[test]
    fn action_planner_and_verified_result_state_survive_ncs97_checkpoint() {
        let mut state = SovereignConversationState::new(identity());
        let task = state
            .begin_turn("model.test", 1, "observe device", vec![1, 2], 8)
            .expect("begin");
        state
            .record_action_planner_status(task, "direct")
            .expect("status");
        state
            .record_verified_action_count(task, 0)
            .expect("verified count");
        state
            .replace_active_prompt_tokens(task, vec![3, 4, 5])
            .expect("replace prompt");

        let encoded = encode_conversation_checkpoint(&state).expect("encode");
        let restored = decode_conversation_checkpoint(&encoded).expect("decode");

        assert_eq!(
            restored.action_planner_status_for_task(task),
            Some("direct")
        );
        assert_eq!(restored.verified_action_count_for_task(task), Some(0));
        assert_eq!(
            restored.active().expect("active").prompt_tokens,
            vec![3, 4, 5]
        );
    }

    #[test]
    fn installed_action_task_graph_survives_ncs97_checkpoint() {
        use ntd_core::{ActionNode, CapabilityId, SideEffectClass};

        let mut state = SovereignConversationState::new(identity());
        let task = state
            .begin_turn("model.test", 1, "observe device", vec![1, 2], 8)
            .expect("begin");
        let plan = AssistantActionPlan {
            graph: TaskGraph {
                actions: vec![ActionNode {
                    id: 1,
                    capability: CapabilityId("device.observe".into()),
                    side_effect: SideEffectClass::ReadOnly,
                    verification_required: true,
                }],
            },
            payloads: std::collections::BTreeMap::from([(
                1,
                crate::TypedAction::DeviceObserve {
                    surface: "battery".into(),
                },
            )]),
            canonical_text: "NTD97_ACTIONS_V1\n1|device.observe|battery\nEND".into(),
        };
        state.install_action_plan(task, &plan).expect("install plan");

        let encoded = encode_conversation_checkpoint(&state).expect("encode");
        let restored = decode_conversation_checkpoint(&encoded).expect("decode");
        let restored_task = restored
            .cognition()
            .state()
            .tasks
            .get(&task)
            .expect("task");

        assert_eq!(restored_task.graph.actions.len(), 1);
        assert_eq!(restored.action_count_for_task(task), Some(1));
        assert_eq!(
            restored.action_plan_protocol_for_task(task),
            Some("NTD97_ACTIONS_V1\n1|device.observe|battery\nEND")
        );
    }

    #[test]
    fn reasoning_cycle_report_survives_ncs97_checkpoint() {
        let mut state = SovereignConversationState::new(identity());
        let task = state
            .begin_turn("model.test", 1, "reason first", vec![1, 2], 8)
            .expect("begin");
        state
            .record_reasoning_cycle_report(
                task,
                &CognitiveCycleReport {
                    task_id: task,
                    budget: ReasoningBudget::Deep,
                    iterations: 4,
                    status: TaskStatus::Running,
                    verification_failures: 1,
                    last_observation: Some("verified probe".into()),
                },
            )
            .expect("report");

        let encoded = encode_conversation_checkpoint(&state).expect("encode");
        let restored = decode_conversation_checkpoint(&encoded).expect("decode");

        assert_eq!(restored.reasoning_iterations_for_task(task), Some(4));
        assert_eq!(
            restored.reasoning_verification_failures_for_task(task),
            Some(1)
        );
    }

    #[test]
    fn paused_conversation_marks_next_turn_as_prior_failure() {
        let mut state = SovereignConversationState::new(identity());
        let task = state
            .begin_turn("model.test", 1, "cancel me", vec![1], 8)
            .expect("begin");
        state.cancel_turn(task, "cancelled").expect("cancel");
        assert!(state.prior_conversation_failure());
        assert_eq!(memory_recall_limit_for_budget(ReasoningBudget::Recovery), 6);
    }

    #[test]
    fn checkpoint_round_trip_preserves_active_generation_prefix() {
        let mut state = SovereignConversationState::new(identity());
        let first = state
            .begin_turn("model.test", 1, "first", vec![10, 11], 4)
            .expect("first");
        state.append_generated(first, 12, "answer").expect("append");
        state.complete_turn(first).expect("complete");

        let active = state
            .begin_turn("model.test", 1, "second", vec![20, 21], 8)
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
