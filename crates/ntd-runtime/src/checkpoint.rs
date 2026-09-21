#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use ntd_core::{
    ActionNode, CapabilityId, Intent, SideEffectClass, TaskGraph,
};

use crate::{
    CognitiveError, CognitiveIdentity, CognitiveState, CognitiveTask, Goal, GoalStatus, LearnedDelta,
    MemoryError, MemoryKind, MemoryRecord, SovereignMemory, TaskStatus, WorldFact,
};

pub const SIK97_MAGIC: [u8; 6] = *b"SIK97\0";
pub const SIK97_MAJOR: u16 = 0;
pub const SIK97_MINOR: u16 = 1;
pub const SIK97_HEADER_LEN: usize = 24;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckpointError {
    Truncated,
    InvalidMagic,
    InvalidHeader,
    UnsupportedVersion { major: u16, minor: u16 },
    InvalidUtf8,
    InvalidBoolean(u8),
    InvalidSideEffect(u8),
    InvalidGoalStatus(u8),
    InvalidTaskStatus(u8),
    InvalidMemoryKind(u8),
    NonCanonicalEncoding,
    Overflow,
    Cognitive(CognitiveError),
    Memory(MemoryError),
}

impl From<CognitiveError> for CheckpointError {
    fn from(value: CognitiveError) -> Self {
        Self::Cognitive(value)
    }
}

impl From<MemoryError> for CheckpointError {
    fn from(value: MemoryError) -> Self {
        Self::Memory(value)
    }
}

pub fn encode_cognitive_checkpoint(
    state: &CognitiveState,
) -> Result<Vec<u8>, CheckpointError> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&state.identity.0);
    push_u64(&mut payload, state.tick);
    push_u64(&mut payload, state.next_goal_id);
    push_u64(&mut payload, state.next_task_id);
    push_u64(&mut payload, state.next_delta_id);
    push_u64(&mut payload, state.memory.next_id());

    push_u32(&mut payload, len_u32(state.goals.len())?);
    push_u32(&mut payload, len_u32(state.tasks.len())?);
    push_u32(&mut payload, len_u32(state.world.len())?);
    push_u32(&mut payload, len_u32(state.memory.records().len())?);
    push_u32(&mut payload, len_u32(state.deltas.len())?);

    for goal in state.goals.values() {
        encode_goal(&mut payload, goal)?;
    }
    for task in state.tasks.values() {
        encode_task(&mut payload, task)?;
    }
    for fact in state.world.values() {
        encode_world_fact(&mut payload, fact)?;
    }
    for record in state.memory.records().values() {
        encode_memory_record(&mut payload, record)?;
    }
    for delta in state.deltas.values() {
        encode_delta(&mut payload, delta)?;
    }

    let payload_len = u64::try_from(payload.len()).map_err(|_| CheckpointError::Overflow)?;
    let mut out = Vec::with_capacity(
        SIK97_HEADER_LEN
            .checked_add(payload.len())
            .ok_or(CheckpointError::Overflow)?,
    );
    out.extend_from_slice(&SIK97_MAGIC);
    push_u16(&mut out, SIK97_HEADER_LEN as u16);
    push_u16(&mut out, SIK97_MAJOR);
    push_u16(&mut out, SIK97_MINOR);
    push_u32(&mut out, 0);
    push_u64(&mut out, payload_len);
    out.extend_from_slice(&payload);
    Ok(out)
}

pub fn decode_cognitive_checkpoint(bytes: &[u8]) -> Result<CognitiveState, CheckpointError> {
    if bytes.len() < SIK97_HEADER_LEN {
        return Err(CheckpointError::Truncated);
    }
    if bytes.get(0..6) != Some(SIK97_MAGIC.as_slice()) {
        return Err(CheckpointError::InvalidMagic);
    }

    let mut header = Cursor::new(bytes);
    header.take(6)?;
    if usize::from(header.u16()?) != SIK97_HEADER_LEN {
        return Err(CheckpointError::InvalidHeader);
    }
    let major = header.u16()?;
    let minor = header.u16()?;
    if major != SIK97_MAJOR || minor > SIK97_MINOR {
        return Err(CheckpointError::UnsupportedVersion { major, minor });
    }
    if header.u32()? != 0 {
        return Err(CheckpointError::InvalidHeader);
    }
    let payload_len = usize::try_from(header.u64()?).map_err(|_| CheckpointError::Overflow)?;
    let expected_len = SIK97_HEADER_LEN
        .checked_add(payload_len)
        .ok_or(CheckpointError::Overflow)?;
    if bytes.len() != expected_len {
        return Err(CheckpointError::NonCanonicalEncoding);
    }

    let mut cursor = Cursor::new(&bytes[SIK97_HEADER_LEN..]);
    let mut identity = [0u8; 16];
    identity.copy_from_slice(cursor.take(16)?);

    let tick = cursor.u64()?;
    let next_goal_id = cursor.u64()?;
    let next_task_id = cursor.u64()?;
    let next_delta_id = cursor.u64()?;
    let next_memory_id = cursor.u64()?;

    let goal_count = cursor.u32()?;
    let task_count = cursor.u32()?;
    let world_count = cursor.u32()?;
    let memory_count = cursor.u32()?;
    let delta_count = cursor.u32()?;

    let mut goals = BTreeMap::new();
    let mut last_goal_id = 0u64;
    for _ in 0..goal_count {
        let goal = decode_goal(&mut cursor)?;
        ensure_increasing(goal.id, &mut last_goal_id)?;
        if goals.insert(goal.id, goal).is_some() {
            return Err(CheckpointError::NonCanonicalEncoding);
        }
    }

    let mut tasks = BTreeMap::new();
    let mut last_task_id = 0u64;
    for _ in 0..task_count {
        let task = decode_task(&mut cursor)?;
        ensure_increasing(task.id, &mut last_task_id)?;
        if tasks.insert(task.id, task).is_some() {
            return Err(CheckpointError::NonCanonicalEncoding);
        }
    }

    let mut world = BTreeMap::new();
    let mut last_world_key: Option<String> = None;
    for _ in 0..world_count {
        let fact = decode_world_fact(&mut cursor)?;
        if let Some(previous) = &last_world_key {
            if fact.key <= *previous {
                return Err(CheckpointError::NonCanonicalEncoding);
            }
        }
        last_world_key = Some(fact.key.clone());
        if world.insert(fact.key.clone(), fact).is_some() {
            return Err(CheckpointError::NonCanonicalEncoding);
        }
    }

    let mut records = BTreeMap::new();
    let mut last_memory_id = 0u64;
    for _ in 0..memory_count {
        let record = decode_memory_record(&mut cursor)?;
        ensure_increasing(record.id, &mut last_memory_id)?;
        if records.insert(record.id, record).is_some() {
            return Err(CheckpointError::NonCanonicalEncoding);
        }
    }

    let mut deltas = BTreeMap::new();
    let mut last_delta_id = 0u64;
    for _ in 0..delta_count {
        let delta = decode_delta(&mut cursor)?;
        ensure_increasing(delta.id, &mut last_delta_id)?;
        if deltas.insert(delta.id, delta).is_some() {
            return Err(CheckpointError::NonCanonicalEncoding);
        }
    }

    if !cursor.is_finished() {
        return Err(CheckpointError::NonCanonicalEncoding);
    }

    let memory = SovereignMemory::from_parts(next_memory_id, records)?;
    let state = CognitiveState {
        identity: CognitiveIdentity(identity),
        tick,
        next_goal_id,
        next_task_id,
        next_delta_id,
        goals,
        tasks,
        world,
        memory,
        deltas,
    };

    crate::CognitiveRuntime::from_state(state.clone())?;
    Ok(state)
}

fn encode_goal(out: &mut Vec<u8>, goal: &Goal) -> Result<(), CheckpointError> {
    push_u64(out, goal.id);
    push_u8(out, goal.status as u8);
    push_u64(out, goal.created_tick);
    push_u64(out, goal.updated_tick);
    push_string(out, &goal.objective)?;
    push_strings(out, &goal.completion_criteria)
}

fn decode_goal(cursor: &mut Cursor<'_>) -> Result<Goal, CheckpointError> {
    let id = cursor.u64()?;
    let status_raw = cursor.u8()?;
    let status =
        GoalStatus::try_from(status_raw).map_err(|_| CheckpointError::InvalidGoalStatus(status_raw))?;
    let created_tick = cursor.u64()?;
    let updated_tick = cursor.u64()?;
    let objective = cursor.string()?;
    let completion_criteria = cursor.strings()?;
    Ok(Goal {
        id,
        objective,
        completion_criteria,
        status,
        created_tick,
        updated_tick,
    })
}

fn encode_task(out: &mut Vec<u8>, task: &CognitiveTask) -> Result<(), CheckpointError> {
    push_u64(out, task.id);
    push_u64(out, task.goal_id.unwrap_or(0));
    push_u8(out, task.status as u8);
    push_u32(out, task.steps_taken);
    push_u32(out, task.verification_failures);
    push_u64(out, task.created_tick);
    push_u64(out, task.updated_tick);

    push_string(out, &task.intent.objective)?;
    push_strings(out, &task.intent.completion_criteria)?;
    push_u32(out, task.intent.max_steps);

    push_u32(out, len_u32(task.graph.actions.len())?);
    for action in &task.graph.actions {
        encode_action(out, action)?;
    }

    push_optional_string(out, task.last_observation.as_deref())
}

fn decode_task(cursor: &mut Cursor<'_>) -> Result<CognitiveTask, CheckpointError> {
    let id = cursor.u64()?;
    let raw_goal = cursor.u64()?;
    let goal_id = (raw_goal != 0).then_some(raw_goal);
    let status_raw = cursor.u8()?;
    let status =
        TaskStatus::try_from(status_raw).map_err(|_| CheckpointError::InvalidTaskStatus(status_raw))?;
    let steps_taken = cursor.u32()?;
    let verification_failures = cursor.u32()?;
    let created_tick = cursor.u64()?;
    let updated_tick = cursor.u64()?;

    let intent = Intent {
        objective: cursor.string()?,
        completion_criteria: cursor.strings()?,
        max_steps: cursor.u32()?,
    };

    let action_count = cursor.u32()?;
    let mut actions = Vec::with_capacity(
        usize::try_from(action_count).map_err(|_| CheckpointError::Overflow)?,
    );
    for _ in 0..action_count {
        actions.push(decode_action(cursor)?);
    }

    let last_observation = cursor.optional_string()?;

    Ok(CognitiveTask {
        id,
        goal_id,
        intent,
        graph: TaskGraph { actions },
        status,
        steps_taken,
        verification_failures,
        last_observation,
        created_tick,
        updated_tick,
    })
}

fn encode_action(out: &mut Vec<u8>, action: &ActionNode) -> Result<(), CheckpointError> {
    push_u32(out, action.id);
    push_string(out, &action.capability.0)?;
    push_u8(out, encode_side_effect(action.side_effect));
    push_u8(out, u8::from(action.verification_required));
    Ok(())
}

fn decode_action(cursor: &mut Cursor<'_>) -> Result<ActionNode, CheckpointError> {
    let id = cursor.u32()?;
    let capability = CapabilityId(cursor.string()?);
    let raw_side_effect = cursor.u8()?;
    let side_effect = decode_side_effect(raw_side_effect)?;
    let verification_required = cursor.boolean()?;
    Ok(ActionNode {
        id,
        capability,
        side_effect,
        verification_required,
    })
}

fn encode_world_fact(out: &mut Vec<u8>, fact: &WorldFact) -> Result<(), CheckpointError> {
    push_string(out, &fact.key)?;
    push_string(out, &fact.value)?;
    push_u64(out, fact.revision);
    push_u64(out, fact.updated_tick);
    Ok(())
}

fn decode_world_fact(cursor: &mut Cursor<'_>) -> Result<WorldFact, CheckpointError> {
    Ok(WorldFact {
        key: cursor.string()?,
        value: cursor.string()?,
        revision: cursor.u64()?,
        updated_tick: cursor.u64()?,
    })
}

fn encode_memory_record(
    out: &mut Vec<u8>,
    record: &MemoryRecord,
) -> Result<(), CheckpointError> {
    push_u64(out, record.id);
    push_u8(out, record.kind as u8);
    push_u16(out, record.importance);
    push_u64(out, record.created_tick);
    push_u64(out, record.last_recalled_tick);
    push_u32(out, record.recall_count);
    push_string(out, &record.content)?;
    push_strings(out, &record.tags)
}

fn decode_memory_record(cursor: &mut Cursor<'_>) -> Result<MemoryRecord, CheckpointError> {
    let id = cursor.u64()?;
    let kind_raw = cursor.u8()?;
    let kind = MemoryKind::try_from(kind_raw)
        .map_err(|_| CheckpointError::InvalidMemoryKind(kind_raw))?;
    let importance = cursor.u16()?;
    let created_tick = cursor.u64()?;
    let last_recalled_tick = cursor.u64()?;
    let recall_count = cursor.u32()?;
    let content = cursor.string()?;
    let tags = cursor.strings()?;

    if !is_strictly_sorted_unique(&tags) {
        return Err(CheckpointError::NonCanonicalEncoding);
    }

    Ok(MemoryRecord {
        id,
        kind,
        content,
        tags,
        importance,
        created_tick,
        last_recalled_tick,
        recall_count,
    })
}

fn encode_delta(out: &mut Vec<u8>, delta: &LearnedDelta) -> Result<(), CheckpointError> {
    push_u64(out, delta.id);
    push_u64(out, delta.generation);
    push_u8(out, u8::from(delta.active));
    push_string(out, &delta.namespace)?;
    push_bytes(out, &delta.payload)
}

fn decode_delta(cursor: &mut Cursor<'_>) -> Result<LearnedDelta, CheckpointError> {
    Ok(LearnedDelta {
        id: cursor.u64()?,
        generation: cursor.u64()?,
        active: cursor.boolean()?,
        namespace: cursor.string()?,
        payload: cursor.bytes()?.to_vec(),
    })
}

fn encode_side_effect(value: SideEffectClass) -> u8 {
    match value {
        SideEffectClass::ReadOnly => 1,
        SideEffectClass::Reversible => 2,
        SideEffectClass::ExternalWrite => 3,
        SideEffectClass::Irreversible => 4,
    }
}

fn decode_side_effect(value: u8) -> Result<SideEffectClass, CheckpointError> {
    match value {
        1 => Ok(SideEffectClass::ReadOnly),
        2 => Ok(SideEffectClass::Reversible),
        3 => Ok(SideEffectClass::ExternalWrite),
        4 => Ok(SideEffectClass::Irreversible),
        other => Err(CheckpointError::InvalidSideEffect(other)),
    }
}

fn ensure_increasing(value: u64, previous: &mut u64) -> Result<(), CheckpointError> {
    if value == 0 || value <= *previous {
        return Err(CheckpointError::NonCanonicalEncoding);
    }
    *previous = value;
    Ok(())
}

fn is_strictly_sorted_unique(values: &[String]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

fn len_u32(value: usize) -> Result<u32, CheckpointError> {
    u32::try_from(value).map_err(|_| CheckpointError::Overflow)
}

fn push_optional_string(
    out: &mut Vec<u8>,
    value: Option<&str>,
) -> Result<(), CheckpointError> {
    match value {
        Some(value) => {
            push_u8(out, 1);
            push_string(out, value)
        }
        None => {
            push_u8(out, 0);
            Ok(())
        }
    }
}

fn push_strings(out: &mut Vec<u8>, values: &[String]) -> Result<(), CheckpointError> {
    push_u32(out, len_u32(values.len())?);
    for value in values {
        push_string(out, value)?;
    }
    Ok(())
}

fn push_string(out: &mut Vec<u8>, value: &str) -> Result<(), CheckpointError> {
    push_bytes(out, value.as_bytes())
}

fn push_bytes(out: &mut Vec<u8>, value: &[u8]) -> Result<(), CheckpointError> {
    push_u32(out, len_u32(value.len())?);
    out.extend_from_slice(value);
    Ok(())
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

    fn take(&mut self, len: usize) -> Result<&'a [u8], CheckpointError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(CheckpointError::Overflow)?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or(CheckpointError::Truncated)?;
        self.offset = end;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8, CheckpointError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, CheckpointError> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> Result<u32, CheckpointError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn u64(&mut self) -> Result<u64, CheckpointError> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn bytes(&mut self) -> Result<&'a [u8], CheckpointError> {
        let len = usize::try_from(self.u32()?).map_err(|_| CheckpointError::Overflow)?;
        self.take(len)
    }

    fn string(&mut self) -> Result<String, CheckpointError> {
        String::from_utf8(self.bytes()?.to_vec()).map_err(|_| CheckpointError::InvalidUtf8)
    }

    fn strings(&mut self) -> Result<Vec<String>, CheckpointError> {
        let count = usize::try_from(self.u32()?).map_err(|_| CheckpointError::Overflow)?;
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(self.string()?);
        }
        Ok(values)
    }

    fn boolean(&mut self) -> Result<bool, CheckpointError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            other => Err(CheckpointError::InvalidBoolean(other)),
        }
    }

    fn optional_string(&mut self) -> Result<Option<String>, CheckpointError> {
        match self.u8()? {
            0 => Ok(None),
            1 => Ok(Some(self.string()?)),
            other => Err(CheckpointError::InvalidBoolean(other)),
        }
    }

    fn is_finished(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CognitiveRuntime, MemoryKind};
    use ntd_core::{ActionNode, CapabilityId, SideEffectClass};

    fn sample_state() -> CognitiveState {
        let mut runtime = CognitiveRuntime::new(CognitiveIdentity(*b"NTD97-COGNITION1"));
        let goal = runtime
            .state_mut()
            .add_goal("preserve identity", vec!["restore".into()])
            .expect("goal");

        let mut intent = Intent::new("checkpoint state");
        intent.completion_criteria.push("restored".into());
        let graph = TaskGraph {
            actions: vec![ActionNode {
                id: 7,
                capability: CapabilityId("internal.verify".into()),
                side_effect: SideEffectClass::ReadOnly,
                verification_required: true,
            }],
        };
        runtime
            .state_mut()
            .submit_task(intent, graph, Some(goal))
            .expect("task");
        runtime
            .state_mut()
            .set_world_fact("mode", "offline")
            .expect("world");
        runtime
            .state_mut()
            .memory
            .store(
                MemoryKind::Semantic,
                "NTD97 owns its memory",
                vec!["identity".into()],
                900,
                1,
            )
            .expect("memory");
        runtime
            .state_mut()
            .install_delta("reasoning.local", 2, vec![1, 2, 3])
            .expect("delta");
        runtime.state().clone()
    }

    #[test]
    fn checkpoint_round_trips_deterministically() {
        let state = sample_state();
        let first = encode_cognitive_checkpoint(&state).expect("encode");
        let second = encode_cognitive_checkpoint(&state).expect("encode");
        assert_eq!(first, second);

        let decoded = decode_cognitive_checkpoint(&first).expect("decode");
        assert_eq!(decoded, state);
    }

    #[test]
    fn unsupported_major_is_rejected() {
        let state = sample_state();
        let mut bytes = encode_cognitive_checkpoint(&state).expect("encode");
        bytes[8] = 1;
        assert!(matches!(
            decode_cognitive_checkpoint(&bytes),
            Err(CheckpointError::UnsupportedVersion { major: 1, .. })
        ));
    }
}
