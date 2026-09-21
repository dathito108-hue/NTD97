#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use ntd_core::{CapabilityId, SideEffectClass};

use crate::{
    action_fabric::validate_fabric_state, ActionFabricError, ActionFabricState, ActionId,
    ActionOutput, ActionPlanId, ActionPlanState, ActionPlanStatus, ActionStatus, ActionValue,
    CapabilityRegistry, PlannedAction, TypedAction,
};

pub const TAF97_MAGIC: [u8; 6] = *b"TAF97\0";
pub const TAF97_MAJOR: u16 = 0;
pub const TAF97_MINOR: u16 = 1;
pub const TAF97_HEADER_LEN: usize = 24;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionCheckpointError {
    Truncated,
    InvalidMagic,
    InvalidHeader,
    UnsupportedVersion { major: u16, minor: u16 },
    InvalidUtf8,
    InvalidBoolean(u8),
    InvalidActionTag(u8),
    InvalidValueTag(u8),
    InvalidSideEffect(u8),
    NonCanonicalEncoding,
    Overflow,
    Fabric(ActionFabricError),
}

impl From<ActionFabricError> for ActionCheckpointError {
    fn from(value: ActionFabricError) -> Self {
        Self::Fabric(value)
    }
}

pub fn encode_action_fabric_checkpoint(
    registry: &CapabilityRegistry,
    state: &ActionFabricState,
) -> Result<Vec<u8>, ActionCheckpointError> {
    validate_fabric_state(registry, state)?;

    let mut payload = Vec::new();
    push_u64(&mut payload, state.next_plan_id);
    push_u64(&mut payload, state.next_action_id);
    push_u32(&mut payload, len_u32(state.plans.len())?);

    for plan in state.plans.values() {
        encode_plan(&mut payload, plan)?;
    }

    let payload_len = u64::try_from(payload.len()).map_err(|_| ActionCheckpointError::Overflow)?;
    let mut out = Vec::with_capacity(
        TAF97_HEADER_LEN
            .checked_add(payload.len())
            .ok_or(ActionCheckpointError::Overflow)?,
    );
    out.extend_from_slice(&TAF97_MAGIC);
    push_u16(&mut out, TAF97_HEADER_LEN as u16);
    push_u16(&mut out, TAF97_MAJOR);
    push_u16(&mut out, TAF97_MINOR);
    push_u32(&mut out, 0);
    push_u64(&mut out, payload_len);
    out.extend_from_slice(&payload);
    Ok(out)
}

pub fn decode_action_fabric_checkpoint(
    registry: &CapabilityRegistry,
    bytes: &[u8],
) -> Result<ActionFabricState, ActionCheckpointError> {
    if bytes.len() < TAF97_HEADER_LEN {
        return Err(ActionCheckpointError::Truncated);
    }

    let mut header = Cursor::new(bytes);
    if header.take(6)? != TAF97_MAGIC.as_slice() {
        return Err(ActionCheckpointError::InvalidMagic);
    }
    if usize::from(header.u16()?) != TAF97_HEADER_LEN {
        return Err(ActionCheckpointError::InvalidHeader);
    }

    let major = header.u16()?;
    let minor = header.u16()?;
    if major != TAF97_MAJOR || minor > TAF97_MINOR {
        return Err(ActionCheckpointError::UnsupportedVersion { major, minor });
    }
    if header.u32()? != 0 {
        return Err(ActionCheckpointError::InvalidHeader);
    }

    let payload_len = usize::try_from(header.u64()?).map_err(|_| ActionCheckpointError::Overflow)?;
    let expected = TAF97_HEADER_LEN
        .checked_add(payload_len)
        .ok_or(ActionCheckpointError::Overflow)?;
    if bytes.len() != expected {
        return Err(ActionCheckpointError::NonCanonicalEncoding);
    }

    let mut cursor = Cursor::new(&bytes[TAF97_HEADER_LEN..]);
    let next_plan_id = cursor.u64()?;
    let next_action_id = cursor.u64()?;
    let count = cursor.u32()?;

    let mut plans = BTreeMap::new();
    let mut previous = 0u64;
    for _ in 0..count {
        let plan = decode_plan(&mut cursor)?;
        if plan.id.0 == 0 || plan.id.0 <= previous {
            return Err(ActionCheckpointError::NonCanonicalEncoding);
        }
        previous = plan.id.0;
        if plans.insert(plan.id.0, plan).is_some() {
            return Err(ActionCheckpointError::NonCanonicalEncoding);
        }
    }

    if !cursor.is_finished() {
        return Err(ActionCheckpointError::NonCanonicalEncoding);
    }

    let state = ActionFabricState {
        next_plan_id,
        next_action_id,
        plans,
    };
    validate_fabric_state(registry, &state)?;
    Ok(state)
}

fn encode_plan(out: &mut Vec<u8>, plan: &ActionPlanState) -> Result<(), ActionCheckpointError> {
    push_u64(out, plan.id.0);
    push_u64(out, plan.task_id);
    push_u64(
        out,
        u64::try_from(plan.cursor).map_err(|_| ActionCheckpointError::Overflow)?,
    );
    push_u8(out, plan.status as u8);
    push_u32(out, len_u32(plan.actions.len())?);
    for action in &plan.actions {
        encode_planned_action(out, action)?;
    }
    Ok(())
}

fn decode_plan(cursor: &mut Cursor<'_>) -> Result<ActionPlanState, ActionCheckpointError> {
    let id = ActionPlanId(cursor.u64()?);
    let task_id = cursor.u64()?;
    let plan_cursor =
        usize::try_from(cursor.u64()?).map_err(|_| ActionCheckpointError::Overflow)?;
    let status_raw = cursor.u8()?;
    let status =
        ActionPlanStatus::try_from(status_raw).map_err(ActionCheckpointError::Fabric)?;
    let action_count =
        usize::try_from(cursor.u32()?).map_err(|_| ActionCheckpointError::Overflow)?;
    let mut actions = Vec::with_capacity(action_count);
    for _ in 0..action_count {
        actions.push(decode_planned_action(cursor)?);
    }

    Ok(ActionPlanState {
        id,
        task_id,
        cursor: plan_cursor,
        status,
        actions,
    })
}

fn encode_planned_action(
    out: &mut Vec<u8>,
    action: &PlannedAction,
) -> Result<(), ActionCheckpointError> {
    push_u64(out, action.id.0);
    push_u32(out, action.node_id);
    push_string(out, &action.capability.0)?;
    push_u32(out, action.capability_version);
    push_u8(out, encode_side_effect(action.side_effect));
    push_bool(out, action.verification_required);
    encode_typed_action(out, &action.action)?;
    push_u8(out, action.status as u8);
    push_u32(out, action.attempts);
    push_optional_output(out, action.output.as_ref())?;
    push_optional_bytes(out, action.resume_token.as_deref())?;
    push_optional_bytes(out, action.rollback_token.as_deref())?;
    push_optional_string(out, action.last_error.as_deref())?;
    Ok(())
}

fn decode_planned_action(cursor: &mut Cursor<'_>) -> Result<PlannedAction, ActionCheckpointError> {
    let id = ActionId(cursor.u64()?);
    let node_id = cursor.u32()?;
    let capability = CapabilityId(cursor.string()?);
    let capability_version = cursor.u32()?;
    let side_effect = decode_side_effect(cursor.u8()?)?;
    let verification_required = cursor.boolean()?;
    let action = decode_typed_action(cursor)?;
    let status_raw = cursor.u8()?;
    let status = ActionStatus::try_from(status_raw).map_err(ActionCheckpointError::Fabric)?;
    let attempts = cursor.u32()?;
    let output = cursor.optional_output()?;
    let resume_token = cursor.optional_bytes()?.map(ToOwned::to_owned);
    let rollback_token = cursor.optional_bytes()?.map(ToOwned::to_owned);
    let last_error = cursor.optional_string()?;

    Ok(PlannedAction {
        id,
        node_id,
        capability,
        capability_version,
        side_effect,
        verification_required,
        action,
        status,
        attempts,
        output,
        resume_token,
        rollback_token,
        last_error,
    })
}

fn encode_typed_action(
    out: &mut Vec<u8>,
    action: &TypedAction,
) -> Result<(), ActionCheckpointError> {
    match action {
        TypedAction::WebSearch { query, max_results } => {
            push_u8(out, 1);
            push_string(out, query)?;
            push_u16(out, *max_results);
        }
        TypedAction::WebFetch { url } => {
            push_u8(out, 2);
            push_string(out, url)?;
        }
        TypedAction::BrowserObserve { target } => {
            push_u8(out, 3);
            push_string(out, target)?;
        }
        TypedAction::BrowserInteract {
            target,
            operation,
            value,
        } => {
            push_u8(out, 4);
            push_string(out, target)?;
            push_string(out, operation)?;
            push_optional_string(out, value.as_deref())?;
        }
        TypedAction::FileRead { path } => {
            push_u8(out, 5);
            push_string(out, path)?;
        }
        TypedAction::FileWrite { path, bytes } => {
            push_u8(out, 6);
            push_string(out, path)?;
            push_bytes(out, bytes)?;
        }
        TypedAction::DeviceObserve { surface } => {
            push_u8(out, 7);
            push_string(out, surface)?;
        }
        TypedAction::DeviceInteract {
            surface,
            operation,
            argument,
        } => {
            push_u8(out, 8);
            push_string(out, surface)?;
            push_string(out, operation)?;
            push_optional_string(out, argument.as_deref())?;
        }
        TypedAction::AppAction {
            app,
            action,
            payload,
        } => {
            push_u8(out, 9);
            push_string(out, app)?;
            push_string(out, action)?;
            push_bytes(out, payload)?;
        }
        TypedAction::Custom { type_name, payload } => {
            push_u8(out, 10);
            push_string(out, type_name)?;
            push_bytes(out, payload)?;
        }
    }
    Ok(())
}

fn decode_typed_action(cursor: &mut Cursor<'_>) -> Result<TypedAction, ActionCheckpointError> {
    match cursor.u8()? {
        1 => Ok(TypedAction::WebSearch {
            query: cursor.string()?,
            max_results: cursor.u16()?,
        }),
        2 => Ok(TypedAction::WebFetch {
            url: cursor.string()?,
        }),
        3 => Ok(TypedAction::BrowserObserve {
            target: cursor.string()?,
        }),
        4 => Ok(TypedAction::BrowserInteract {
            target: cursor.string()?,
            operation: cursor.string()?,
            value: cursor.optional_string()?,
        }),
        5 => Ok(TypedAction::FileRead {
            path: cursor.string()?,
        }),
        6 => Ok(TypedAction::FileWrite {
            path: cursor.string()?,
            bytes: cursor.bytes()?.to_vec(),
        }),
        7 => Ok(TypedAction::DeviceObserve {
            surface: cursor.string()?,
        }),
        8 => Ok(TypedAction::DeviceInteract {
            surface: cursor.string()?,
            operation: cursor.string()?,
            argument: cursor.optional_string()?,
        }),
        9 => Ok(TypedAction::AppAction {
            app: cursor.string()?,
            action: cursor.string()?,
            payload: cursor.bytes()?.to_vec(),
        }),
        10 => Ok(TypedAction::Custom {
            type_name: cursor.string()?,
            payload: cursor.bytes()?.to_vec(),
        }),
        other => Err(ActionCheckpointError::InvalidActionTag(other)),
    }
}

fn push_optional_output(
    out: &mut Vec<u8>,
    value: Option<&ActionOutput>,
) -> Result<(), ActionCheckpointError> {
    match value {
        None => {
            push_u8(out, 0);
            Ok(())
        }
        Some(output) => {
            push_u8(out, 1);
            push_string(out, &output.summary)?;
            encode_action_value(out, &output.value)?;
            push_strings(out, &output.evidence)
        }
    }
}

fn encode_action_value(
    out: &mut Vec<u8>,
    value: &ActionValue,
) -> Result<(), ActionCheckpointError> {
    match value {
        ActionValue::None => push_u8(out, 0),
        ActionValue::Text(value) => {
            push_u8(out, 1);
            push_string(out, value)?;
        }
        ActionValue::Bytes(value) => {
            push_u8(out, 2);
            push_bytes(out, value)?;
        }
        ActionValue::TextList(values) => {
            push_u8(out, 3);
            push_strings(out, values)?;
        }
        ActionValue::Fields(fields) => {
            push_u8(out, 4);
            push_u32(out, len_u32(fields.len())?);
            for (key, value) in fields {
                push_string(out, key)?;
                push_string(out, value)?;
            }
        }
    }
    Ok(())
}

fn decode_action_value(cursor: &mut Cursor<'_>) -> Result<ActionValue, ActionCheckpointError> {
    match cursor.u8()? {
        0 => Ok(ActionValue::None),
        1 => Ok(ActionValue::Text(cursor.string()?)),
        2 => Ok(ActionValue::Bytes(cursor.bytes()?.to_vec())),
        3 => Ok(ActionValue::TextList(cursor.strings()?)),
        4 => {
            let count =
                usize::try_from(cursor.u32()?).map_err(|_| ActionCheckpointError::Overflow)?;
            let mut fields = BTreeMap::new();
            let mut previous: Option<String> = None;
            for _ in 0..count {
                let key = cursor.string()?;
                if previous.as_ref().is_some_and(|value| key <= *value) {
                    return Err(ActionCheckpointError::NonCanonicalEncoding);
                }
                previous = Some(key.clone());
                let value = cursor.string()?;
                if fields.insert(key, value).is_some() {
                    return Err(ActionCheckpointError::NonCanonicalEncoding);
                }
            }
            Ok(ActionValue::Fields(fields))
        }
        other => Err(ActionCheckpointError::InvalidValueTag(other)),
    }
}

fn encode_side_effect(value: SideEffectClass) -> u8 {
    match value {
        SideEffectClass::ReadOnly => 1,
        SideEffectClass::Reversible => 2,
        SideEffectClass::ExternalWrite => 3,
        SideEffectClass::Irreversible => 4,
    }
}

fn decode_side_effect(value: u8) -> Result<SideEffectClass, ActionCheckpointError> {
    match value {
        1 => Ok(SideEffectClass::ReadOnly),
        2 => Ok(SideEffectClass::Reversible),
        3 => Ok(SideEffectClass::ExternalWrite),
        4 => Ok(SideEffectClass::Irreversible),
        other => Err(ActionCheckpointError::InvalidSideEffect(other)),
    }
}

fn len_u32(value: usize) -> Result<u32, ActionCheckpointError> {
    u32::try_from(value).map_err(|_| ActionCheckpointError::Overflow)
}

fn push_strings(out: &mut Vec<u8>, values: &[String]) -> Result<(), ActionCheckpointError> {
    push_u32(out, len_u32(values.len())?);
    for value in values {
        push_string(out, value)?;
    }
    Ok(())
}

fn push_optional_string(
    out: &mut Vec<u8>,
    value: Option<&str>,
) -> Result<(), ActionCheckpointError> {
    match value {
        None => {
            push_u8(out, 0);
            Ok(())
        }
        Some(value) => {
            push_u8(out, 1);
            push_string(out, value)
        }
    }
}

fn push_optional_bytes(
    out: &mut Vec<u8>,
    value: Option<&[u8]>,
) -> Result<(), ActionCheckpointError> {
    match value {
        None => {
            push_u8(out, 0);
            Ok(())
        }
        Some(value) => {
            push_u8(out, 1);
            push_bytes(out, value)
        }
    }
}

fn push_string(out: &mut Vec<u8>, value: &str) -> Result<(), ActionCheckpointError> {
    push_bytes(out, value.as_bytes())
}

fn push_bytes(out: &mut Vec<u8>, value: &[u8]) -> Result<(), ActionCheckpointError> {
    push_u32(out, len_u32(value.len())?);
    out.extend_from_slice(value);
    Ok(())
}

fn push_bool(out: &mut Vec<u8>, value: bool) {
    push_u8(out, u8::from(value));
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

    fn take(&mut self, len: usize) -> Result<&'a [u8], ActionCheckpointError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(ActionCheckpointError::Overflow)?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or(ActionCheckpointError::Truncated)?;
        self.offset = end;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8, ActionCheckpointError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, ActionCheckpointError> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> Result<u32, ActionCheckpointError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn u64(&mut self) -> Result<u64, ActionCheckpointError> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn bytes(&mut self) -> Result<&'a [u8], ActionCheckpointError> {
        let len = usize::try_from(self.u32()?).map_err(|_| ActionCheckpointError::Overflow)?;
        self.take(len)
    }

    fn string(&mut self) -> Result<String, ActionCheckpointError> {
        String::from_utf8(self.bytes()?.to_vec()).map_err(|_| ActionCheckpointError::InvalidUtf8)
    }

    fn strings(&mut self) -> Result<Vec<String>, ActionCheckpointError> {
        let count = usize::try_from(self.u32()?).map_err(|_| ActionCheckpointError::Overflow)?;
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(self.string()?);
        }
        Ok(values)
    }

    fn boolean(&mut self) -> Result<bool, ActionCheckpointError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            other => Err(ActionCheckpointError::InvalidBoolean(other)),
        }
    }

    fn optional_string(&mut self) -> Result<Option<String>, ActionCheckpointError> {
        match self.u8()? {
            0 => Ok(None),
            1 => Ok(Some(self.string()?)),
            other => Err(ActionCheckpointError::InvalidBoolean(other)),
        }
    }

    fn optional_bytes(&mut self) -> Result<Option<&'a [u8]>, ActionCheckpointError> {
        match self.u8()? {
            0 => Ok(None),
            1 => Ok(Some(self.bytes()?)),
            other => Err(ActionCheckpointError::InvalidBoolean(other)),
        }
    }

    fn optional_output(&mut self) -> Result<Option<ActionOutput>, ActionCheckpointError> {
        match self.u8()? {
            0 => Ok(None),
            1 => Ok(Some(ActionOutput {
                summary: self.string()?,
                value: decode_action_value(self)?,
                evidence: self.strings()?,
            })),
            other => Err(ActionCheckpointError::InvalidBoolean(other)),
        }
    }

    fn is_finished(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AuthorityScope, CapabilityDescriptor, CapabilityDomain, CapabilityRegistry,
    };
    use ntd_core::SideEffectClass;

    fn registry() -> CapabilityRegistry {
        let mut registry = CapabilityRegistry::new();
        let mut descriptor = CapabilityDescriptor::new(
            CapabilityId("web.search".into()),
            1,
            CapabilityDomain::Web,
            SideEffectClass::ReadOnly,
        )
        .expect("descriptor");
        descriptor.required_scopes =
            vec![AuthorityScope::new("network.read").expect("scope")];
        registry.register(descriptor).expect("register");
        registry
    }

    #[test]
    fn checkpoint_round_trips_action_state() {
        let registry = registry();
        let state = ActionFabricState {
            next_plan_id: 2,
            next_action_id: 2,
            plans: BTreeMap::from([(
                1,
                ActionPlanState {
                    id: ActionPlanId(1),
                    task_id: 9,
                    cursor: 0,
                    status: ActionPlanStatus::Suspended,
                    actions: vec![PlannedAction {
                        id: ActionId(1),
                        node_id: 4,
                        capability: CapabilityId("web.search".into()),
                        capability_version: 1,
                        side_effect: SideEffectClass::ReadOnly,
                        verification_required: true,
                        action: TypedAction::WebSearch {
                            query: "NTD97".into(),
                            max_results: 3,
                        },
                        status: ActionStatus::Suspended,
                        attempts: 1,
                        output: None,
                        resume_token: Some(vec![7, 9]),
                        rollback_token: None,
                        last_error: Some("network paused".into()),
                    }],
                },
            )]),
        };

        let first = encode_action_fabric_checkpoint(&registry, &state).expect("encode");
        let second = encode_action_fabric_checkpoint(&registry, &state).expect("encode");
        assert_eq!(first, second);
        assert_eq!(
            decode_action_fabric_checkpoint(&registry, &first).expect("decode"),
            state
        );
    }
}
