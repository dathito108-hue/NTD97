#![forbid(unsafe_code)]

use ntd_runtime::{
    decode_action_fabric_checkpoint, decode_cognitive_checkpoint, encode_action_fabric_checkpoint,
    encode_cognitive_checkpoint, ActionFabric, ActionFabricState, AuthorityScope,
    CapabilityDescriptor, CapabilityDomain, CapabilityRegistry, CognitiveIdentity, CognitiveRuntime,
};

pub const MCS97_MAGIC: [u8; 6] = *b"MCS97\0";
pub const MCS97_MAJOR: u16 = 0;
pub const MCS97_MINOR: u16 = 1;
pub const MCS97_HEADER_LEN: usize = 24;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MobileContinuityState {
    Interactive = 1,
    ActiveExecution = 2,
    Checkpointed = 3,
    SuspendedByOs = 4,
    WaitingCondition = 5,
    WaitingApproval = 6,
    Reconstructing = 7,
    VerifyingResume = 8,
    Completed = 9,
    FailedRecoverable = 10,
}

impl TryFrom<u8> for MobileContinuityState {
    type Error = MobileContinuityError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Interactive),
            2 => Ok(Self::ActiveExecution),
            3 => Ok(Self::Checkpointed),
            4 => Ok(Self::SuspendedByOs),
            5 => Ok(Self::WaitingCondition),
            6 => Ok(Self::WaitingApproval),
            7 => Ok(Self::Reconstructing),
            8 => Ok(Self::VerifyingResume),
            9 => Ok(Self::Completed),
            10 => Ok(Self::FailedRecoverable),
            other => Err(MobileContinuityError::InvalidContinuityState(other)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WakeReason {
    UserInteraction = 1,
    ForegroundExecution = 2,
    ScheduledWork = 3,
    RetryBackoff = 4,
    Reboot = 5,
    ApprovalResolved = 6,
    NetworkAvailable = 7,
    Charging = 8,
    ManualRecovery = 9,
}

impl TryFrom<u8> for WakeReason {
    type Error = MobileContinuityError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::UserInteraction),
            2 => Ok(Self::ForegroundExecution),
            3 => Ok(Self::ScheduledWork),
            4 => Ok(Self::RetryBackoff),
            5 => Ok(Self::Reboot),
            6 => Ok(Self::ApprovalResolved),
            7 => Ok(Self::NetworkAvailable),
            8 => Ok(Self::Charging),
            9 => Ok(Self::ManualRecovery),
            other => Err(MobileContinuityError::InvalidWakeReason(other)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingApproval {
    pub task_id: u64,
    pub plan_id: u64,
    pub action_id: u64,
    pub capability: String,
    pub rationale: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetryBackoff {
    pub attempts: u32,
    pub not_before_epoch_ms: u64,
    pub base_delay_ms: u64,
    pub max_delay_ms: u64,
}

impl RetryBackoff {
    pub fn next_delay_ms(&self) -> u64 {
        let exponent = self.attempts.min(20);
        self.base_delay_ms
            .saturating_mul(1u64 << exponent)
            .min(self.max_delay_ms)
    }

    fn validate(&self) -> Result<(), MobileContinuityError> {
        if self.base_delay_ms == 0 || self.max_delay_ms < self.base_delay_ms {
            return Err(MobileContinuityError::InvalidRetryBackoff);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MobileContinuityBundle {
    pub identity: CognitiveIdentity,
    pub state: MobileContinuityState,
    pub wake_reason: WakeReason,
    pub checkpoint_sequence: u64,
    pub cognitive_checkpoint: Vec<u8>,
    pub action_checkpoint: Vec<u8>,
    pub capabilities: Vec<CapabilityDescriptor>,
    pub pending_approval: Option<PendingApproval>,
    pub retry: Option<RetryBackoff>,
}

pub struct RestoredMobileSession {
    pub cognitive: CognitiveRuntime,
    pub actions: ActionFabric,
    pub state: MobileContinuityState,
    pub wake_reason: WakeReason,
    pub checkpoint_sequence: u64,
    pub pending_approval: Option<PendingApproval>,
    pub retry: Option<RetryBackoff>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MobileContinuityError {
    Truncated,
    InvalidMagic,
    InvalidHeader,
    UnsupportedVersion { major: u16, minor: u16 },
    InvalidUtf8,
    InvalidBoolean(u8),
    InvalidContinuityState(u8),
    InvalidWakeReason(u8),
    NonCanonicalEncoding,
    InvalidIdentity,
    InvalidRetryBackoff,
    InvalidApproval,
    TaskActionMismatch { task_id: u64 },
    Overflow,
    CognitiveCheckpoint(String),
    ActionCheckpoint(String),
}

pub fn build_mobile_continuity_bundle(
    cognitive: &CognitiveRuntime,
    registry: &CapabilityRegistry,
    actions: &ActionFabricState,
    state: MobileContinuityState,
    wake_reason: WakeReason,
    checkpoint_sequence: u64,
    pending_approval: Option<PendingApproval>,
    retry: Option<RetryBackoff>,
) -> Result<MobileContinuityBundle, MobileContinuityError> {
    if checkpoint_sequence == 0 {
        return Err(MobileContinuityError::NonCanonicalEncoding);
    }
    if let Some(retry) = &retry {
        retry.validate()?;
    }

    let cognitive_checkpoint = encode_cognitive_checkpoint(cognitive.state())
        .map_err(|error| MobileContinuityError::CognitiveCheckpoint(format!("{error:?}")))?;
    let action_checkpoint = encode_action_fabric_checkpoint(registry, actions)
        .map_err(|error| MobileContinuityError::ActionCheckpoint(format!("{error:?}")))?;
    let capabilities = registry.descriptors().cloned().collect::<Vec<_>>();

    validate_links(cognitive, actions, pending_approval.as_ref())?;

    let bundle = MobileContinuityBundle {
        identity: cognitive.state().identity,
        state,
        wake_reason,
        checkpoint_sequence,
        cognitive_checkpoint,
        action_checkpoint,
        capabilities,
        pending_approval,
        retry,
    };
    validate_bundle(&bundle)?;
    Ok(bundle)
}

pub fn restore_mobile_continuity_bundle(
    bundle: MobileContinuityBundle,
) -> Result<RestoredMobileSession, MobileContinuityError> {
    validate_bundle(&bundle)?;
    let registry = registry_from_snapshot(&bundle.capabilities)?;

    let cognitive_state = decode_cognitive_checkpoint(&bundle.cognitive_checkpoint)
        .map_err(|error| MobileContinuityError::CognitiveCheckpoint(format!("{error:?}")))?;
    if cognitive_state.identity != bundle.identity {
        return Err(MobileContinuityError::InvalidIdentity);
    }
    let cognitive = CognitiveRuntime::from_state(cognitive_state)
        .map_err(|error| MobileContinuityError::CognitiveCheckpoint(format!("{error:?}")))?;

    let action_state = decode_action_fabric_checkpoint(&registry, &bundle.action_checkpoint)
        .map_err(|error| MobileContinuityError::ActionCheckpoint(format!("{error:?}")))?;
    validate_links(&cognitive, &action_state, bundle.pending_approval.as_ref())?;
    let actions = ActionFabric::from_state(registry, action_state)
        .map_err(|error| MobileContinuityError::ActionCheckpoint(format!("{error:?}")))?;

    Ok(RestoredMobileSession {
        cognitive,
        actions,
        state: bundle.state,
        wake_reason: bundle.wake_reason,
        checkpoint_sequence: bundle.checkpoint_sequence,
        pending_approval: bundle.pending_approval,
        retry: bundle.retry,
    })
}

pub fn encode_mobile_continuity_bundle(
    bundle: &MobileContinuityBundle,
) -> Result<Vec<u8>, MobileContinuityError> {
    validate_bundle(bundle)?;

    let mut payload = Vec::new();
    payload.extend_from_slice(&bundle.identity.0);
    push_u8(&mut payload, bundle.state as u8);
    push_u8(&mut payload, bundle.wake_reason as u8);
    push_u16(&mut payload, 0);
    push_u64(&mut payload, bundle.checkpoint_sequence);
    push_bytes(&mut payload, &bundle.cognitive_checkpoint)?;
    push_bytes(&mut payload, &bundle.action_checkpoint)?;
    push_capabilities(&mut payload, &bundle.capabilities)?;
    push_optional_approval(&mut payload, bundle.pending_approval.as_ref())?;
    push_optional_retry(&mut payload, bundle.retry.as_ref());

    let payload_len = u64::try_from(payload.len()).map_err(|_| MobileContinuityError::Overflow)?;
    let mut out = Vec::with_capacity(
        MCS97_HEADER_LEN
            .checked_add(payload.len())
            .ok_or(MobileContinuityError::Overflow)?,
    );
    out.extend_from_slice(&MCS97_MAGIC);
    push_u16(&mut out, MCS97_HEADER_LEN as u16);
    push_u16(&mut out, MCS97_MAJOR);
    push_u16(&mut out, MCS97_MINOR);
    push_u32(&mut out, 0);
    push_u64(&mut out, payload_len);
    out.extend_from_slice(&payload);
    Ok(out)
}

pub fn decode_mobile_continuity_bundle(
    bytes: &[u8],
) -> Result<MobileContinuityBundle, MobileContinuityError> {
    if bytes.len() < MCS97_HEADER_LEN {
        return Err(MobileContinuityError::Truncated);
    }
    let mut header = Cursor::new(bytes);
    if header.take(6)? != MCS97_MAGIC.as_slice() {
        return Err(MobileContinuityError::InvalidMagic);
    }
    if usize::from(header.u16()?) != MCS97_HEADER_LEN {
        return Err(MobileContinuityError::InvalidHeader);
    }
    let major = header.u16()?;
    let minor = header.u16()?;
    if major != MCS97_MAJOR || minor > MCS97_MINOR {
        return Err(MobileContinuityError::UnsupportedVersion { major, minor });
    }
    if header.u32()? != 0 {
        return Err(MobileContinuityError::InvalidHeader);
    }
    let payload_len =
        usize::try_from(header.u64()?).map_err(|_| MobileContinuityError::Overflow)?;
    let expected = MCS97_HEADER_LEN
        .checked_add(payload_len)
        .ok_or(MobileContinuityError::Overflow)?;
    if bytes.len() != expected {
        return Err(MobileContinuityError::NonCanonicalEncoding);
    }

    let mut cursor = Cursor::new(&bytes[MCS97_HEADER_LEN..]);
    let mut identity = [0u8; 16];
    identity.copy_from_slice(cursor.take(16)?);
    let state = MobileContinuityState::try_from(cursor.u8()?)?;
    let wake_reason = WakeReason::try_from(cursor.u8()?)?;
    if cursor.u16()? != 0 {
        return Err(MobileContinuityError::NonCanonicalEncoding);
    }
    let checkpoint_sequence = cursor.u64()?;
    let cognitive_checkpoint = cursor.bytes()?.to_vec();
    let action_checkpoint = cursor.bytes()?.to_vec();
    let capabilities = cursor.capabilities()?;
    let pending_approval = cursor.optional_approval()?;
    let retry = cursor.optional_retry()?;

    if !cursor.is_finished() {
        return Err(MobileContinuityError::NonCanonicalEncoding);
    }

    let bundle = MobileContinuityBundle {
        identity: CognitiveIdentity(identity),
        state,
        wake_reason,
        checkpoint_sequence,
        cognitive_checkpoint,
        action_checkpoint,
        capabilities,
        pending_approval,
        retry,
    };
    validate_bundle(&bundle)?;
    Ok(bundle)
}


fn registry_from_snapshot(
    descriptors: &[CapabilityDescriptor],
) -> Result<CapabilityRegistry, MobileContinuityError> {
    let mut registry = CapabilityRegistry::new();
    for descriptor in descriptors.iter().cloned() {
        registry
            .register(descriptor)
            .map_err(|error| MobileContinuityError::ActionCheckpoint(format!("{error:?}")))?;
    }
    Ok(registry)
}

fn push_capabilities(
    out: &mut Vec<u8>,
    descriptors: &[CapabilityDescriptor],
) -> Result<(), MobileContinuityError> {
    let count = u32::try_from(descriptors.len()).map_err(|_| MobileContinuityError::Overflow)?;
    push_u32(out, count);
    for descriptor in descriptors {
        push_string(out, &descriptor.id.0)?;
        push_u32(out, descriptor.version);
        push_u8(out, descriptor.domain as u8);
        push_u8(out, encode_side_effect(descriptor.side_effect));
        push_u8(out, u8::from(descriptor.verification_required));
        push_u8(out, u8::from(descriptor.rollback_supported));
        push_u8(out, u8::from(descriptor.resumable));
        let scope_count =
            u32::try_from(descriptor.required_scopes.len()).map_err(|_| MobileContinuityError::Overflow)?;
        push_u32(out, scope_count);
        for scope in &descriptor.required_scopes {
            push_string(out, scope.as_str())?;
        }
    }
    Ok(())
}

fn encode_side_effect(value: ntd_core::SideEffectClass) -> u8 {
    match value {
        ntd_core::SideEffectClass::ReadOnly => 1,
        ntd_core::SideEffectClass::Reversible => 2,
        ntd_core::SideEffectClass::ExternalWrite => 3,
        ntd_core::SideEffectClass::Irreversible => 4,
    }
}

fn decode_side_effect(value: u8) -> Result<ntd_core::SideEffectClass, MobileContinuityError> {
    match value {
        1 => Ok(ntd_core::SideEffectClass::ReadOnly),
        2 => Ok(ntd_core::SideEffectClass::Reversible),
        3 => Ok(ntd_core::SideEffectClass::ExternalWrite),
        4 => Ok(ntd_core::SideEffectClass::Irreversible),
        _ => Err(MobileContinuityError::NonCanonicalEncoding),
    }
}

fn validate_bundle(bundle: &MobileContinuityBundle) -> Result<(), MobileContinuityError> {
    if bundle.checkpoint_sequence == 0
        || bundle.cognitive_checkpoint.is_empty()
        || bundle.action_checkpoint.is_empty()
    {
        return Err(MobileContinuityError::NonCanonicalEncoding);
    }
    if !bundle
        .capabilities
        .windows(2)
        .all(|pair| pair[0].id.0 < pair[1].id.0)
    {
        return Err(MobileContinuityError::NonCanonicalEncoding);
    }
    let registry = registry_from_snapshot(&bundle.capabilities)?;
    decode_action_fabric_checkpoint(&registry, &bundle.action_checkpoint)
        .map_err(|error| MobileContinuityError::ActionCheckpoint(format!("{error:?}")))?;
    if let Some(retry) = &bundle.retry {
        retry.validate()?;
    }
    if let Some(approval) = &bundle.pending_approval {
        validate_approval(approval)?;
        if bundle.state != MobileContinuityState::WaitingApproval {
            return Err(MobileContinuityError::InvalidApproval);
        }
    }
    if bundle.state == MobileContinuityState::WaitingApproval && bundle.pending_approval.is_none() {
        return Err(MobileContinuityError::InvalidApproval);
    }
    Ok(())
}

fn validate_links(
    cognitive: &CognitiveRuntime,
    actions: &ActionFabricState,
    approval: Option<&PendingApproval>,
) -> Result<(), MobileContinuityError> {
    for plan in actions.plans.values() {
        if !cognitive.state().tasks.contains_key(&plan.task_id) {
            return Err(MobileContinuityError::TaskActionMismatch {
                task_id: plan.task_id,
            });
        }
    }

    if let Some(approval) = approval {
        validate_approval(approval)?;
        let plan = actions
            .plans
            .get(&approval.plan_id)
            .ok_or(MobileContinuityError::InvalidApproval)?;
        if plan.task_id != approval.task_id
            || !plan.actions.iter().any(|action| {
                action.id.0 == approval.action_id && action.capability.0 == approval.capability
            })
        {
            return Err(MobileContinuityError::InvalidApproval);
        }
    }

    Ok(())
}

fn validate_approval(approval: &PendingApproval) -> Result<(), MobileContinuityError> {
    if approval.task_id == 0
        || approval.plan_id == 0
        || approval.action_id == 0
        || approval.capability.trim().is_empty()
        || approval.rationale.trim().is_empty()
    {
        return Err(MobileContinuityError::InvalidApproval);
    }
    Ok(())
}

fn push_optional_approval(
    out: &mut Vec<u8>,
    approval: Option<&PendingApproval>,
) -> Result<(), MobileContinuityError> {
    match approval {
        None => push_u8(out, 0),
        Some(approval) => {
            push_u8(out, 1);
            push_u64(out, approval.task_id);
            push_u64(out, approval.plan_id);
            push_u64(out, approval.action_id);
            push_string(out, &approval.capability)?;
            push_string(out, &approval.rationale)?;
        }
    }
    Ok(())
}

fn push_optional_retry(out: &mut Vec<u8>, retry: Option<&RetryBackoff>) {
    match retry {
        None => push_u8(out, 0),
        Some(retry) => {
            push_u8(out, 1);
            push_u32(out, retry.attempts);
            push_u64(out, retry.not_before_epoch_ms);
            push_u64(out, retry.base_delay_ms);
            push_u64(out, retry.max_delay_ms);
        }
    }
}

fn push_string(out: &mut Vec<u8>, value: &str) -> Result<(), MobileContinuityError> {
    push_bytes(out, value.as_bytes())
}

fn push_bytes(out: &mut Vec<u8>, value: &[u8]) -> Result<(), MobileContinuityError> {
    let len = u32::try_from(value.len()).map_err(|_| MobileContinuityError::Overflow)?;
    push_u32(out, len);
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

    fn take(&mut self, len: usize) -> Result<&'a [u8], MobileContinuityError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(MobileContinuityError::Overflow)?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or(MobileContinuityError::Truncated)?;
        self.offset = end;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8, MobileContinuityError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, MobileContinuityError> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> Result<u32, MobileContinuityError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn u64(&mut self) -> Result<u64, MobileContinuityError> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn bytes(&mut self) -> Result<&'a [u8], MobileContinuityError> {
        let len = usize::try_from(self.u32()?).map_err(|_| MobileContinuityError::Overflow)?;
        self.take(len)
    }

    fn string(&mut self) -> Result<String, MobileContinuityError> {
        String::from_utf8(self.bytes()?.to_vec()).map_err(|_| MobileContinuityError::InvalidUtf8)
    }


    fn capabilities(&mut self) -> Result<Vec<CapabilityDescriptor>, MobileContinuityError> {
        let count = usize::try_from(self.u32()?).map_err(|_| MobileContinuityError::Overflow)?;
        let mut descriptors = Vec::with_capacity(count);
        let mut previous: Option<String> = None;

        for _ in 0..count {
            let id = self.string()?;
            if id.trim().is_empty()
                || previous.as_ref().is_some_and(|previous_id| id <= *previous_id)
            {
                return Err(MobileContinuityError::NonCanonicalEncoding);
            }
            previous = Some(id.clone());

            let version = self.u32()?;
            let domain = CapabilityDomain::try_from(self.u8()?)
                .map_err(|error| MobileContinuityError::ActionCheckpoint(format!("{error:?}")))?;
            let side_effect = decode_side_effect(self.u8()?)?;
            let verification_required = self.boolean()?;
            let rollback_supported = self.boolean()?;
            let resumable = self.boolean()?;
            let scope_count =
                usize::try_from(self.u32()?).map_err(|_| MobileContinuityError::Overflow)?;
            let mut required_scopes = Vec::with_capacity(scope_count);
            for _ in 0..scope_count {
                required_scopes.push(
                    AuthorityScope::new(self.string()?)
                        .map_err(|error| MobileContinuityError::ActionCheckpoint(format!("{error:?}")))?,
                );
            }

            let mut descriptor = CapabilityDescriptor::new(
                ntd_core::CapabilityId(id),
                version,
                domain,
                side_effect,
            )
            .map_err(|error| MobileContinuityError::ActionCheckpoint(format!("{error:?}")))?;
            descriptor.verification_required = verification_required;
            descriptor.rollback_supported = rollback_supported;
            descriptor.resumable = resumable;
            descriptor.required_scopes = required_scopes;
            descriptor
                .normalize()
                .map_err(|error| MobileContinuityError::ActionCheckpoint(format!("{error:?}")))?;
            descriptors.push(descriptor);
        }

        Ok(descriptors)
    }

    fn boolean(&mut self) -> Result<bool, MobileContinuityError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            other => Err(MobileContinuityError::InvalidBoolean(other)),
        }
    }

    fn optional_approval(&mut self) -> Result<Option<PendingApproval>, MobileContinuityError> {
        match self.u8()? {
            0 => Ok(None),
            1 => Ok(Some(PendingApproval {
                task_id: self.u64()?,
                plan_id: self.u64()?,
                action_id: self.u64()?,
                capability: self.string()?,
                rationale: self.string()?,
            })),
            other => Err(MobileContinuityError::InvalidBoolean(other)),
        }
    }

    fn optional_retry(&mut self) -> Result<Option<RetryBackoff>, MobileContinuityError> {
        match self.u8()? {
            0 => Ok(None),
            1 => Ok(Some(RetryBackoff {
                attempts: self.u32()?,
                not_before_epoch_ms: self.u64()?,
                base_delay_ms: self.u64()?,
                max_delay_ms: self.u64()?,
            })),
            other => Err(MobileContinuityError::InvalidBoolean(other)),
        }
    }

    fn is_finished(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use ntd_core::{ActionNode, CapabilityId, Intent, SideEffectClass, TaskGraph};
    use ntd_runtime::{
        ActionFabric, AuthorityScope, CapabilityDescriptor, CapabilityDomain, CapabilityRegistry,
        CognitiveIdentity, TypedAction,
    };

    use super::*;

    fn registry() -> CapabilityRegistry {
        let mut registry = CapabilityRegistry::new();
        let mut descriptor = CapabilityDescriptor::new(
            CapabilityId("web.search".into()),
            1,
            CapabilityDomain::Web,
            SideEffectClass::ReadOnly,
        )
        .expect("descriptor");
        descriptor.resumable = true;
        descriptor.required_scopes =
            vec![AuthorityScope::new("network.read").expect("scope")];
        registry.register(descriptor).expect("register");
        registry
    }

    #[test]
    fn bundle_round_trips_and_restores_same_task_identity() {
        let registry = registry();
        let mut cognitive = CognitiveRuntime::new(CognitiveIdentity(*b"NTD97-COGNITION1"));
        let task_id = cognitive
            .state_mut()
            .submit_task(
                Intent::new("resume web task"),
                TaskGraph {
                    actions: vec![ActionNode {
                        id: 1,
                        capability: CapabilityId("web.search".into()),
                        side_effect: SideEffectClass::ReadOnly,
                        verification_required: true,
                    }],
                },
                None,
            )
            .expect("task");

        let task = cognitive.state().tasks.get(&task_id).expect("task").clone();
        let mut fabric = ActionFabric::new(registry.clone());
        let plan_id = fabric
            .prepare_cognitive_task(
                &task,
                BTreeMap::from([(
                    1,
                    TypedAction::WebSearch {
                        query: "NTD97".into(),
                        max_results: 2,
                    },
                )]),
            )
            .expect("plan");

        let bundle = build_mobile_continuity_bundle(
            &cognitive,
            &registry,
            fabric.state(),
            MobileContinuityState::Checkpointed,
            WakeReason::ScheduledWork,
            1,
            None,
            Some(RetryBackoff {
                attempts: 2,
                not_before_epoch_ms: 99,
                base_delay_ms: 1_000,
                max_delay_ms: 60_000,
            }),
        )
        .expect("bundle");

        let first = encode_mobile_continuity_bundle(&bundle).expect("encode");
        let second = encode_mobile_continuity_bundle(&bundle).expect("encode");
        assert_eq!(first, second);

        let decoded = decode_mobile_continuity_bundle(&first).expect("decode");
        let restored =
            restore_mobile_continuity_bundle(decoded).expect("restore mobile session");

        assert_eq!(restored.cognitive.state().identity, bundle.identity);
        assert!(restored.cognitive.state().tasks.contains_key(&task_id));
        assert_eq!(
            restored.actions.state().plans.get(&plan_id.0).expect("plan").task_id,
            task_id
        );
        assert_eq!(restored.checkpoint_sequence, 1);
    }

    #[test]
    fn capability_version_tampering_breaks_restore() {
        let registry = registry();
        let mut cognitive = CognitiveRuntime::new(CognitiveIdentity(*b"NTD97-COGNITION1"));
        let task_id = cognitive
            .state_mut()
            .submit_task(
                Intent::new("version-bound task"),
                TaskGraph {
                    actions: vec![ActionNode {
                        id: 1,
                        capability: CapabilityId("web.search".into()),
                        side_effect: SideEffectClass::ReadOnly,
                        verification_required: true,
                    }],
                },
                None,
            )
            .expect("task");
        let task = cognitive.state().tasks.get(&task_id).expect("task").clone();
        let mut fabric = ActionFabric::new(registry.clone());
        fabric
            .prepare_cognitive_task(
                &task,
                BTreeMap::from([(
                    1,
                    TypedAction::WebSearch {
                        query: "version".into(),
                        max_results: 1,
                    },
                )]),
            )
            .expect("plan");

        let mut bundle = build_mobile_continuity_bundle(
            &cognitive,
            &registry,
            fabric.state(),
            MobileContinuityState::Checkpointed,
            WakeReason::Reboot,
            1,
            None,
            None,
        )
        .expect("bundle");

        bundle.capabilities[0].version = 2;
        assert!(matches!(
            restore_mobile_continuity_bundle(bundle),
            Err(MobileContinuityError::ActionCheckpoint(_))
        ));
    }

    #[test]
    fn approval_must_match_action_plan() {
        let registry = registry();
        let mut cognitive = CognitiveRuntime::new(CognitiveIdentity(*b"NTD97-COGNITION1"));
        let task_id = cognitive
            .state_mut()
            .submit_task(Intent::new("noop"), TaskGraph::default(), None)
            .expect("task");
        let fabric = ActionFabric::new(registry.clone());

        let result = build_mobile_continuity_bundle(
            &cognitive,
            &registry,
            fabric.state(),
            MobileContinuityState::WaitingApproval,
            WakeReason::UserInteraction,
            1,
            Some(PendingApproval {
                task_id,
                plan_id: 1,
                action_id: 1,
                capability: "web.search".into(),
                rationale: "needs approval".into(),
            }),
            None,
        );

        assert_eq!(result, Err(MobileContinuityError::InvalidApproval));
    }
}
