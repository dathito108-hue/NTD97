#![forbid(unsafe_code)]

use crate::{
    ApprovalRequest, ContinuityError, ContinuityPhase, PlatformContinuityState, RetryBackoff,
    WakeReason,
};
use ntd_runtime::{ActionId, ActionPlanId};

pub const CNT97_MAGIC: [u8; 6] = *b"CNT97\0";
pub const CNT97_MAJOR: u16 = 0;
pub const CNT97_MINOR: u16 = 1;
pub const CNT97_HEADER_LEN: usize = 24;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContinuityBundle {
    pub capsule_root: [u8; 32],
    pub platform: PlatformContinuityState,
    pub cognitive_checkpoint: Vec<u8>,
    pub action_checkpoint: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BundleError {
    Truncated,
    InvalidMagic,
    InvalidHeader,
    UnsupportedVersion { major: u16, minor: u16 },
    InvalidUtf8,
    InvalidBoolean(u8),
    InvalidPhase(u8),
    InvalidWakeReason(u8),
    NonCanonicalEncoding,
    InvalidState,
    Overflow,
}

pub fn encode_continuity_bundle(
    bundle: &ContinuityBundle,
) -> Result<Vec<u8>, BundleError> {
    bundle
        .platform
        .validate()
        .map_err(|_| BundleError::InvalidState)?;
    if bundle.cognitive_checkpoint.is_empty() || bundle.action_checkpoint.is_empty() {
        return Err(BundleError::InvalidState);
    }

    let mut payload = Vec::new();
    payload.extend_from_slice(&bundle.capsule_root);
    encode_platform_state(&mut payload, &bundle.platform)?;
    push_bytes(&mut payload, &bundle.cognitive_checkpoint)?;
    push_bytes(&mut payload, &bundle.action_checkpoint)?;

    let payload_len = u64::try_from(payload.len()).map_err(|_| BundleError::Overflow)?;
    let mut out = Vec::with_capacity(
        CNT97_HEADER_LEN
            .checked_add(payload.len())
            .ok_or(BundleError::Overflow)?,
    );
    out.extend_from_slice(&CNT97_MAGIC);
    push_u16(&mut out, CNT97_HEADER_LEN as u16);
    push_u16(&mut out, CNT97_MAJOR);
    push_u16(&mut out, CNT97_MINOR);
    push_u32(&mut out, 0);
    push_u64(&mut out, payload_len);
    out.extend_from_slice(&payload);
    Ok(out)
}

pub fn decode_continuity_bundle(bytes: &[u8]) -> Result<ContinuityBundle, BundleError> {
    if bytes.len() < CNT97_HEADER_LEN {
        return Err(BundleError::Truncated);
    }

    let mut header = Cursor::new(bytes);
    if header.take(6)? != CNT97_MAGIC.as_slice() {
        return Err(BundleError::InvalidMagic);
    }
    if usize::from(header.u16()?) != CNT97_HEADER_LEN {
        return Err(BundleError::InvalidHeader);
    }

    let major = header.u16()?;
    let minor = header.u16()?;
    if major != CNT97_MAJOR || minor > CNT97_MINOR {
        return Err(BundleError::UnsupportedVersion { major, minor });
    }
    if header.u32()? != 0 {
        return Err(BundleError::InvalidHeader);
    }

    let payload_len = usize::try_from(header.u64()?).map_err(|_| BundleError::Overflow)?;
    let expected = CNT97_HEADER_LEN
        .checked_add(payload_len)
        .ok_or(BundleError::Overflow)?;
    if bytes.len() != expected {
        return Err(BundleError::NonCanonicalEncoding);
    }

    let mut cursor = Cursor::new(&bytes[CNT97_HEADER_LEN..]);
    let mut capsule_root = [0u8; 32];
    capsule_root.copy_from_slice(cursor.take(32)?);

    let platform = decode_platform_state(&mut cursor)?;
    platform
        .validate()
        .map_err(|_| BundleError::InvalidState)?;

    let cognitive_checkpoint = cursor.bytes()?.to_vec();
    let action_checkpoint = cursor.bytes()?.to_vec();
    if cognitive_checkpoint.is_empty() || action_checkpoint.is_empty() || !cursor.is_finished() {
        return Err(BundleError::NonCanonicalEncoding);
    }

    Ok(ContinuityBundle {
        capsule_root,
        platform,
        cognitive_checkpoint,
        action_checkpoint,
    })
}

fn encode_platform_state(
    out: &mut Vec<u8>,
    state: &PlatformContinuityState,
) -> Result<(), BundleError> {
    push_u8(out, state.phase as u8);
    push_u64(out, state.logical_tick);
    push_u64(out, state.checkpoint_generation);
    push_u64(out, state.boot_count);
    push_u64(out, state.next_approval_id);

    match state.last_wake_reason {
        Some(reason) => {
            push_u8(out, 1);
            push_u8(out, reason as u8);
        }
        None => push_u8(out, 0),
    }

    push_u32(out, state.retry.attempts);
    push_u64(out, state.retry.next_eligible_tick);
    push_u64(out, state.retry.base_delay_ticks);
    push_u64(out, state.retry.max_delay_ticks);

    match &state.pending_approval {
        Some(approval) => {
            push_u8(out, 1);
            push_u64(out, approval.id);
            push_u64(out, approval.plan_id.0);
            push_u64(out, approval.action_id.0);
            push_bool(out, approval.external_write);
            push_bool(out, approval.irreversible);
            push_u32(out, len_u32(approval.required_scopes.len())?);
            for scope in &approval.required_scopes {
                push_string(out, scope)?;
            }
        }
        None => push_u8(out, 0),
    }

    Ok(())
}

fn decode_platform_state(
    cursor: &mut Cursor<'_>,
) -> Result<PlatformContinuityState, BundleError> {
    let phase = decode_phase(cursor.u8()?)?;
    let logical_tick = cursor.u64()?;
    let checkpoint_generation = cursor.u64()?;
    let boot_count = cursor.u64()?;
    let next_approval_id = cursor.u64()?;

    let last_wake_reason = match cursor.u8()? {
        0 => None,
        1 => Some(decode_wake_reason(cursor.u8()?)?),
        other => return Err(BundleError::InvalidBoolean(other)),
    };

    let retry = RetryBackoff {
        attempts: cursor.u32()?,
        next_eligible_tick: cursor.u64()?,
        base_delay_ticks: cursor.u64()?,
        max_delay_ticks: cursor.u64()?,
    };

    let pending_approval = match cursor.u8()? {
        0 => None,
        1 => {
            let id = cursor.u64()?;
            let plan_id = ActionPlanId(cursor.u64()?);
            let action_id = ActionId(cursor.u64()?);
            let external_write = cursor.boolean()?;
            let irreversible = cursor.boolean()?;
            let count = usize::try_from(cursor.u32()?).map_err(|_| BundleError::Overflow)?;
            let mut required_scopes = Vec::with_capacity(count);
            for _ in 0..count {
                required_scopes.push(cursor.string()?);
            }
            if !required_scopes.windows(2).all(|pair| pair[0] < pair[1]) {
                return Err(BundleError::NonCanonicalEncoding);
            }
            Some(ApprovalRequest {
                id,
                plan_id,
                action_id,
                required_scopes,
                external_write,
                irreversible,
            })
        }
        other => return Err(BundleError::InvalidBoolean(other)),
    };

    Ok(PlatformContinuityState {
        phase,
        logical_tick,
        checkpoint_generation,
        boot_count,
        next_approval_id,
        pending_approval,
        retry,
        last_wake_reason,
    })
}

fn decode_phase(value: u8) -> Result<ContinuityPhase, BundleError> {
    match value {
        1 => Ok(ContinuityPhase::Interactive),
        2 => Ok(ContinuityPhase::ActiveExecution),
        3 => Ok(ContinuityPhase::Checkpointed),
        4 => Ok(ContinuityPhase::SuspendedByOs),
        5 => Ok(ContinuityPhase::WaitingCondition),
        6 => Ok(ContinuityPhase::WaitingApproval),
        7 => Ok(ContinuityPhase::Reconstructing),
        8 => Ok(ContinuityPhase::VerifyingResume),
        9 => Ok(ContinuityPhase::Completed),
        10 => Ok(ContinuityPhase::FailedRecoverable),
        other => Err(BundleError::InvalidPhase(other)),
    }
}

fn decode_wake_reason(value: u8) -> Result<WakeReason, BundleError> {
    match value {
        1 => Ok(WakeReason::UserLaunch),
        2 => Ok(WakeReason::ForegroundService),
        3 => Ok(WakeReason::Scheduled),
        4 => Ok(WakeReason::Retry),
        5 => Ok(WakeReason::Boot),
        6 => Ok(WakeReason::Connectivity),
        7 => Ok(WakeReason::Approval),
        8 => Ok(WakeReason::Notification),
        other => Err(BundleError::InvalidWakeReason(other)),
    }
}

fn len_u32(value: usize) -> Result<u32, BundleError> {
    u32::try_from(value).map_err(|_| BundleError::Overflow)
}

fn push_string(out: &mut Vec<u8>, value: &str) -> Result<(), BundleError> {
    push_bytes(out, value.as_bytes())
}

fn push_bytes(out: &mut Vec<u8>, value: &[u8]) -> Result<(), BundleError> {
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

    fn take(&mut self, len: usize) -> Result<&'a [u8], BundleError> {
        let end = self.offset.checked_add(len).ok_or(BundleError::Overflow)?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or(BundleError::Truncated)?;
        self.offset = end;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8, BundleError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, BundleError> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> Result<u32, BundleError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn u64(&mut self) -> Result<u64, BundleError> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn bytes(&mut self) -> Result<&'a [u8], BundleError> {
        let len = usize::try_from(self.u32()?).map_err(|_| BundleError::Overflow)?;
        self.take(len)
    }

    fn string(&mut self) -> Result<String, BundleError> {
        String::from_utf8(self.bytes()?.to_vec()).map_err(|_| BundleError::InvalidUtf8)
    }

    fn boolean(&mut self) -> Result<bool, BundleError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            other => Err(BundleError::InvalidBoolean(other)),
        }
    }

    fn is_finished(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundle_rejects_empty_runtime_checkpoints() {
        let bundle = ContinuityBundle {
            capsule_root: [7; 32],
            platform: PlatformContinuityState::default(),
            cognitive_checkpoint: Vec::new(),
            action_checkpoint: vec![1],
        };
        assert_eq!(
            encode_continuity_bundle(&bundle),
            Err(BundleError::InvalidState)
        );
    }
}
