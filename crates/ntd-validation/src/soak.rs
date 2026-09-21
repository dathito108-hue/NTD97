#![forbid(unsafe_code)]

use ntd_mobile_shell::{
    build_mobile_continuity_bundle, decode_mobile_continuity_bundle,
    encode_mobile_continuity_bundle, restore_mobile_continuity_bundle, MobileContinuityState,
    WakeReason,
};
use ntd_runtime::{ActionFabric, CapabilityRegistry, CognitiveIdentity, CognitiveRuntime};

use crate::ValidationError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContinuitySoakReport {
    pub logical_minutes: u64,
    pub checkpoint_cycles: u64,
    pub final_sequence: u64,
}

pub fn run_logical_continuity_soak(
    hours: u64,
    checkpoint_interval_minutes: u64,
) -> Result<ContinuitySoakReport, ValidationError> {
    if hours == 0 || checkpoint_interval_minutes == 0 {
        return Err(ValidationError::InvalidConfiguration);
    }
    let logical_minutes = hours
        .checked_mul(60)
        .ok_or(ValidationError::InvalidConfiguration)?;
    let checkpoint_cycles = logical_minutes
        .checked_div(checkpoint_interval_minutes)
        .ok_or(ValidationError::InvalidConfiguration)?;
    if checkpoint_cycles == 0 {
        return Err(ValidationError::InvalidConfiguration);
    }

    let registry = CapabilityRegistry::new();
    let identity = CognitiveIdentity(*b"NTD97-COGNITION1");
    let mut cognitive = CognitiveRuntime::new(identity);
    let mut actions = ActionFabric::new(registry.clone());

    for sequence in 1..=checkpoint_cycles {
        let bundle = build_mobile_continuity_bundle(
            &cognitive,
            &registry,
            actions.state(),
            MobileContinuityState::Checkpointed,
            WakeReason::ScheduledWork,
            sequence,
            None,
            None,
        )
        .map_err(|error| ValidationError::Continuity(format!("{error:?}")))?;
        let encoded = encode_mobile_continuity_bundle(&bundle)
            .map_err(|error| ValidationError::Continuity(format!("{error:?}")))?;
        let decoded = decode_mobile_continuity_bundle(&encoded)
            .map_err(|error| ValidationError::Continuity(format!("{error:?}")))?;
        let restored = restore_mobile_continuity_bundle(decoded)
            .map_err(|error| ValidationError::Continuity(format!("{error:?}")))?;
        if restored.cognitive.state().identity != identity
            || restored.checkpoint_sequence != sequence
        {
            return Err(ValidationError::Continuity(
                "identity or sequence drift".into(),
            ));
        }
        cognitive = restored.cognitive;
        actions = restored.actions;
    }

    Ok(ContinuitySoakReport {
        logical_minutes,
        checkpoint_cycles,
        final_sequence: checkpoint_cycles,
    })
}
