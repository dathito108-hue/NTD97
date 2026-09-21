#![forbid(unsafe_code)]

use ntd_runtime::{ResourceSnapshot, ThermalState};

use crate::{MobileContinuityBundle, MobileContinuityState};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WakeContext {
    pub now_epoch_ms: u64,
    pub user_visible: bool,
    pub network_available: bool,
    pub device_unlocked: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlatformDirective {
    KeepInteractive,
    StartForegroundExecution {
        reason: String,
    },
    SchedulePersistentWork {
        delay_ms: u64,
        require_network: bool,
        persisted: bool,
    },
    ShowApproval {
        capability: String,
        rationale: String,
    },
    CheckpointAndSuspend,
    VerifyResume,
    NoEligibleWork,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WakePolicyError {
    MissingApproval,
    InvalidRetryWindow,
}

pub fn choose_platform_directive(
    bundle: &MobileContinuityBundle,
    resources: ResourceSnapshot,
    context: WakeContext,
) -> Result<PlatformDirective, WakePolicyError> {
    if bundle.state == MobileContinuityState::WaitingApproval {
        let approval = bundle
            .pending_approval
            .as_ref()
            .ok_or(WakePolicyError::MissingApproval)?;
        return Ok(PlatformDirective::ShowApproval {
            capability: approval.capability.clone(),
            rationale: approval.rationale.clone(),
        });
    }

    if matches!(
        bundle.state,
        MobileContinuityState::Completed | MobileContinuityState::FailedRecoverable
    ) {
        return Ok(PlatformDirective::NoEligibleWork);
    }

    if matches!(
        bundle.state,
        MobileContinuityState::Reconstructing | MobileContinuityState::VerifyingResume
    ) {
        return Ok(PlatformDirective::VerifyResume);
    }

    if let Some(retry) = &bundle.retry {
        if retry.not_before_epoch_ms > context.now_epoch_ms {
            return Ok(PlatformDirective::SchedulePersistentWork {
                delay_ms: retry.not_before_epoch_ms - context.now_epoch_ms,
                require_network: false,
                persisted: true,
            });
        }
        if retry.base_delay_ms == 0 || retry.max_delay_ms < retry.base_delay_ms {
            return Err(WakePolicyError::InvalidRetryWindow);
        }
    }

    if resources.thermal == ThermalState::Critical
        || (!resources.charging && resources.battery_percent <= 5)
    {
        return Ok(PlatformDirective::CheckpointAndSuspend);
    }

    match bundle.state {
        MobileContinuityState::Interactive if context.user_visible => {
            Ok(PlatformDirective::KeepInteractive)
        }
        MobileContinuityState::ActiveExecution if context.user_visible => {
            Ok(PlatformDirective::KeepInteractive)
        }
        MobileContinuityState::ActiveExecution => Ok(PlatformDirective::StartForegroundExecution {
            reason: "visible active task".into(),
        }),
        MobileContinuityState::Checkpointed
        | MobileContinuityState::SuspendedByOs
        | MobileContinuityState::WaitingCondition => {
            let require_network = !context.network_available;
            Ok(PlatformDirective::SchedulePersistentWork {
                delay_ms: if context.device_unlocked {
                    1_000
                } else {
                    15_000
                },
                require_network,
                persisted: true,
            })
        }
        MobileContinuityState::Interactive => Ok(PlatformDirective::SchedulePersistentWork {
            delay_ms: 1_000,
            require_network: false,
            persisted: false,
        }),
        MobileContinuityState::WaitingApproval
        | MobileContinuityState::Reconstructing
        | MobileContinuityState::VerifyingResume
        | MobileContinuityState::Completed
        | MobileContinuityState::FailedRecoverable => unreachable!("handled above"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MobileContinuityBundle, PendingApproval, RetryBackoff, WakeReason};
    use ntd_runtime::CognitiveIdentity;

    fn snapshot(battery: u8, thermal: ThermalState) -> ResourceSnapshot {
        ResourceSnapshot {
            available_ram_bytes: 2 * 1024 * 1024 * 1024,
            battery_percent: battery,
            charging: false,
            thermal,
            latency_budget_ms: 250,
        }
    }

    fn bundle(state: MobileContinuityState) -> MobileContinuityBundle {
        MobileContinuityBundle {
            identity: CognitiveIdentity(*b"NTD97-COGNITION1"),
            state,
            wake_reason: WakeReason::ScheduledWork,
            checkpoint_sequence: 1,
            cognitive_checkpoint: vec![1],
            action_checkpoint: vec![1],
            capabilities: Vec::new(),
            pending_approval: None,
            retry: None,
        }
    }

    #[test]
    fn active_background_work_uses_foreground_execution() {
        let decision = choose_platform_directive(
            &bundle(MobileContinuityState::ActiveExecution),
            snapshot(80, ThermalState::Nominal),
            WakeContext {
                now_epoch_ms: 0,
                user_visible: false,
                network_available: true,
                device_unlocked: true,
            },
        )
        .expect("policy");

        assert!(matches!(
            decision,
            PlatformDirective::StartForegroundExecution { .. }
        ));
    }

    #[test]
    fn critical_thermal_state_checkpoints_instead_of_running() {
        let decision = choose_platform_directive(
            &bundle(MobileContinuityState::ActiveExecution),
            snapshot(80, ThermalState::Critical),
            WakeContext {
                now_epoch_ms: 0,
                user_visible: false,
                network_available: true,
                device_unlocked: true,
            },
        )
        .expect("policy");

        assert_eq!(decision, PlatformDirective::CheckpointAndSuspend);
    }

    #[test]
    fn approval_state_routes_to_explicit_surface() {
        let mut bundle = bundle(MobileContinuityState::WaitingApproval);
        bundle.pending_approval = Some(PendingApproval {
            task_id: 1,
            plan_id: 1,
            action_id: 1,
            capability: "file.write".into(),
            rationale: "write requested".into(),
        });

        let decision = choose_platform_directive(
            &bundle,
            snapshot(80, ThermalState::Nominal),
            WakeContext {
                now_epoch_ms: 0,
                user_visible: false,
                network_available: true,
                device_unlocked: true,
            },
        )
        .expect("policy");

        assert_eq!(
            decision,
            PlatformDirective::ShowApproval {
                capability: "file.write".into(),
                rationale: "write requested".into()
            }
        );
    }

    #[test]
    fn retry_window_is_preserved_across_wake() {
        let mut bundle = bundle(MobileContinuityState::Checkpointed);
        bundle.retry = Some(RetryBackoff {
            attempts: 2,
            not_before_epoch_ms: 10_000,
            base_delay_ms: 1_000,
            max_delay_ms: 60_000,
        });

        let decision = choose_platform_directive(
            &bundle,
            snapshot(80, ThermalState::Nominal),
            WakeContext {
                now_epoch_ms: 4_000,
                user_visible: false,
                network_available: true,
                device_unlocked: true,
            },
        )
        .expect("policy");

        assert_eq!(
            decision,
            PlatformDirective::SchedulePersistentWork {
                delay_ms: 6_000,
                require_network: false,
                persisted: true
            }
        );
    }
}
