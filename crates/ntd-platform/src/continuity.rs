#![forbid(unsafe_code)]

use ntd_core::SideEffectClass;
use ntd_runtime::{
    decode_action_fabric_checkpoint, decode_cognitive_checkpoint, encode_action_fabric_checkpoint,
    encode_cognitive_checkpoint, ActionCheckpointError, ActionFabric, ActionFabricError, ActionId,
    ActionPlanId, ActionPlanStatus, ActionStatus, ActionStepReport, ActionVerifier, AuthorityGrant,
    CapabilityError, CapabilityRegistry, CheckpointError, CognitiveRuntime,
};

use crate::{decode_continuity_bundle, encode_continuity_bundle, BundleError, ContinuityBundle};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ContinuityPhase {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WakeReason {
    UserLaunch = 1,
    ForegroundService = 2,
    Scheduled = 3,
    Retry = 4,
    Boot = 5,
    Connectivity = 6,
    Approval = 7,
    Notification = 8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlatformConstraints {
    pub user_visible: bool,
    pub foreground_allowed: bool,
    pub background_execution_allowed: bool,
    pub network_available: bool,
    pub charging: bool,
    pub battery_percent: u8,
    pub thermal_critical: bool,
}

impl PlatformConstraints {
    pub fn normalized(self) -> Self {
        Self {
            battery_percent: self.battery_percent.min(100),
            ..self
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WakeDisposition {
    Interactive,
    Foreground,
    BackgroundOnce,
    AwaitApproval,
    Defer { until_tick: u64 },
    Idle,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetryBackoff {
    pub attempts: u32,
    pub next_eligible_tick: u64,
    pub base_delay_ticks: u64,
    pub max_delay_ticks: u64,
}

impl Default for RetryBackoff {
    fn default() -> Self {
        Self {
            attempts: 0,
            next_eligible_tick: 0,
            base_delay_ticks: 1,
            max_delay_ticks: 1024,
        }
    }
}

impl RetryBackoff {
    pub fn reset(&mut self) {
        self.attempts = 0;
        self.next_eligible_tick = 0;
    }

    pub fn record_failure(&mut self, now_tick: u64) {
        self.attempts = self.attempts.saturating_add(1);
        let shift = self.attempts.saturating_sub(1).min(20);
        let multiplier = 1u64.checked_shl(shift).unwrap_or(u64::MAX);
        let delay = self
            .base_delay_ticks
            .saturating_mul(multiplier)
            .min(self.max_delay_ticks.max(self.base_delay_ticks));
        self.next_eligible_tick = now_tick.saturating_add(delay.max(1));
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApprovalRequest {
    pub id: u64,
    pub plan_id: ActionPlanId,
    pub action_id: ActionId,
    pub required_scopes: Vec<String>,
    pub external_write: bool,
    pub irreversible: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlatformContinuityState {
    pub phase: ContinuityPhase,
    pub logical_tick: u64,
    pub checkpoint_generation: u64,
    pub boot_count: u64,
    pub next_approval_id: u64,
    pub pending_approval: Option<ApprovalRequest>,
    pub retry: RetryBackoff,
    pub last_wake_reason: Option<WakeReason>,
}

impl Default for PlatformContinuityState {
    fn default() -> Self {
        Self {
            phase: ContinuityPhase::Interactive,
            logical_tick: 0,
            checkpoint_generation: 0,
            boot_count: 0,
            next_approval_id: 1,
            pending_approval: None,
            retry: RetryBackoff::default(),
            last_wake_reason: None,
        }
    }
}

impl PlatformContinuityState {
    pub(crate) fn validate(&self) -> Result<(), ContinuityError> {
        if self.next_approval_id == 0
            || self.retry.base_delay_ticks == 0
            || self.retry.max_delay_ticks < self.retry.base_delay_ticks
        {
            return Err(ContinuityError::InvalidState);
        }

        if let Some(approval) = &self.pending_approval {
            if approval.id == 0
                || self.next_approval_id <= approval.id
                || approval.plan_id.0 == 0
                || approval.action_id.0 == 0
                || !approval
                    .required_scopes
                    .windows(2)
                    .all(|pair| pair[0] < pair[1])
            {
                return Err(ContinuityError::InvalidState);
            }
        }

        Ok(())
    }
}

pub trait CheckpointStore {
    fn commit(&mut self, bytes: &[u8]) -> Result<(), String>;
    fn load(&mut self) -> Result<Option<Vec<u8>>, String>;
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InMemoryCheckpointStore {
    bytes: Option<Vec<u8>>,
    pub commit_count: u64,
}

impl InMemoryCheckpointStore {
    pub fn bytes(&self) -> Option<&[u8]> {
        self.bytes.as_deref()
    }
}

impl CheckpointStore for InMemoryCheckpointStore {
    fn commit(&mut self, bytes: &[u8]) -> Result<(), String> {
        self.bytes = Some(bytes.to_vec());
        self.commit_count = self.commit_count.saturating_add(1);
        Ok(())
    }

    fn load(&mut self) -> Result<Option<Vec<u8>>, String> {
        Ok(self.bytes.clone())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContinuityError {
    Store(String),
    Bundle(BundleError),
    CognitiveCheckpoint(CheckpointError),
    ActionCheckpoint(ActionCheckpointError),
    Action(ActionFabricError),
    MissingCheckpoint,
    MissingPlan(ActionPlanId),
    InvalidState,
}

impl From<BundleError> for ContinuityError {
    fn from(value: BundleError) -> Self {
        Self::Bundle(value)
    }
}

impl From<CheckpointError> for ContinuityError {
    fn from(value: CheckpointError) -> Self {
        Self::CognitiveCheckpoint(value)
    }
}

impl From<ActionCheckpointError> for ContinuityError {
    fn from(value: ActionCheckpointError) -> Self {
        Self::ActionCheckpoint(value)
    }
}

impl From<ActionFabricError> for ContinuityError {
    fn from(value: ActionFabricError) -> Self {
        Self::Action(value)
    }
}

pub struct ContinuitySupervisor {
    capsule_root: [u8; 32],
    cognitive: CognitiveRuntime,
    actions: ActionFabric,
    platform: PlatformContinuityState,
}

impl ContinuitySupervisor {
    pub fn new(
        capsule_root: [u8; 32],
        cognitive: CognitiveRuntime,
        actions: ActionFabric,
    ) -> Self {
        Self {
            capsule_root,
            cognitive,
            actions,
            platform: PlatformContinuityState::default(),
        }
    }

    pub fn restore<S: CheckpointStore>(
        registry: CapabilityRegistry,
        store: &mut S,
    ) -> Result<Self, ContinuityError> {
        let bytes = store
            .load()
            .map_err(ContinuityError::Store)?
            .ok_or(ContinuityError::MissingCheckpoint)?;
        let bundle = decode_continuity_bundle(&bytes)?;
        bundle.platform.validate()?;

        let cognitive_state = decode_cognitive_checkpoint(&bundle.cognitive_checkpoint)?;
        let action_state =
            decode_action_fabric_checkpoint(&registry, &bundle.action_checkpoint)?;

        let mut platform = bundle.platform;
        platform.phase = ContinuityPhase::VerifyingResume;

        Ok(Self {
            capsule_root: bundle.capsule_root,
            cognitive: CognitiveRuntime::from_state(cognitive_state)
                .map_err(|_| ContinuityError::InvalidState)?,
            actions: ActionFabric::from_state(registry, action_state)?,
            platform,
        })
    }

    pub fn capsule_root(&self) -> [u8; 32] {
        self.capsule_root
    }

    pub fn cognitive(&self) -> &CognitiveRuntime {
        &self.cognitive
    }

    pub fn cognitive_mut(&mut self) -> &mut CognitiveRuntime {
        &mut self.cognitive
    }

    pub fn actions(&self) -> &ActionFabric {
        &self.actions
    }

    pub fn actions_mut(&mut self) -> &mut ActionFabric {
        &mut self.actions
    }

    pub fn platform(&self) -> &PlatformContinuityState {
        &self.platform
    }

    pub fn decide_wake(
        &self,
        constraints: PlatformConstraints,
    ) -> WakeDisposition {
        let constraints = constraints.normalized();

        if matches!(
            self.platform.phase,
            ContinuityPhase::Completed
        ) {
            return WakeDisposition::Idle;
        }

        if self.platform.pending_approval.is_some() {
            return WakeDisposition::AwaitApproval;
        }

        if constraints.thermal_critical
            || (constraints.battery_percent <= 5 && !constraints.charging)
        {
            return WakeDisposition::Defer {
                until_tick: self
                    .platform
                    .retry
                    .next_eligible_tick
                    .max(self.platform.logical_tick.saturating_add(1)),
            };
        }

        if self.platform.logical_tick < self.platform.retry.next_eligible_tick {
            return WakeDisposition::Defer {
                until_tick: self.platform.retry.next_eligible_tick,
            };
        }

        if constraints.user_visible {
            WakeDisposition::Interactive
        } else if constraints.background_execution_allowed {
            WakeDisposition::BackgroundOnce
        } else if constraints.foreground_allowed {
            WakeDisposition::Foreground
        } else {
            WakeDisposition::Defer {
                until_tick: self.platform.logical_tick.saturating_add(
                    self.platform.retry.base_delay_ticks.max(1),
                ),
            }
        }
    }

    pub fn on_wake(
        &mut self,
        reason: WakeReason,
        constraints: PlatformConstraints,
    ) -> WakeDisposition {
        self.platform.logical_tick = self.platform.logical_tick.saturating_add(1);
        self.platform.last_wake_reason = Some(reason);
        if reason == WakeReason::Boot {
            self.platform.boot_count = self.platform.boot_count.saturating_add(1);
        }

        let disposition = self.decide_wake(constraints);
        self.platform.phase = match disposition {
            WakeDisposition::Interactive => ContinuityPhase::Interactive,
            WakeDisposition::Foreground | WakeDisposition::BackgroundOnce => {
                ContinuityPhase::VerifyingResume
            }
            WakeDisposition::AwaitApproval => ContinuityPhase::WaitingApproval,
            WakeDisposition::Defer { .. } => ContinuityPhase::WaitingCondition,
            WakeDisposition::Idle => ContinuityPhase::Completed,
        };
        disposition
    }

    pub fn checkpoint_to<S: CheckpointStore>(
        &mut self,
        store: &mut S,
    ) -> Result<u64, ContinuityError> {
        self.platform.validate()?;
        self.platform.checkpoint_generation = self
            .platform
            .checkpoint_generation
            .checked_add(1)
            .ok_or(ContinuityError::InvalidState)?;

        let cognitive_checkpoint = encode_cognitive_checkpoint(self.cognitive.state())?;
        let action_checkpoint =
            encode_action_fabric_checkpoint(self.actions.registry(), self.actions.state())?;
        let bundle = ContinuityBundle {
            capsule_root: self.capsule_root,
            platform: self.platform.clone(),
            cognitive_checkpoint,
            action_checkpoint,
        };
        let bytes = encode_continuity_bundle(&bundle)?;
        store.commit(&bytes).map_err(ContinuityError::Store)?;
        Ok(self.platform.checkpoint_generation)
    }

    pub fn mark_suspended_by_os<S: CheckpointStore>(
        &mut self,
        store: &mut S,
    ) -> Result<u64, ContinuityError> {
        self.platform.phase = ContinuityPhase::SuspendedByOs;
        self.checkpoint_to(store)
    }

    pub fn request_current_action_approval(
        &mut self,
        plan_id: ActionPlanId,
    ) -> Result<&ApprovalRequest, ContinuityError> {
        let plan = self
            .actions
            .state()
            .plans
            .get(&plan_id.0)
            .ok_or(ContinuityError::MissingPlan(plan_id))?;
        let action = plan
            .actions
            .get(plan.cursor)
            .ok_or(ContinuityError::InvalidState)?;
        let descriptor = self
            .actions
            .registry()
            .descriptor(&action.capability)
            .map_err(ActionFabricError::Capability)?;

        let mut required_scopes = descriptor
            .required_scopes
            .iter()
            .map(|scope| scope.as_str().to_owned())
            .collect::<Vec<_>>();
        required_scopes.sort();
        required_scopes.dedup();

        let id = self.platform.next_approval_id;
        self.platform.next_approval_id = self
            .platform
            .next_approval_id
            .checked_add(1)
            .ok_or(ContinuityError::InvalidState)?;

        self.platform.pending_approval = Some(ApprovalRequest {
            id,
            plan_id,
            action_id: action.id,
            required_scopes,
            external_write: action.side_effect == SideEffectClass::ExternalWrite,
            irreversible: action.side_effect == SideEffectClass::Irreversible,
        });
        self.platform.phase = ContinuityPhase::WaitingApproval;
        self.platform
            .pending_approval
            .as_ref()
            .ok_or(ContinuityError::InvalidState)
    }

    pub fn acknowledge_approval(&mut self, approval_id: u64) -> Result<(), ContinuityError> {
        let approval = self
            .platform
            .pending_approval
            .as_ref()
            .ok_or(ContinuityError::InvalidState)?;
        if approval.id != approval_id {
            return Err(ContinuityError::InvalidState);
        }
        self.platform.pending_approval = None;
        self.platform.phase = ContinuityPhase::VerifyingResume;
        Ok(())
    }

    pub fn resume_plan_once_durable<V, S>(
        &mut self,
        plan_id: ActionPlanId,
        authority: &AuthorityGrant,
        verifier: &mut V,
        store: &mut S,
    ) -> Result<ActionStepReport, ContinuityError>
    where
        V: ActionVerifier,
        S: CheckpointStore,
    {
        self.platform.logical_tick = self.platform.logical_tick.saturating_add(1);
        self.platform.phase = ContinuityPhase::VerifyingResume;

        let cognitive_checkpoint = encode_cognitive_checkpoint(self.cognitive.state())?;
        let capsule_root = self.capsule_root;

        let result = {
            let platform = &mut self.platform;
            let actions = &mut self.actions;
            let mut barrier =
                |registry: &CapabilityRegistry, action_state: &ntd_runtime::ActionFabricState| {
                    platform.checkpoint_generation = platform
                        .checkpoint_generation
                        .checked_add(1)
                        .ok_or_else(|| "checkpoint generation overflow".to_owned())?;
                    platform.phase = ContinuityPhase::ActiveExecution;

                    let action_checkpoint =
                        encode_action_fabric_checkpoint(registry, action_state)
                            .map_err(|error| format!("{error:?}"))?;
                    let bundle = ContinuityBundle {
                        capsule_root,
                        platform: platform.clone(),
                        cognitive_checkpoint: cognitive_checkpoint.clone(),
                        action_checkpoint,
                    };
                    let bytes =
                        encode_continuity_bundle(&bundle).map_err(|error| format!("{error:?}"))?;
                    store.commit(&bytes)
                };

            actions.execute_next_with_barrier(
                plan_id,
                authority,
                verifier,
                &mut barrier,
            )
        };

        let report = match result {
            Ok(report) => {
                self.platform.retry.reset();
                report
            }
            Err(error) if is_authority_error(&error) => {
                self.request_current_action_approval(plan_id)?;
                self.checkpoint_to(store)?;
                return Err(ContinuityError::Action(error));
            }
            Err(error) => {
                self.platform.retry.record_failure(self.platform.logical_tick);
                self.platform.phase = ContinuityPhase::FailedRecoverable;
                self.checkpoint_to(store)?;
                return Err(ContinuityError::Action(error));
            }
        };

        self.platform.pending_approval = None;
        self.platform.phase = phase_from_report(&report);
        self.checkpoint_to(store)?;
        Ok(report)
    }
}

fn is_authority_error(error: &ActionFabricError) -> bool {
    matches!(
        error,
        ActionFabricError::Capability(
            CapabilityError::MissingAuthority(_)
                | CapabilityError::ExternalWriteNotAuthorized
                | CapabilityError::IrreversibleNotAuthorized
        )
    )
}

fn phase_from_report(report: &ActionStepReport) -> ContinuityPhase {
    match report.plan_status {
        ActionPlanStatus::Completed => ContinuityPhase::Completed,
        ActionPlanStatus::Suspended => ContinuityPhase::WaitingCondition,
        ActionPlanStatus::Failed => ContinuityPhase::FailedRecoverable,
        ActionPlanStatus::RolledBack => ContinuityPhase::Checkpointed,
        ActionPlanStatus::Ready => match report.action_status {
            Some(ActionStatus::Retryable) => ContinuityPhase::WaitingCondition,
            _ => ContinuityPhase::ActiveExecution,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_is_bounded_and_monotonic() {
        let mut retry = RetryBackoff {
            base_delay_ticks: 2,
            max_delay_ticks: 16,
            ..RetryBackoff::default()
        };

        retry.record_failure(10);
        assert_eq!(retry.next_eligible_tick, 12);
        retry.record_failure(12);
        assert_eq!(retry.next_eligible_tick, 16);
        retry.record_failure(16);
        assert_eq!(retry.next_eligible_tick, 24);
        retry.record_failure(24);
        assert_eq!(retry.next_eligible_tick, 40);
        retry.record_failure(40);
        assert_eq!(retry.next_eligible_tick, 56);
    }

    #[test]
    fn pending_approval_blocks_background_execution() {
        let mut state = PlatformContinuityState::default();
        state.next_approval_id = 2;
        state.pending_approval = Some(ApprovalRequest {
            id: 1,
            plan_id: ActionPlanId(1),
            action_id: ActionId(1),
            required_scopes: vec!["device.write".into()],
            external_write: true,
            irreversible: false,
        });

        let supervisor = TestSupervisorState::from_state(state);
        assert_eq!(
            supervisor.decide(PlatformConstraints {
                user_visible: false,
                foreground_allowed: true,
                background_execution_allowed: true,
                network_available: true,
                charging: false,
                battery_percent: 80,
                thermal_critical: false,
            }),
            WakeDisposition::AwaitApproval
        );
    }

    struct TestSupervisorState {
        platform: PlatformContinuityState,
    }

    impl TestSupervisorState {
        fn from_state(platform: PlatformContinuityState) -> Self {
            Self { platform }
        }

        fn decide(&self, constraints: PlatformConstraints) -> WakeDisposition {
            if self.platform.pending_approval.is_some() {
                return WakeDisposition::AwaitApproval;
            }
            if constraints.user_visible {
                WakeDisposition::Interactive
            } else {
                WakeDisposition::BackgroundOnce
            }
        }
    }
}
