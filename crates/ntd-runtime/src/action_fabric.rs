#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use ntd_core::{ActionNode, CapabilityId, SideEffectClass, TaskGraph};

use crate::{
    ActionOutput, AuthorityGrant, CapabilityDescriptor, CapabilityError, CapabilityRegistry,
    TypedAction,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ActionId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ActionPlanId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ActionStatus {
    Prepared = 1,
    Running = 2,
    Suspended = 3,
    Retryable = 4,
    Committed = 5,
    RolledBack = 6,
    Failed = 7,
}

impl TryFrom<u8> for ActionStatus {
    type Error = ActionFabricError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Prepared),
            2 => Ok(Self::Running),
            3 => Ok(Self::Suspended),
            4 => Ok(Self::Retryable),
            5 => Ok(Self::Committed),
            6 => Ok(Self::RolledBack),
            7 => Ok(Self::Failed),
            other => Err(ActionFabricError::InvalidActionStatus(other)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ActionPlanStatus {
    Ready = 1,
    Suspended = 2,
    Completed = 3,
    Failed = 4,
    RolledBack = 5,
}

impl TryFrom<u8> for ActionPlanStatus {
    type Error = ActionFabricError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Ready),
            2 => Ok(Self::Suspended),
            3 => Ok(Self::Completed),
            4 => Ok(Self::Failed),
            5 => Ok(Self::RolledBack),
            other => Err(ActionFabricError::InvalidPlanStatus(other)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedAction {
    pub id: ActionId,
    pub node_id: u32,
    pub capability: CapabilityId,
    pub capability_version: u32,
    pub side_effect: SideEffectClass,
    pub verification_required: bool,
    pub action: TypedAction,
    pub status: ActionStatus,
    pub attempts: u32,
    pub output: Option<ActionOutput>,
    pub resume_token: Option<Vec<u8>>,
    pub rollback_token: Option<Vec<u8>>,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionPlanState {
    pub id: ActionPlanId,
    pub task_id: u64,
    pub cursor: usize,
    pub status: ActionPlanStatus,
    pub actions: Vec<PlannedAction>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionFabricState {
    pub next_plan_id: u64,
    pub next_action_id: u64,
    pub plans: BTreeMap<u64, ActionPlanState>,
}

impl Default for ActionFabricState {
    fn default() -> Self {
        Self {
            next_plan_id: 1,
            next_action_id: 1,
            plans: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdapterResult {
    Completed {
        output: ActionOutput,
        rollback_token: Option<Vec<u8>>,
    },
    Suspended {
        resume_token: Vec<u8>,
        note: String,
    },
    Retryable {
        reason: String,
        resume_token: Option<Vec<u8>>,
    },
}

pub trait CapabilityAdapter {
    fn execute(
        &mut self,
        action_id: ActionId,
        action: &TypedAction,
    ) -> Result<AdapterResult, String>;

    fn resume(
        &mut self,
        action_id: ActionId,
        action: &TypedAction,
        resume_token: &[u8],
    ) -> Result<AdapterResult, String> {
        let _ = (action_id, action, resume_token);
        Err("resume is not supported by this adapter".into())
    }

    fn rollback(
        &mut self,
        action_id: ActionId,
        action: &TypedAction,
        rollback_token: &[u8],
    ) -> Result<(), String> {
        let _ = (action_id, action, rollback_token);
        Err("rollback is not supported by this adapter".into())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionVerification {
    Accept,
    Retry { reason: String },
    Reject { reason: String },
}

pub trait ActionVerifier {
    fn verify(
        &mut self,
        descriptor: &CapabilityDescriptor,
        action: &TypedAction,
        output: &ActionOutput,
    ) -> ActionVerification;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionStepReport {
    pub plan_id: ActionPlanId,
    pub action_id: Option<ActionId>,
    pub action_status: Option<ActionStatus>,
    pub plan_status: ActionPlanStatus,
    pub cursor: usize,
    pub summary: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionFabricError {
    Capability(CapabilityError),
    DuplicateAdapter(CapabilityId),
    MissingAdapter(CapabilityId),
    MissingPlan(ActionPlanId),
    MissingActionPayload(u32),
    UnexpectedActionPayload(u32),
    SideEffectMismatch {
        node_id: u32,
        planned: SideEffectClass,
        registered: SideEffectClass,
    },
    VerificationContractMismatch(u32),
    NonResumableSuspension(CapabilityId),
    AdapterFailure {
        capability: CapabilityId,
        reason: String,
    },
    RollbackFailure {
        capability: CapabilityId,
        reason: String,
    },
    RollbackUnavailable(ActionId),
    TerminalPlan(ActionPlanStatus),
    InvalidActionStatus(u8),
    InvalidPlanStatus(u8),
    InvalidState,
    Overflow,
}

impl From<CapabilityError> for ActionFabricError {
    fn from(value: CapabilityError) -> Self {
        Self::Capability(value)
    }
}

pub struct ActionFabric {
    registry: CapabilityRegistry,
    adapters: BTreeMap<String, Box<dyn CapabilityAdapter>>,
    state: ActionFabricState,
}

impl ActionFabric {
    pub fn new(registry: CapabilityRegistry) -> Self {
        Self {
            registry,
            adapters: BTreeMap::new(),
            state: ActionFabricState::default(),
        }
    }

    pub fn from_state(
        registry: CapabilityRegistry,
        state: ActionFabricState,
    ) -> Result<Self, ActionFabricError> {
        validate_fabric_state(&registry, &state)?;
        Ok(Self {
            registry,
            adapters: BTreeMap::new(),
            state,
        })
    }

    pub fn registry(&self) -> &CapabilityRegistry {
        &self.registry
    }

    pub fn state(&self) -> &ActionFabricState {
        &self.state
    }

    pub fn register_adapter<A>(
        &mut self,
        capability: CapabilityId,
        adapter: A,
    ) -> Result<(), ActionFabricError>
    where
        A: CapabilityAdapter + 'static,
    {
        self.registry.descriptor(&capability)?;
        if self.adapters.contains_key(&capability.0) {
            return Err(ActionFabricError::DuplicateAdapter(capability));
        }
        self.adapters.insert(capability.0, Box::new(adapter));
        Ok(())
    }

    pub fn prepare_plan(
        &mut self,
        task_id: u64,
        graph: &TaskGraph,
        mut payloads: BTreeMap<u32, TypedAction>,
    ) -> Result<ActionPlanId, ActionFabricError> {
        if graph.actions.is_empty() {
            return Err(ActionFabricError::InvalidState);
        }

        let mut seen_nodes = BTreeSet::new();
        let mut validated = Vec::with_capacity(graph.actions.len());
        for node in &graph.actions {
            if !seen_nodes.insert(node.id) {
                return Err(ActionFabricError::InvalidState);
            }

            let action = payloads
                .remove(&node.id)
                .ok_or(ActionFabricError::MissingActionPayload(node.id))?;
            let descriptor = self.registry.validate_action(&node.capability, &action)?;

            if node.side_effect != descriptor.side_effect {
                return Err(ActionFabricError::SideEffectMismatch {
                    node_id: node.id,
                    planned: node.side_effect,
                    registered: descriptor.side_effect,
                });
            }
            if descriptor.verification_required && !node.verification_required {
                return Err(ActionFabricError::VerificationContractMismatch(node.id));
            }

            validated.push((
                node.clone(),
                action,
                descriptor.version,
                node.verification_required || descriptor.verification_required,
            ));
        }

        if let Some((&unexpected, _)) = payloads.iter().next() {
            return Err(ActionFabricError::UnexpectedActionPayload(unexpected));
        }

        let plan_id = ActionPlanId(self.state.next_plan_id);
        let next_plan_id = self
            .state
            .next_plan_id
            .checked_add(1)
            .ok_or(ActionFabricError::Overflow)?;
        let action_count =
            u64::try_from(validated.len()).map_err(|_| ActionFabricError::Overflow)?;
        let first_action_id = self.state.next_action_id;
        let next_action_id = first_action_id
            .checked_add(action_count)
            .ok_or(ActionFabricError::Overflow)?;

        let mut actions = Vec::with_capacity(validated.len());
        for (offset, (node, action, capability_version, verification_required)) in
            validated.into_iter().enumerate()
        {
            let offset = u64::try_from(offset).map_err(|_| ActionFabricError::Overflow)?;
            let action_id = ActionId(
                first_action_id
                    .checked_add(offset)
                    .ok_or(ActionFabricError::Overflow)?,
            );
            actions.push(PlannedAction {
                id: action_id,
                node_id: node.id,
                capability: node.capability,
                capability_version,
                side_effect: node.side_effect,
                verification_required,
                action,
                status: ActionStatus::Prepared,
                attempts: 0,
                output: None,
                resume_token: None,
                rollback_token: None,
                last_error: None,
            });
        }

        self.state.next_plan_id = next_plan_id;
        self.state.next_action_id = next_action_id;
        self.state.plans.insert(
            plan_id.0,
            ActionPlanState {
                id: plan_id,
                task_id,
                cursor: 0,
                status: ActionPlanStatus::Ready,
                actions,
            },
        );
        Ok(plan_id)
    }

    pub fn execute_next<V>(
        &mut self,
        plan_id: ActionPlanId,
        authority: &AuthorityGrant,
        verifier: &mut V,
    ) -> Result<ActionStepReport, ActionFabricError>
    where
        V: ActionVerifier,
    {
        let (cursor, action_snapshot) = {
            let plan = self
                .state
                .plans
                .get(&plan_id.0)
                .ok_or(ActionFabricError::MissingPlan(plan_id))?;
            if matches!(
                plan.status,
                ActionPlanStatus::Completed
                    | ActionPlanStatus::Failed
                    | ActionPlanStatus::RolledBack
            ) {
                return Err(ActionFabricError::TerminalPlan(plan.status));
            }

            let action = plan.actions.get(plan.cursor).cloned();
            (plan.cursor, action)
        };

        let Some(action_snapshot) = action_snapshot else {
            self.finish_plan(plan_id, ActionPlanStatus::Completed)?;
            return Ok(ActionStepReport {
                plan_id,
                action_id: None,
                action_status: None,
                plan_status: ActionPlanStatus::Completed,
                cursor,
                summary: "plan already exhausted".into(),
            });
        };

        let descriptor = self
            .registry
            .validate_action(&action_snapshot.capability, &action_snapshot.action)?
            .clone();
        authority.permits(&descriptor)?;

        let use_resume = matches!(
            action_snapshot.status,
            ActionStatus::Suspended | ActionStatus::Retryable
        ) && action_snapshot.resume_token.is_some();

        if action_snapshot.status == ActionStatus::Suspended && !descriptor.resumable {
            return Err(ActionFabricError::NonResumableSuspension(
                action_snapshot.capability,
            ));
        }
        if matches!(
            action_snapshot.status,
            ActionStatus::Committed | ActionStatus::RolledBack | ActionStatus::Failed
        ) {
            return Err(ActionFabricError::InvalidState);
        }

        {
            let action = self.action_mut(plan_id, cursor)?;
            action.status = ActionStatus::Running;
            action.attempts = action.attempts.saturating_add(1);
        }
        self.set_plan_status(plan_id, ActionPlanStatus::Ready)?;

        let adapter = self
            .adapters
            .get_mut(&action_snapshot.capability.0)
            .ok_or_else(|| ActionFabricError::MissingAdapter(action_snapshot.capability.clone()))?;

        let adapter_result = if use_resume {
            adapter.resume(
                action_snapshot.id,
                &action_snapshot.action,
                action_snapshot
                    .resume_token
                    .as_deref()
                    .ok_or(ActionFabricError::InvalidState)?,
            )
        } else {
            adapter.execute(action_snapshot.id, &action_snapshot.action)
        };

        let adapter_result = match adapter_result {
            Ok(result) => result,
            Err(reason) => {
                let action = self.action_mut(plan_id, cursor)?;
                action.status = ActionStatus::Retryable;
                action.last_error = Some(reason.clone());
                action.resume_token = None;
                let action_id = action.id;
                self.set_plan_status(plan_id, ActionPlanStatus::Ready)?;
                return Ok(ActionStepReport {
                    plan_id,
                    action_id: Some(action_id),
                    action_status: Some(ActionStatus::Retryable),
                    plan_status: ActionPlanStatus::Ready,
                    cursor,
                    summary: reason,
                });
            }
        };

        self.apply_adapter_result(
            plan_id,
            cursor,
            action_snapshot,
            descriptor,
            adapter_result,
            verifier,
        )
    }

    pub fn rollback_plan(
        &mut self,
        plan_id: ActionPlanId,
    ) -> Result<ActionStepReport, ActionFabricError> {
        let snapshots = {
            let plan = self
                .state
                .plans
                .get(&plan_id.0)
                .ok_or(ActionFabricError::MissingPlan(plan_id))?;
            plan.actions.clone()
        };

        for (index, action) in snapshots.iter().enumerate().rev() {
            if action.status != ActionStatus::Committed {
                continue;
            }

            let descriptor = self.registry.descriptor(&action.capability)?.clone();
            if action.side_effect == SideEffectClass::ReadOnly {
                continue;
            }
            if !descriptor.rollback_supported {
                return Err(ActionFabricError::RollbackUnavailable(action.id));
            }
            let token = action
                .rollback_token
                .as_deref()
                .ok_or(ActionFabricError::RollbackUnavailable(action.id))?;

            let adapter = self
                .adapters
                .get_mut(&action.capability.0)
                .ok_or_else(|| ActionFabricError::MissingAdapter(action.capability.clone()))?;
            adapter
                .rollback(action.id, &action.action, token)
                .map_err(|reason| ActionFabricError::RollbackFailure {
                    capability: action.capability.clone(),
                    reason,
                })?;

            let stored = self.action_mut(plan_id, index)?;
            stored.status = ActionStatus::RolledBack;
        }

        self.finish_plan(plan_id, ActionPlanStatus::RolledBack)?;
        let plan = self
            .state
            .plans
            .get(&plan_id.0)
            .ok_or(ActionFabricError::MissingPlan(plan_id))?;

        Ok(ActionStepReport {
            plan_id,
            action_id: None,
            action_status: None,
            plan_status: plan.status,
            cursor: plan.cursor,
            summary: "committed reversible actions rolled back".into(),
        })
    }

    fn apply_adapter_result<V>(
        &mut self,
        plan_id: ActionPlanId,
        cursor: usize,
        snapshot: PlannedAction,
        descriptor: CapabilityDescriptor,
        result: AdapterResult,
        verifier: &mut V,
    ) -> Result<ActionStepReport, ActionFabricError>
    where
        V: ActionVerifier,
    {
        match result {
            AdapterResult::Suspended { resume_token, note } => {
                if !descriptor.resumable {
                    let action = self.action_mut(plan_id, cursor)?;
                    action.status = ActionStatus::Failed;
                    action.last_error = Some("adapter attempted unsupported suspension".into());
                    self.finish_plan(plan_id, ActionPlanStatus::Failed)?;
                    return Err(ActionFabricError::NonResumableSuspension(
                        snapshot.capability,
                    ));
                }
                let action = self.action_mut(plan_id, cursor)?;
                action.status = ActionStatus::Suspended;
                action.resume_token = Some(resume_token);
                action.last_error = Some(note.clone());
                let action_id = action.id;
                self.set_plan_status(plan_id, ActionPlanStatus::Suspended)?;
                Ok(ActionStepReport {
                    plan_id,
                    action_id: Some(action_id),
                    action_status: Some(ActionStatus::Suspended),
                    plan_status: ActionPlanStatus::Suspended,
                    cursor,
                    summary: note,
                })
            }
            AdapterResult::Retryable {
                reason,
                resume_token,
            } => {
                let action = self.action_mut(plan_id, cursor)?;
                action.status = ActionStatus::Retryable;
                action.resume_token = resume_token;
                action.last_error = Some(reason.clone());
                let action_id = action.id;
                self.set_plan_status(plan_id, ActionPlanStatus::Ready)?;
                Ok(ActionStepReport {
                    plan_id,
                    action_id: Some(action_id),
                    action_status: Some(ActionStatus::Retryable),
                    plan_status: ActionPlanStatus::Ready,
                    cursor,
                    summary: reason,
                })
            }
            AdapterResult::Completed {
                output,
                rollback_token,
            } => {
                let verification = if snapshot.verification_required {
                    verifier.verify(&descriptor, &snapshot.action, &output)
                } else {
                    ActionVerification::Accept
                };

                match verification {
                    ActionVerification::Accept => {
                        let action = self.action_mut(plan_id, cursor)?;
                        action.status = ActionStatus::Committed;
                        action.output = Some(output.clone());
                        action.resume_token = None;
                        action.rollback_token = rollback_token;
                        action.last_error = None;
                        let action_id = action.id;

                        let (next_cursor, plan_status) = self.advance_cursor(plan_id)?;
                        Ok(ActionStepReport {
                            plan_id,
                            action_id: Some(action_id),
                            action_status: Some(ActionStatus::Committed),
                            plan_status,
                            cursor: next_cursor,
                            summary: output.summary,
                        })
                    }
                    ActionVerification::Retry { reason } => {
                        let action = self.action_mut(plan_id, cursor)?;
                        action.status = ActionStatus::Retryable;
                        action.output = Some(output);
                        action.resume_token = None;
                        action.rollback_token = rollback_token;
                        action.last_error = Some(reason.clone());
                        let action_id = action.id;
                        self.set_plan_status(plan_id, ActionPlanStatus::Ready)?;
                        Ok(ActionStepReport {
                            plan_id,
                            action_id: Some(action_id),
                            action_status: Some(ActionStatus::Retryable),
                            plan_status: ActionPlanStatus::Ready,
                            cursor,
                            summary: reason,
                        })
                    }
                    ActionVerification::Reject { reason } => {
                        let rolled_back = self.rollback_current_if_possible(
                            plan_id,
                            cursor,
                            &snapshot,
                            &descriptor,
                            rollback_token.as_deref(),
                        )?;
                        let action = self.action_mut(plan_id, cursor)?;
                        action.output = Some(output);
                        action.last_error = Some(reason.clone());
                        let action_status = if rolled_back {
                            ActionStatus::RolledBack
                        } else {
                            ActionStatus::Failed
                        };
                        action.status = action_status;
                        let action_id = action.id;
                        self.finish_plan(plan_id, ActionPlanStatus::Failed)?;
                        Ok(ActionStepReport {
                            plan_id,
                            action_id: Some(action_id),
                            action_status: Some(action_status),
                            plan_status: ActionPlanStatus::Failed,
                            cursor,
                            summary: reason,
                        })
                    }
                }
            }
        }
    }

    fn rollback_current_if_possible(
        &mut self,
        plan_id: ActionPlanId,
        cursor: usize,
        snapshot: &PlannedAction,
        descriptor: &CapabilityDescriptor,
        rollback_token: Option<&[u8]>,
    ) -> Result<bool, ActionFabricError> {
        if !descriptor.rollback_supported {
            return Ok(false);
        }
        let Some(token) = rollback_token else {
            return Ok(false);
        };

        let adapter = self
            .adapters
            .get_mut(&snapshot.capability.0)
            .ok_or_else(|| ActionFabricError::MissingAdapter(snapshot.capability.clone()))?;
        adapter
            .rollback(snapshot.id, &snapshot.action, token)
            .map_err(|reason| ActionFabricError::RollbackFailure {
                capability: snapshot.capability.clone(),
                reason,
            })?;

        let action = self.action_mut(plan_id, cursor)?;
        action.rollback_token = Some(token.to_vec());
        Ok(true)
    }

    fn advance_cursor(
        &mut self,
        plan_id: ActionPlanId,
    ) -> Result<(usize, ActionPlanStatus), ActionFabricError> {
        let plan = self
            .state
            .plans
            .get_mut(&plan_id.0)
            .ok_or(ActionFabricError::MissingPlan(plan_id))?;
        plan.cursor = plan.cursor.saturating_add(1);
        plan.status = if plan.cursor >= plan.actions.len() {
            ActionPlanStatus::Completed
        } else {
            ActionPlanStatus::Ready
        };
        Ok((plan.cursor, plan.status))
    }

    fn action_mut(
        &mut self,
        plan_id: ActionPlanId,
        index: usize,
    ) -> Result<&mut PlannedAction, ActionFabricError> {
        self.state
            .plans
            .get_mut(&plan_id.0)
            .ok_or(ActionFabricError::MissingPlan(plan_id))?
            .actions
            .get_mut(index)
            .ok_or(ActionFabricError::InvalidState)
    }

    fn set_plan_status(
        &mut self,
        plan_id: ActionPlanId,
        status: ActionPlanStatus,
    ) -> Result<(), ActionFabricError> {
        let plan = self
            .state
            .plans
            .get_mut(&plan_id.0)
            .ok_or(ActionFabricError::MissingPlan(plan_id))?;
        plan.status = status;
        Ok(())
    }

    fn finish_plan(
        &mut self,
        plan_id: ActionPlanId,
        status: ActionPlanStatus,
    ) -> Result<(), ActionFabricError> {
        self.set_plan_status(plan_id, status)
    }
}

pub(crate) fn validate_fabric_state(
    registry: &CapabilityRegistry,
    state: &ActionFabricState,
) -> Result<(), ActionFabricError> {
    if state.next_plan_id == 0 || state.next_action_id == 0 {
        return Err(ActionFabricError::InvalidState);
    }
    if state
        .plans
        .keys()
        .next_back()
        .is_some_and(|id| state.next_plan_id <= *id)
    {
        return Err(ActionFabricError::InvalidState);
    }

    let mut action_ids = BTreeSet::new();
    let mut max_action_id = 0u64;
    for (key, plan) in &state.plans {
        if *key != plan.id.0 || plan.id.0 == 0 || plan.cursor > plan.actions.len() {
            return Err(ActionFabricError::InvalidState);
        }

        for action in &plan.actions {
            if action.id.0 == 0 || !action_ids.insert(action.id.0) {
                return Err(ActionFabricError::InvalidState);
            }
            max_action_id = max_action_id.max(action.id.0);
            let descriptor = registry.validate_action(&action.capability, &action.action)?;
            if descriptor.version != action.capability_version
                || descriptor.side_effect != action.side_effect
            {
                return Err(ActionFabricError::InvalidState);
            }
            if descriptor.verification_required && !action.verification_required {
                return Err(ActionFabricError::InvalidState);
            }
        }

        if plan.status == ActionPlanStatus::Completed && plan.cursor != plan.actions.len() {
            return Err(ActionFabricError::InvalidState);
        }
        if plan.cursor < plan.actions.len() {
            let current = &plan.actions[plan.cursor];
            if plan.status == ActionPlanStatus::Suspended
                && (current.status != ActionStatus::Suspended || current.resume_token.is_none())
            {
                return Err(ActionFabricError::InvalidState);
            }
        }
    }

    if state.next_action_id <= max_action_id {
        return Err(ActionFabricError::InvalidState);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AuthorityScope, CapabilityDomain};

    struct EchoAdapter;

    impl CapabilityAdapter for EchoAdapter {
        fn execute(
            &mut self,
            _action_id: ActionId,
            action: &TypedAction,
        ) -> Result<AdapterResult, String> {
            Ok(AdapterResult::Completed {
                output: ActionOutput::text("ok", format!("{action:?}")),
                rollback_token: Some(vec![1]),
            })
        }

        fn rollback(
            &mut self,
            _action_id: ActionId,
            _action: &TypedAction,
            _rollback_token: &[u8],
        ) -> Result<(), String> {
            Ok(())
        }
    }

    struct AcceptVerifier;

    impl ActionVerifier for AcceptVerifier {
        fn verify(
            &mut self,
            _descriptor: &CapabilityDescriptor,
            _action: &TypedAction,
            _output: &ActionOutput,
        ) -> ActionVerification {
            ActionVerification::Accept
        }
    }

    fn registry() -> CapabilityRegistry {
        let mut registry = CapabilityRegistry::new();
        let mut descriptor = CapabilityDescriptor::new(
            CapabilityId("file.write".into()),
            1,
            CapabilityDomain::File,
            SideEffectClass::ExternalWrite,
        )
        .expect("descriptor");
        descriptor.rollback_supported = true;
        descriptor.required_scopes = vec![AuthorityScope::new("files.write").expect("scope")];
        registry.register(descriptor).expect("register");
        registry
    }

    #[test]
    fn external_write_requires_authority_then_commits() {
        let mut fabric = ActionFabric::new(registry());
        fabric
            .register_adapter(CapabilityId("file.write".into()), EchoAdapter)
            .expect("adapter");

        let graph = TaskGraph {
            actions: vec![ActionNode {
                id: 1,
                capability: CapabilityId("file.write".into()),
                side_effect: SideEffectClass::ExternalWrite,
                verification_required: true,
            }],
        };
        let plan = fabric
            .prepare_plan(
                7,
                &graph,
                BTreeMap::from([(
                    1,
                    TypedAction::FileWrite {
                        path: "/tmp/out".into(),
                        bytes: b"hello".to_vec(),
                    },
                )]),
            )
            .expect("plan");

        assert!(matches!(
            fabric.execute_next(plan, &AuthorityGrant::new(), &mut AcceptVerifier),
            Err(ActionFabricError::Capability(
                CapabilityError::MissingAuthority(_)
            ))
        ));

        let mut grant =
            AuthorityGrant::new().with_scope(AuthorityScope::new("files.write").expect("scope"));
        grant.allow_external_write = true;
        let report = fabric
            .execute_next(plan, &grant, &mut AcceptVerifier)
            .expect("execute");

        assert_eq!(report.plan_status, ActionPlanStatus::Completed);
        assert_eq!(report.action_status, Some(ActionStatus::Committed));
    }
}
