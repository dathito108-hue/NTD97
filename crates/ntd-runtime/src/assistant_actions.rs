#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use ntd_core::{ActionNode, CapabilityId, SideEffectClass, TaskGraph};

use crate::{
    ActionFabric, ActionFabricError, ActionOutput, ActionPlanId, ActionPlanState, ActionPlanStatus,
    ActionStatus, ActionStepReport, ActionValue, ActionVerifier, AuthorityGrant, CognitiveTask,
    TypedAction,
};

pub const NATIVE_ACTION_PROTOCOL_V1: &str = "NTD97_ACTIONS_V1";
pub const NATIVE_ACTION_DIRECT: &str = "DIRECT";
const MAX_ACTIONS: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssistantActionPlan {
    pub graph: TaskGraph,
    pub payloads: BTreeMap<u32, TypedAction>,
    pub canonical_text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssistantPlanDecision {
    Direct,
    Actions(AssistantActionPlan),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssistantPlanError {
    Empty,
    InvalidHeader,
    MissingEnd,
    TrailingContent,
    InvalidLine,
    InvalidNodeId,
    NonCanonicalNodeOrder,
    UnsupportedCapability(String),
    InvalidPayload,
    TooManyActions,
    EmptyActionPlan,
}

pub fn parse_native_action_plan(text: &str) -> Result<AssistantPlanDecision, AssistantPlanError> {
    let normalized = text.trim();
    if normalized.is_empty() {
        return Err(AssistantPlanError::Empty);
    }
    if normalized == NATIVE_ACTION_DIRECT {
        return Ok(AssistantPlanDecision::Direct);
    }

    let mut lines = normalized.lines();
    if lines.next() != Some(NATIVE_ACTION_PROTOCOL_V1) {
        return Err(AssistantPlanError::InvalidHeader);
    }

    let mut graph = TaskGraph::default();
    let mut payloads = BTreeMap::new();
    let mut canonical = String::from(NATIVE_ACTION_PROTOCOL_V1);
    canonical.push('\n');
    let mut expected_id = 1u32;
    let mut ended = false;

    for line in lines.by_ref() {
        if line == "END" {
            ended = true;
            break;
        }
        if graph.actions.len() >= MAX_ACTIONS {
            return Err(AssistantPlanError::TooManyActions);
        }
        let (node, payload, canonical_line) = parse_action_line(line, expected_id)?;
        expected_id = expected_id
            .checked_add(1)
            .ok_or(AssistantPlanError::InvalidNodeId)?;
        canonical.push_str(&canonical_line);
        canonical.push('\n');
        payloads.insert(node.id, payload);
        graph.actions.push(node);
    }

    if !ended {
        return Err(AssistantPlanError::MissingEnd);
    }
    if lines.any(|line| !line.trim().is_empty()) {
        return Err(AssistantPlanError::TrailingContent);
    }
    if graph.actions.is_empty() {
        return Err(AssistantPlanError::EmptyActionPlan);
    }
    canonical.push_str("END");

    Ok(AssistantPlanDecision::Actions(AssistantActionPlan {
        graph,
        payloads,
        canonical_text: canonical,
    }))
}

fn parse_action_line(
    line: &str,
    expected_id: u32,
) -> Result<(ActionNode, TypedAction, String), AssistantPlanError> {
    let mut fields = line.splitn(3, '|');
    let id = fields
        .next()
        .ok_or(AssistantPlanError::InvalidLine)?
        .parse::<u32>()
        .map_err(|_| AssistantPlanError::InvalidNodeId)?;
    let capability = fields.next().ok_or(AssistantPlanError::InvalidLine)?;
    let payload = fields.next().ok_or(AssistantPlanError::InvalidLine)?;
    if id == 0 || id != expected_id {
        return Err(AssistantPlanError::NonCanonicalNodeOrder);
    }
    if payload.trim().is_empty()
        || payload != payload.trim()
        || payload.contains('|')
        || payload.contains('\r')
        || payload.contains('\n')
    {
        return Err(AssistantPlanError::InvalidPayload);
    }

    let typed = match capability {
        "web.search" => TypedAction::WebSearch {
            query: payload.to_owned(),
            max_results: 5,
        },
        "web.fetch" => TypedAction::WebFetch {
            url: payload.to_owned(),
        },
        "browser.observe" => TypedAction::BrowserObserve {
            target: payload.to_owned(),
        },
        "file.read" => TypedAction::FileRead {
            path: payload.to_owned(),
        },
        "device.observe" => TypedAction::DeviceObserve {
            surface: payload.to_owned(),
        },
        "pc.observe" => {
            let (peer, surface) = payload
                .split_once('\t')
                .ok_or(AssistantPlanError::InvalidPayload)?;
            if peer.trim().is_empty()
                || surface.trim().is_empty()
                || peer != peer.trim()
                || surface != surface.trim()
            {
                return Err(AssistantPlanError::InvalidPayload);
            }
            TypedAction::PcObserve {
                peer: peer.to_owned(),
                surface: surface.to_owned(),
            }
        }
        other => return Err(AssistantPlanError::UnsupportedCapability(other.to_owned())),
    };
    typed
        .validate()
        .map_err(|_| AssistantPlanError::InvalidPayload)?;

    let capability_id = CapabilityId(capability.to_owned());
    let node = ActionNode {
        id,
        capability: capability_id,
        side_effect: SideEffectClass::ReadOnly,
        verification_required: true,
    };
    Ok((node, typed, line.to_owned()))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedActionEvidence {
    pub node_id: u32,
    pub capability: CapabilityId,
    pub summary: String,
    pub value: String,
    pub evidence: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifiedActionEvidenceError {
    PlanNotCompleted,
    EmptyPlan,
    ActionNotCommitted(u32),
    MissingOutput(u32),
    InvalidOutput(u32),
}

pub fn collect_verified_action_evidence(
    plan: &ActionPlanState,
) -> Result<Vec<VerifiedActionEvidence>, VerifiedActionEvidenceError> {
    if plan.status != ActionPlanStatus::Completed {
        return Err(VerifiedActionEvidenceError::PlanNotCompleted);
    }
    if plan.actions.is_empty() {
        return Err(VerifiedActionEvidenceError::EmptyPlan);
    }

    plan.actions
        .iter()
        .map(|action| {
            if action.status != ActionStatus::Committed {
                return Err(VerifiedActionEvidenceError::ActionNotCommitted(
                    action.node_id,
                ));
            }
            let output = action
                .output
                .as_ref()
                .ok_or(VerifiedActionEvidenceError::MissingOutput(action.node_id))?;
            Ok(VerifiedActionEvidence {
                node_id: action.node_id,
                capability: action.capability.clone(),
                summary: nonempty_output(action.node_id, output)?.to_owned(),
                value: render_action_value(&output.value),
                evidence: output.evidence.clone(),
            })
        })
        .collect()
}

fn nonempty_output(
    node_id: u32,
    output: &ActionOutput,
) -> Result<&str, VerifiedActionEvidenceError> {
    if output.summary.trim().is_empty() {
        Err(VerifiedActionEvidenceError::InvalidOutput(node_id))
    } else {
        Ok(output.summary.as_str())
    }
}

fn render_action_value(value: &ActionValue) -> String {
    match value {
        ActionValue::None => "none".into(),
        ActionValue::Text(text) => text.clone(),
        ActionValue::Bytes(bytes) => format!("{} bytes", bytes.len()),
        ActionValue::TextList(items) => items.join(
            "
",
        ),
        ActionValue::Fields(fields) => fields
            .iter()
            .map(|(key, value)| format!("{key}={value}"))
            .collect::<Vec<_>>()
            .join(
                "
",
            ),
    }
}

pub fn build_verified_answer_prompt(
    user_message: &str,
    evidence: &[VerifiedActionEvidence],
) -> Result<String, VerifiedActionEvidenceError> {
    if evidence.is_empty() {
        return Err(VerifiedActionEvidenceError::EmptyPlan);
    }
    let mut out = String::from(
        "Verified action results:
",
    );
    for item in evidence {
        out.push_str(&format!(
            "[{} {}] {}
",
            item.node_id, item.capability.0, item.summary
        ));
        if !item.value.is_empty() {
            out.push_str("value: ");
            out.push_str(&item.value);
            out.push('\n');
        }
        for evidence_line in &item.evidence {
            out.push_str("evidence: ");
            out.push_str(evidence_line);
            out.push('\n');
        }
    }
    out.push_str(
        "
User: ",
    );
    out.push_str(user_message);
    out.push_str(
        "
Assistant:",
    );
    Ok(out)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedAssistantActionRun {
    pub plan_id: ActionPlanId,
    pub reports: Vec<ActionStepReport>,
    pub evidence: Vec<VerifiedActionEvidence>,
    pub synthesis_prompt: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssistantActionRunError {
    TaskGraphMismatch,
    Fabric(ActionFabricError),
    Incomplete(ActionPlanStatus),
    Evidence(VerifiedActionEvidenceError),
    MissingPlan,
    StepLimit,
}

impl From<ActionFabricError> for AssistantActionRunError {
    fn from(value: ActionFabricError) -> Self {
        Self::Fabric(value)
    }
}

impl From<VerifiedActionEvidenceError> for AssistantActionRunError {
    fn from(value: VerifiedActionEvidenceError) -> Self {
        Self::Evidence(value)
    }
}

pub fn execute_verified_assistant_plan<V>(
    fabric: &mut ActionFabric,
    task: &CognitiveTask,
    action_plan: &AssistantActionPlan,
    authority: &AuthorityGrant,
    verifier: &mut V,
    user_message: &str,
) -> Result<VerifiedAssistantActionRun, AssistantActionRunError>
where
    V: ActionVerifier,
{
    if task.graph.actions.is_empty() || task.graph != action_plan.graph {
        return Err(AssistantActionRunError::TaskGraphMismatch);
    }

    let plan_id = fabric.prepare_cognitive_task(task, action_plan.payloads.clone())?;
    let mut reports = Vec::with_capacity(action_plan.graph.actions.len());

    for _ in 0..action_plan.graph.actions.len() {
        let report = fabric.execute_next(plan_id, authority, verifier)?;
        let status = report.plan_status;
        let action_status = report.action_status;
        reports.push(report);

        match status {
            ActionPlanStatus::Completed => break,
            ActionPlanStatus::Failed
            | ActionPlanStatus::RolledBack
            | ActionPlanStatus::Suspended => {
                return Err(AssistantActionRunError::Incomplete(status));
            }
            ActionPlanStatus::Ready => {
                if matches!(
                    action_status,
                    Some(ActionStatus::Retryable | ActionStatus::Suspended)
                ) {
                    return Err(AssistantActionRunError::Incomplete(status));
                }
            }
        }
    }

    let plan = fabric
        .state()
        .plans
        .get(&plan_id.0)
        .ok_or(AssistantActionRunError::MissingPlan)?;
    if plan.status != ActionPlanStatus::Completed {
        return Err(AssistantActionRunError::StepLimit);
    }

    let evidence = collect_verified_action_evidence(plan)?;
    let synthesis_prompt = build_verified_answer_prompt(user_message, &evidence)?;
    Ok(VerifiedAssistantActionRun {
        plan_id,
        reports,
        evidence,
        synthesis_prompt,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ActionId, PlannedAction};

    #[test]
    fn direct_plan_is_explicit() {
        assert_eq!(
            parse_native_action_plan("DIRECT"),
            Ok(AssistantPlanDecision::Direct)
        );
    }

    #[test]
    fn read_only_protocol_materializes_nonempty_task_graph_and_payloads() {
        let parsed = parse_native_action_plan(
            "NTD97_ACTIONS_V1
1|web.search|NTD97 mobile
2|file.read|/notes/result.txt
END",
        )
        .expect("plan");
        let AssistantPlanDecision::Actions(plan) = parsed else {
            panic!("expected actions");
        };

        assert_eq!(plan.graph.actions.len(), 2);
        assert_eq!(plan.payloads.len(), 2);
        assert_eq!(plan.graph.actions[0].capability.0, "web.search");
        assert_eq!(plan.graph.actions[1].capability.0, "file.read");
        assert!(plan
            .graph
            .actions
            .iter()
            .all(|node| node.side_effect == SideEffectClass::ReadOnly));
    }

    #[test]
    fn unsupported_write_action_fails_closed() {
        assert_eq!(
            parse_native_action_plan(
                "NTD97_ACTIONS_V1
1|file.write|/tmp/out
END"
            ),
            Err(AssistantPlanError::UnsupportedCapability(
                "file.write".into()
            ))
        );
    }

    #[test]
    fn noncanonical_node_order_is_rejected() {
        assert_eq!(
            parse_native_action_plan(
                "NTD97_ACTIONS_V1
2|web.search|query
END"
            ),
            Err(AssistantPlanError::NonCanonicalNodeOrder)
        );
    }

    fn planned_action(
        node_id: u32,
        capability: &str,
        status: ActionStatus,
        output: Option<ActionOutput>,
    ) -> PlannedAction {
        PlannedAction {
            id: ActionId(u64::from(node_id)),
            node_id,
            capability: CapabilityId(capability.into()),
            capability_version: 1,
            side_effect: SideEffectClass::ReadOnly,
            verification_required: true,
            action: TypedAction::DeviceObserve {
                surface: "battery".into(),
            },
            status,
            attempts: 1,
            output,
            resume_token: None,
            rollback_token: None,
            last_error: None,
        }
    }

    #[test]
    fn final_answer_context_accepts_only_committed_verified_outputs() {
        let plan = ActionPlanState {
            id: crate::ActionPlanId(1),
            task_id: 7,
            cursor: 1,
            status: ActionPlanStatus::Completed,
            actions: vec![planned_action(
                1,
                "device.observe",
                ActionStatus::Committed,
                Some(ActionOutput {
                    summary: "battery observed".into(),
                    value: ActionValue::Fields(BTreeMap::from([
                        ("percent".into(), "77".into()),
                        ("charging".into(), "true".into()),
                    ])),
                    evidence: vec!["device-local".into()],
                }),
            )],
        };

        let evidence = collect_verified_action_evidence(&plan).expect("verified evidence");
        let prompt = build_verified_answer_prompt("battery?", &evidence).expect("prompt");

        assert!(prompt.contains("battery observed"));
        assert!(prompt.contains("percent=77"));
        assert!(prompt.contains("evidence: device-local"));
        assert!(prompt.ends_with(
            "User: battery?
Assistant:"
        ));
    }

    struct DeviceAdapter;

    impl crate::CapabilityAdapter for DeviceAdapter {
        fn execute(
            &mut self,
            _action_id: crate::ActionId,
            action: &TypedAction,
        ) -> Result<crate::AdapterResult, String> {
            match action {
                TypedAction::DeviceObserve { surface } => Ok(crate::AdapterResult::Completed {
                    output: ActionOutput {
                        summary: format!("observed {surface}"),
                        value: ActionValue::Fields(BTreeMap::from([
                            ("battery".into(), "77".into()),
                            ("charging".into(), "true".into()),
                        ])),
                        evidence: vec!["verified:device-local".into()],
                    },
                    rollback_token: None,
                }),
                _ => Err("unexpected action".into()),
            }
        }
    }

    struct AcceptVerifier;

    impl ActionVerifier for AcceptVerifier {
        fn verify(
            &mut self,
            _descriptor: &crate::CapabilityDescriptor,
            _action: &TypedAction,
            output: &ActionOutput,
        ) -> crate::ActionVerification {
            if output
                .evidence
                .iter()
                .any(|item| item == "verified:device-local")
            {
                crate::ActionVerification::Accept
            } else {
                crate::ActionVerification::Reject {
                    reason: "missing device evidence".into(),
                }
            }
        }
    }

    fn device_fabric() -> ActionFabric {
        let mut registry = crate::CapabilityRegistry::new();
        registry
            .register(
                crate::CapabilityDescriptor::new(
                    CapabilityId("device.observe".into()),
                    1,
                    crate::CapabilityDomain::Device,
                    SideEffectClass::ReadOnly,
                )
                .expect("descriptor"),
            )
            .expect("register");
        let mut fabric = ActionFabric::new(registry);
        fabric
            .register_adapter(CapabilityId("device.observe".into()), DeviceAdapter)
            .expect("adapter");
        fabric
    }

    #[test]
    fn governed_execution_synthesizes_only_verified_committed_results() {
        let parsed = parse_native_action_plan(
            "NTD97_ACTIONS_V1\n1|device.observe|battery\nEND",
        )
        .expect("parse");
        let AssistantPlanDecision::Actions(plan) = parsed else {
            panic!("expected actions");
        };

        let mut cognition =
            crate::CognitiveRuntime::new(crate::CognitiveIdentity(*b"NTD97-ACTIONRUN1"));
        let task_id = cognition
            .state_mut()
            .submit_task(
                ntd_core::Intent::new("battery status"),
                plan.graph.clone(),
                None,
            )
            .expect("task");
        let task = cognition.state().tasks.get(&task_id).expect("task").clone();

        let result = execute_verified_assistant_plan(
            &mut device_fabric(),
            &task,
            &plan,
            &AuthorityGrant::new(),
            &mut AcceptVerifier,
            "battery status",
        )
        .expect("verified run");

        assert_eq!(result.evidence.len(), 1);
        assert_eq!(result.reports.len(), 1);
        assert!(result.synthesis_prompt.contains("observed battery"));
        assert!(result.synthesis_prompt.contains("battery=77"));
        assert!(result
            .synthesis_prompt
            .contains("evidence: verified:device-local"));
    }

    struct RetryAdapter;

    impl crate::CapabilityAdapter for RetryAdapter {
        fn execute(
            &mut self,
            _action_id: crate::ActionId,
            _action: &TypedAction,
        ) -> Result<crate::AdapterResult, String> {
            Ok(crate::AdapterResult::Retryable {
                reason: "not ready".into(),
                resume_token: None,
            })
        }
    }

    #[test]
    fn incomplete_action_plan_never_produces_synthesis_prompt() {
        let parsed = parse_native_action_plan(
            "NTD97_ACTIONS_V1\n1|device.observe|battery\nEND",
        )
        .expect("parse");
        let AssistantPlanDecision::Actions(plan) = parsed else {
            panic!("expected actions");
        };

        let mut registry = crate::CapabilityRegistry::new();
        registry
            .register(
                crate::CapabilityDescriptor::new(
                    CapabilityId("device.observe".into()),
                    1,
                    crate::CapabilityDomain::Device,
                    SideEffectClass::ReadOnly,
                )
                .expect("descriptor"),
            )
            .expect("register");
        let mut fabric = ActionFabric::new(registry);
        fabric
            .register_adapter(CapabilityId("device.observe".into()), RetryAdapter)
            .expect("adapter");

        let mut cognition =
            crate::CognitiveRuntime::new(crate::CognitiveIdentity(*b"NTD97-ACTIONRUN1"));
        let task_id = cognition
            .state_mut()
            .submit_task(
                ntd_core::Intent::new("battery status"),
                plan.graph.clone(),
                None,
            )
            .expect("task");
        let task = cognition.state().tasks.get(&task_id).expect("task").clone();

        assert!(matches!(
            execute_verified_assistant_plan(
                &mut fabric,
                &task,
                &plan,
                &AuthorityGrant::new(),
                &mut AcceptVerifier,
                "battery status",
            ),
            Err(AssistantActionRunError::Incomplete(ActionPlanStatus::Ready))
        ));
    }

    #[test]
    fn final_answer_context_rejects_uncommitted_action() {
        let plan = ActionPlanState {
            id: crate::ActionPlanId(1),
            task_id: 7,
            cursor: 0,
            status: ActionPlanStatus::Completed,
            actions: vec![planned_action(
                1,
                "device.observe",
                ActionStatus::Running,
                Some(ActionOutput::text("unsafe", "not verified")),
            )],
        };

        assert_eq!(
            collect_verified_action_evidence(&plan),
            Err(VerifiedActionEvidenceError::ActionNotCommitted(1))
        );
    }
}
