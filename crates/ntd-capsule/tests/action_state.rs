#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use ntd_capsule::{
    sha256, CapsuleBuilder, CapsuleKind, CapsuleView, ChunkStorageView, SectionKind,
};
use ntd_core::{ActionNode, CapabilityId, SideEffectClass, TaskGraph};
use ntd_runtime::{
    decode_action_fabric_checkpoint, encode_action_fabric_checkpoint, ActionFabric, ActionOutput,
    ActionPlanStatus, ActionVerification, ActionVerifier, AdapterResult, AuthorityGrant,
    AuthorityScope, CapabilityAdapter, CapabilityDescriptor, CapabilityDomain, CapabilityRegistry,
    TypedAction,
};

struct SuspendAdapter;

impl CapabilityAdapter for SuspendAdapter {
    fn execute(
        &mut self,
        _action_id: ntd_runtime::ActionId,
        _action: &TypedAction,
    ) -> Result<AdapterResult, String> {
        Ok(AdapterResult::Suspended {
            resume_token: b"resume-web".to_vec(),
            note: "paused".into(),
        })
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
fn ncc97_state_capsule_restores_suspended_action_plan() {
    let registry = registry();
    let mut fabric = ActionFabric::new(registry.clone());
    fabric
        .register_adapter(CapabilityId("web.search".into()), SuspendAdapter)
        .expect("adapter");

    let graph = TaskGraph {
        actions: vec![ActionNode {
            id: 1,
            capability: CapabilityId("web.search".into()),
            side_effect: SideEffectClass::ReadOnly,
            verification_required: true,
        }],
    };
    let plan = fabric
        .prepare_plan(
            42,
            &graph,
            BTreeMap::from([(
                1,
                TypedAction::WebSearch {
                    query: "NTD97 continuity".into(),
                    max_results: 3,
                },
            )]),
        )
        .expect("plan");

    let grant =
        AuthorityGrant::new().with_scope(AuthorityScope::new("network.read").expect("scope"));
    let report = fabric
        .execute_next(plan, &grant, &mut AcceptVerifier)
        .expect("suspend");
    assert_eq!(report.plan_status, ActionPlanStatus::Suspended);

    let checkpoint =
        encode_action_fabric_checkpoint(&registry, fabric.state()).expect("TAF97 checkpoint");

    let base_root = sha256(b"ntd97-cognitive-base");
    let mut builder =
        CapsuleBuilder::new(CapsuleKind::State, *b"NTD97-ACTION-001").with_base_root(base_root);
    builder.push_embedded(SectionKind::ContinuityState, checkpoint.clone());

    let bytes = builder.write().expect("write capsule");
    let view = CapsuleView::read(&bytes).expect("verify capsule");
    let chunk = view
        .chunks
        .iter()
        .find(|chunk| chunk.kind == SectionKind::ContinuityState)
        .expect("continuity chunk");
    let stored = match chunk.storage {
        ChunkStorageView::Embedded(bytes) => bytes,
        ChunkStorageView::External => panic!("action checkpoint should be embedded"),
    };

    let restored_state =
        decode_action_fabric_checkpoint(&registry, stored).expect("decode TAF97");
    let restored = ActionFabric::from_state(registry, restored_state).expect("restore fabric");

    let restored_plan = restored.state().plans.get(&plan.0).expect("plan");
    assert_eq!(restored_plan.task_id, 42);
    assert_eq!(restored_plan.status, ActionPlanStatus::Suspended);
    assert_eq!(
        restored_plan.actions[0].resume_token.as_deref(),
        Some(b"resume-web".as_slice())
    );
}
