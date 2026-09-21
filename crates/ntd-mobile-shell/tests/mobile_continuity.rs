#![forbid(unsafe_code)]

use std::{
    cell::RefCell,
    collections::BTreeMap,
    rc::Rc,
};

use ntd_core::{ActionNode, CapabilityId, Intent, SideEffectClass, TaskGraph};
use ntd_mobile_shell::{
    build_mobile_continuity_bundle, decode_mobile_continuity_bundle,
    encode_mobile_continuity_bundle, restore_mobile_continuity_bundle, MobileContinuityState,
    WakeReason,
};
use ntd_runtime::{
    ActionFabric, ActionOutput, ActionPlanStatus, ActionVerification, ActionVerifier, AdapterResult,
    AuthorityGrant, AuthorityScope, CapabilityAdapter, CapabilityDescriptor, CapabilityDomain,
    CapabilityRegistry, CognitiveIdentity, CognitiveRuntime, TypedAction,
};

struct WriteAdapter {
    calls: Rc<RefCell<u32>>,
}

impl CapabilityAdapter for WriteAdapter {
    fn execute(
        &mut self,
        _action_id: ntd_runtime::ActionId,
        _action: &TypedAction,
    ) -> Result<AdapterResult, String> {
        *self.calls.borrow_mut() += 1;
        Ok(AdapterResult::Completed {
            output: ActionOutput::text("write committed", "ok"),
            rollback_token: None,
        })
    }
}

struct SuspendThenResumeSearch {
    execute_calls: Rc<RefCell<u32>>,
    resume_calls: Rc<RefCell<u32>>,
}

impl CapabilityAdapter for SuspendThenResumeSearch {
    fn execute(
        &mut self,
        _action_id: ntd_runtime::ActionId,
        _action: &TypedAction,
    ) -> Result<AdapterResult, String> {
        *self.execute_calls.borrow_mut() += 1;
        Ok(AdapterResult::Suspended {
            resume_token: b"page-1".to_vec(),
            note: "network paused".into(),
        })
    }

    fn resume(
        &mut self,
        _action_id: ntd_runtime::ActionId,
        _action: &TypedAction,
        resume_token: &[u8],
    ) -> Result<AdapterResult, String> {
        if resume_token != b"page-1" {
            return Err("unexpected resume token".into());
        }
        *self.resume_calls.borrow_mut() += 1;
        Ok(AdapterResult::Completed {
            output: ActionOutput::text("search resumed", "done"),
            rollback_token: None,
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

fn scope(value: &str) -> AuthorityScope {
    AuthorityScope::new(value).expect("scope")
}

fn registry() -> CapabilityRegistry {
    let mut registry = CapabilityRegistry::new();

    let mut write = CapabilityDescriptor::new(
        CapabilityId("file.write".into()),
        1,
        CapabilityDomain::File,
        SideEffectClass::ExternalWrite,
    )
    .expect("write");
    write.required_scopes = vec![scope("files.write")];
    registry.register(write).expect("write register");

    let mut search = CapabilityDescriptor::new(
        CapabilityId("web.search".into()),
        1,
        CapabilityDomain::Web,
        SideEffectClass::ReadOnly,
    )
    .expect("search");
    search.resumable = true;
    search.required_scopes = vec![scope("network.read")];
    registry.register(search).expect("search register");

    registry
}

fn grant() -> AuthorityGrant {
    let mut grant = AuthorityGrant::new()
        .with_scope(scope("files.write"))
        .with_scope(scope("network.read"));
    grant.allow_external_write = true;
    grant
}

#[test]
fn process_death_and_reboot_resume_without_duplicate_committed_side_effect() {
    let registry = registry();
    let write_calls = Rc::new(RefCell::new(0));
    let search_execute_calls = Rc::new(RefCell::new(0));
    let search_resume_calls = Rc::new(RefCell::new(0));

    let mut cognitive = CognitiveRuntime::new(CognitiveIdentity(*b"NTD97-COGNITION1"));
    let task_id = cognitive
        .state_mut()
        .submit_task(
            Intent::new("write result then finish network lookup"),
            TaskGraph {
                actions: vec![
                    ActionNode {
                        id: 1,
                        capability: CapabilityId("file.write".into()),
                        side_effect: SideEffectClass::ExternalWrite,
                        verification_required: true,
                    },
                    ActionNode {
                        id: 2,
                        capability: CapabilityId("web.search".into()),
                        side_effect: SideEffectClass::ReadOnly,
                        verification_required: true,
                    },
                ],
            },
            None,
        )
        .expect("task");
    let task = cognitive.state().tasks.get(&task_id).expect("task").clone();

    let mut actions = ActionFabric::new(registry.clone());
    actions
        .register_adapter(
            CapabilityId("file.write".into()),
            WriteAdapter {
                calls: Rc::clone(&write_calls),
            },
        )
        .expect("write adapter");
    actions
        .register_adapter(
            CapabilityId("web.search".into()),
            SuspendThenResumeSearch {
                execute_calls: Rc::clone(&search_execute_calls),
                resume_calls: Rc::clone(&search_resume_calls),
            },
        )
        .expect("search adapter");

    let plan_id = actions
        .prepare_cognitive_task(
            &task,
            BTreeMap::from([
                (
                    1,
                    TypedAction::FileWrite {
                        path: "/user/result.txt".into(),
                        bytes: b"committed once".to_vec(),
                    },
                ),
                (
                    2,
                    TypedAction::WebSearch {
                        query: "NTD97".into(),
                        max_results: 2,
                    },
                ),
            ]),
        )
        .expect("plan");

    let first = actions
        .execute_next(plan_id, &grant(), &mut AcceptVerifier)
        .expect("write");
    assert_eq!(first.cursor, 1);
    assert_eq!(*write_calls.borrow(), 1);

    let second = actions
        .execute_next(plan_id, &grant(), &mut AcceptVerifier)
        .expect("suspend search");
    assert_eq!(second.plan_status, ActionPlanStatus::Suspended);
    assert_eq!(*search_execute_calls.borrow(), 1);

    let bundle = build_mobile_continuity_bundle(
        &cognitive,
        &registry,
        actions.state(),
        MobileContinuityState::SuspendedByOs,
        WakeReason::Reboot,
        7,
        None,
        None,
    )
    .expect("bundle");

    let bytes = encode_mobile_continuity_bundle(&bundle).expect("encode");
    drop(actions);

    let decoded = decode_mobile_continuity_bundle(&bytes).expect("decode");
    let mut restored = restore_mobile_continuity_bundle(decoded).expect("restore session");

    restored
        .actions
        .register_adapter(
            CapabilityId("web.search".into()),
            SuspendThenResumeSearch {
                execute_calls: Rc::clone(&search_execute_calls),
                resume_calls: Rc::clone(&search_resume_calls),
            },
        )
        .expect("reattach search");

    let resumed = restored
        .actions
        .execute_next(plan_id, &grant(), &mut AcceptVerifier)
        .expect("resume search");

    assert_eq!(resumed.plan_status, ActionPlanStatus::Completed);
    assert_eq!(*write_calls.borrow(), 1);
    assert_eq!(*search_execute_calls.borrow(), 1);
    assert_eq!(*search_resume_calls.borrow(), 1);
    assert_eq!(restored.cognitive.state().identity, bundle.identity);
    assert_eq!(
        restored
            .actions
            .state()
            .plans
            .get(&plan_id.0)
            .expect("plan")
            .task_id,
        task_id
    );
}
