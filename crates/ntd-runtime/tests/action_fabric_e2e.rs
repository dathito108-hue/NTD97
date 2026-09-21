#![forbid(unsafe_code)]

use std::{
    cell::RefCell,
    collections::BTreeMap,
    rc::Rc,
};

use ntd_core::{ActionNode, CapabilityId, Intent, SideEffectClass, TaskGraph};
use ntd_runtime::{
    decode_action_fabric_checkpoint, encode_action_fabric_checkpoint, ActionFabric,
    ActionFabricError, ActionOutput, ActionPlanStatus, ActionStatus, ActionValue,
    ActionVerification, ActionVerifier, AdapterResult, AuthorityGrant, AuthorityScope,
    CapabilityAdapter, CapabilityDescriptor, CapabilityDomain, CapabilityError, CapabilityRegistry,
    CognitiveIdentity, CognitiveRuntime, TypedAction,
};

struct FlakySearchAdapter {
    calls: Rc<RefCell<u32>>,
    action_ids: Rc<RefCell<Vec<u64>>>,
}

impl CapabilityAdapter for FlakySearchAdapter {
    fn execute(
        &mut self,
        action_id: ntd_runtime::ActionId,
        _action: &TypedAction,
    ) -> Result<AdapterResult, String> {
        self.action_ids.borrow_mut().push(action_id.0);
        let mut calls = self.calls.borrow_mut();
        *calls += 1;
        if *calls == 1 {
            Err("temporary network failure".into())
        } else {
            Ok(AdapterResult::Completed {
                output: ActionOutput::text("retry succeeded", "ok"),
                rollback_token: None,
            })
        }
    }
}

struct SearchAdapter;

impl CapabilityAdapter for SearchAdapter {
    fn execute(
        &mut self,
        _action_id: ntd_runtime::ActionId,
        _action: &TypedAction,
    ) -> Result<AdapterResult, String> {
        Ok(AdapterResult::Suspended {
            resume_token: b"search-page-1".to_vec(),
            note: "network continuation required".into(),
        })
    }

    fn resume(
        &mut self,
        _action_id: ntd_runtime::ActionId,
        _action: &TypedAction,
        resume_token: &[u8],
    ) -> Result<AdapterResult, String> {
        if resume_token != b"search-page-1" {
            return Err("unexpected resume token".into());
        }
        Ok(AdapterResult::Completed {
            output: ActionOutput {
                summary: "search completed".into(),
                value: ActionValue::TextList(vec![
                    "https://example.test/a".into(),
                    "https://example.test/b".into(),
                ]),
                evidence: vec!["provider:web.search".into()],
            },
            rollback_token: None,
        })
    }
}

struct FileWriteAdapter {
    rollback_count: Rc<RefCell<u32>>,
}

impl CapabilityAdapter for FileWriteAdapter {
    fn execute(
        &mut self,
        action_id: ntd_runtime::ActionId,
        _action: &TypedAction,
    ) -> Result<AdapterResult, String> {
        Ok(AdapterResult::Completed {
            output: ActionOutput::text("file written", format!("action:{}", action_id.0)),
            rollback_token: Some(format!("restore:{}", action_id.0).into_bytes()),
        })
    }

    fn rollback(
        &mut self,
        _action_id: ntd_runtime::ActionId,
        _action: &TypedAction,
        rollback_token: &[u8],
    ) -> Result<(), String> {
        if !rollback_token.starts_with(b"restore:") {
            return Err("invalid rollback token".into());
        }
        *self.rollback_count.borrow_mut() += 1;
        Ok(())
    }
}

struct ObserveAdapter;

impl CapabilityAdapter for ObserveAdapter {
    fn execute(
        &mut self,
        _action_id: ntd_runtime::ActionId,
        _action: &TypedAction,
    ) -> Result<AdapterResult, String> {
        Ok(AdapterResult::Completed {
            output: ActionOutput::text("device observed", "screen:ready"),
            rollback_token: None,
        })
    }
}

struct AppAdapter;

impl CapabilityAdapter for AppAdapter {
    fn execute(
        &mut self,
        action_id: ntd_runtime::ActionId,
        _action: &TypedAction,
    ) -> Result<AdapterResult, String> {
        Ok(AdapterResult::Completed {
            output: ActionOutput::text("app action completed", format!("id:{}", action_id.0)),
            rollback_token: Some(format!("undo:{}", action_id.0).into_bytes()),
        })
    }

    fn rollback(
        &mut self,
        _action_id: ntd_runtime::ActionId,
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

struct RejectVerifier;

impl ActionVerifier for RejectVerifier {
    fn verify(
        &mut self,
        _descriptor: &CapabilityDescriptor,
        _action: &TypedAction,
        _output: &ActionOutput,
    ) -> ActionVerification {
        ActionVerification::Reject {
            reason: "postcondition mismatch".into(),
        }
    }
}

fn scope(value: &str) -> AuthorityScope {
    AuthorityScope::new(value).expect("scope")
}

fn descriptor(
    id: &str,
    domain: CapabilityDomain,
    side_effect: SideEffectClass,
    required_scope: &str,
) -> CapabilityDescriptor {
    let mut descriptor =
        CapabilityDescriptor::new(CapabilityId(id.into()), 1, domain, side_effect)
            .expect("descriptor");
    descriptor.required_scopes = vec![scope(required_scope)];
    descriptor
}

fn registry() -> CapabilityRegistry {
    let mut registry = CapabilityRegistry::new();

    let mut web = descriptor(
        "web.search",
        CapabilityDomain::Web,
        SideEffectClass::ReadOnly,
        "network.read",
    );
    web.resumable = true;
    registry.register(web).expect("web");

    let mut file = descriptor(
        "file.write",
        CapabilityDomain::File,
        SideEffectClass::ExternalWrite,
        "files.write",
    );
    file.rollback_supported = true;
    registry.register(file).expect("file");

    registry
        .register(descriptor(
            "device.observe",
            CapabilityDomain::Device,
            SideEffectClass::ReadOnly,
            "device.observe",
        ))
        .expect("device");

    let mut app = descriptor(
        "app.action",
        CapabilityDomain::App,
        SideEffectClass::ExternalWrite,
        "android.interact",
    );
    app.rollback_supported = true;
    registry.register(app).expect("app");

    registry
}

fn authority() -> AuthorityGrant {
    let mut grant = AuthorityGrant::new()
        .with_scope(scope("network.read"))
        .with_scope(scope("files.write"))
        .with_scope(scope("device.observe"))
        .with_scope(scope("android.interact"));
    grant.allow_external_write = true;
    grant
}

fn graph() -> TaskGraph {
    TaskGraph {
        actions: vec![
            ActionNode {
                id: 10,
                capability: CapabilityId("web.search".into()),
                side_effect: SideEffectClass::ReadOnly,
                verification_required: true,
            },
            ActionNode {
                id: 20,
                capability: CapabilityId("file.write".into()),
                side_effect: SideEffectClass::ExternalWrite,
                verification_required: true,
            },
            ActionNode {
                id: 30,
                capability: CapabilityId("device.observe".into()),
                side_effect: SideEffectClass::ReadOnly,
                verification_required: true,
            },
            ActionNode {
                id: 40,
                capability: CapabilityId("app.action".into()),
                side_effect: SideEffectClass::ExternalWrite,
                verification_required: true,
            },
        ],
    }
}

fn payloads() -> BTreeMap<u32, TypedAction> {
    BTreeMap::from([
        (
            10,
            TypedAction::WebSearch {
                query: "NTD97 native runtime".into(),
                max_results: 4,
            },
        ),
        (
            20,
            TypedAction::FileWrite {
                path: "/workspace/result.txt".into(),
                bytes: b"verified result".to_vec(),
            },
        ),
        (
            30,
            TypedAction::DeviceObserve {
                surface: "screen.current".into(),
            },
        ),
        (
            40,
            TypedAction::AppAction {
                app: "com.example.notes".into(),
                action: "append".into(),
                payload: b"NTD97 completed".to_vec(),
            },
        ),
    ])
}

#[test]
fn multi_surface_action_plan_resumes_after_checkpoint() {
    let registry = registry();
    let mut fabric = ActionFabric::new(registry.clone());
    fabric
        .register_adapter(CapabilityId("web.search".into()), SearchAdapter)
        .expect("search adapter");

    let mut cognition = CognitiveRuntime::new(CognitiveIdentity(*b"NTD97-COGNITION1"));
    let task_id = cognition
        .state_mut()
        .submit_task(Intent::new("execute governed multi-surface task"), graph(), None)
        .expect("cognitive task");
    let task = cognition.state().tasks.get(&task_id).expect("task").clone();

    let plan = fabric
        .prepare_cognitive_task(&task, payloads())
        .expect("prepare");

    let first = fabric
        .execute_next(plan, &authority(), &mut AcceptVerifier)
        .expect("first step");
    assert_eq!(first.plan_status, ActionPlanStatus::Suspended);
    assert_eq!(first.action_status, Some(ActionStatus::Suspended));

    let checkpoint =
        encode_action_fabric_checkpoint(&registry, fabric.state()).expect("checkpoint");
    let restored_state =
        decode_action_fabric_checkpoint(&registry, &checkpoint).expect("restore state");
    let mut restored = ActionFabric::from_state(registry.clone(), restored_state).expect("fabric");

    restored
        .register_adapter(CapabilityId("web.search".into()), SearchAdapter)
        .expect("search adapter");
    restored
        .register_adapter(
            CapabilityId("file.write".into()),
            FileWriteAdapter {
                rollback_count: Rc::new(RefCell::new(0)),
            },
        )
        .expect("file adapter");
    restored
        .register_adapter(CapabilityId("device.observe".into()), ObserveAdapter)
        .expect("device adapter");
    restored
        .register_adapter(CapabilityId("app.action".into()), AppAdapter)
        .expect("app adapter");

    assert!(matches!(
        restored.execute_next(plan, &AuthorityGrant::new(), &mut AcceptVerifier),
        Err(ActionFabricError::Capability(
            CapabilityError::MissingAuthority(_)
        ))
    ));

    let second = restored
        .execute_next(plan, &authority(), &mut AcceptVerifier)
        .expect("resume search");
    assert_eq!(second.action_status, Some(ActionStatus::Committed));
    assert_eq!(second.cursor, 1);

    let third = restored
        .execute_next(plan, &authority(), &mut AcceptVerifier)
        .expect("file");
    assert_eq!(third.cursor, 2);

    let fourth = restored
        .execute_next(plan, &authority(), &mut AcceptVerifier)
        .expect("observe");
    assert_eq!(fourth.cursor, 3);

    let fifth = restored
        .execute_next(plan, &authority(), &mut AcceptVerifier)
        .expect("app");
    assert_eq!(fifth.plan_status, ActionPlanStatus::Completed);
    assert_eq!(fifth.cursor, 4);

    let plan_state = restored.state().plans.get(&plan.0).expect("plan state");
    assert_eq!(plan_state.task_id, task_id);
    assert_eq!(plan_state.actions[0].attempts, 2);
    assert!(plan_state
        .actions
        .iter()
        .all(|action| action.status == ActionStatus::Committed));
}

#[test]
fn transient_adapter_failure_retries_with_same_action_id() {
    let registry = registry();
    let calls = Rc::new(RefCell::new(0));
    let action_ids = Rc::new(RefCell::new(Vec::new()));
    let mut fabric = ActionFabric::new(registry);
    fabric
        .register_adapter(
            CapabilityId("web.search".into()),
            FlakySearchAdapter {
                calls: Rc::clone(&calls),
                action_ids: Rc::clone(&action_ids),
            },
        )
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
            88,
            &graph,
            BTreeMap::from([(
                1,
                TypedAction::WebSearch {
                    query: "retry".into(),
                    max_results: 1,
                },
            )]),
        )
        .expect("plan");

    let first = fabric
        .execute_next(plan, &authority(), &mut AcceptVerifier)
        .expect("retryable report");
    assert_eq!(first.action_status, Some(ActionStatus::Retryable));

    let second = fabric
        .execute_next(plan, &authority(), &mut AcceptVerifier)
        .expect("retry success");
    assert_eq!(second.plan_status, ActionPlanStatus::Completed);
    assert_eq!(action_ids.borrow().as_slice(), &[1, 1]);
}

#[test]
fn rejected_external_write_is_rolled_back() {
    let registry = registry();
    let rollback_count = Rc::new(RefCell::new(0));
    let mut fabric = ActionFabric::new(registry);
    fabric
        .register_adapter(
            CapabilityId("file.write".into()),
            FileWriteAdapter {
                rollback_count: Rc::clone(&rollback_count),
            },
        )
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
            9,
            &graph,
            BTreeMap::from([(
                1,
                TypedAction::FileWrite {
                    path: "/workspace/reject.txt".into(),
                    bytes: b"candidate".to_vec(),
                },
            )]),
        )
        .expect("plan");

    let report = fabric
        .execute_next(plan, &authority(), &mut RejectVerifier)
        .expect("execute");

    assert_eq!(report.plan_status, ActionPlanStatus::Failed);
    assert_eq!(report.action_status, Some(ActionStatus::RolledBack));
    assert_eq!(*rollback_count.borrow(), 1);
}
