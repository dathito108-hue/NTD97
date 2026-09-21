#![forbid(unsafe_code)]

use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    rc::Rc,
};

use ntd_core::{ActionNode, CapabilityId, Intent, SideEffectClass, TaskGraph};
use ntd_platform::{
    CheckpointStore, ContinuityError, ContinuityPhase, ContinuitySupervisor, PlatformConstraints,
    WakeDisposition, WakeReason,
};
use ntd_runtime::{
    ActionFabric, ActionOutput, ActionVerification, ActionVerifier, AdapterResult, AuthorityGrant,
    AuthorityScope, CapabilityAdapter, CapabilityDescriptor, CapabilityDomain, CapabilityRegistry,
    CognitiveIdentity, CognitiveRuntime, TypedAction,
};

#[derive(Clone)]
struct SharedStore {
    bytes: Rc<RefCell<Option<Vec<u8>>>>,
    commits: Rc<Cell<u64>>,
}

impl SharedStore {
    fn new() -> Self {
        Self {
            bytes: Rc::new(RefCell::new(None)),
            commits: Rc::new(Cell::new(0)),
        }
    }

    fn from_bytes(bytes: Vec<u8>) -> Self {
        Self {
            bytes: Rc::new(RefCell::new(Some(bytes))),
            commits: Rc::new(Cell::new(0)),
        }
    }
}

impl CheckpointStore for SharedStore {
    fn commit(&mut self, bytes: &[u8]) -> Result<(), String> {
        *self.bytes.borrow_mut() = Some(bytes.to_vec());
        self.commits.set(self.commits.get().saturating_add(1));
        Ok(())
    }

    fn load(&mut self) -> Result<Option<Vec<u8>>, String> {
        Ok(self.bytes.borrow().clone())
    }
}

struct CrashStore {
    bytes: Rc<RefCell<Option<Vec<u8>>>>,
    commits: u32,
}

impl CrashStore {
    fn new() -> Self {
        Self {
            bytes: Rc::new(RefCell::new(None)),
            commits: 0,
        }
    }
}

impl CheckpointStore for CrashStore {
    fn commit(&mut self, bytes: &[u8]) -> Result<(), String> {
        self.commits = self.commits.saturating_add(1);
        if self.commits == 1 {
            *self.bytes.borrow_mut() = Some(bytes.to_vec());
            Err("simulated process death after durable running state".into())
        } else {
            Err("process unavailable".into())
        }
    }

    fn load(&mut self) -> Result<Option<Vec<u8>>, String> {
        Ok(self.bytes.borrow().clone())
    }
}

struct RecordingAdapter {
    action_ids: Rc<RefCell<Vec<u64>>>,
    durable_commits: Rc<Cell<u64>>,
}

impl CapabilityAdapter for RecordingAdapter {
    fn execute(
        &mut self,
        action_id: ntd_runtime::ActionId,
        _action: &TypedAction,
    ) -> Result<AdapterResult, String> {
        if self.durable_commits.get() == 0 {
            return Err("side effect reached adapter before durability barrier".into());
        }
        self.action_ids.borrow_mut().push(action_id.0);
        Ok(AdapterResult::Completed {
            output: ActionOutput::text("app action committed", format!("action:{}", action_id.0)),
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

fn registry() -> CapabilityRegistry {
    let mut registry = CapabilityRegistry::new();
    let mut descriptor = CapabilityDescriptor::new(
        CapabilityId("app.action".into()),
        1,
        CapabilityDomain::App,
        SideEffectClass::ExternalWrite,
    )
    .expect("descriptor");
    descriptor.required_scopes =
        vec![AuthorityScope::new("android.interact").expect("scope")];
    registry.register(descriptor).expect("register");
    registry
}

fn authority() -> AuthorityGrant {
    let mut grant = AuthorityGrant::new()
        .with_scope(AuthorityScope::new("android.interact").expect("scope"));
    grant.allow_external_write = true;
    grant
}

fn graph() -> TaskGraph {
    TaskGraph {
        actions: vec![
            ActionNode {
                id: 10,
                capability: CapabilityId("app.action".into()),
                side_effect: SideEffectClass::ExternalWrite,
                verification_required: true,
            },
            ActionNode {
                id: 20,
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
            TypedAction::AppAction {
                app: "com.example.notes".into(),
                action: "append".into(),
                payload: b"first".to_vec(),
            },
        ),
        (
            20,
            TypedAction::AppAction {
                app: "com.example.notes".into(),
                action: "append".into(),
                payload: b"second".to_vec(),
            },
        ),
    ])
}

fn build_supervisor(
    registry: CapabilityRegistry,
    adapter: Option<RecordingAdapter>,
) -> (ContinuitySupervisor, ntd_runtime::ActionPlanId, u64) {
    let mut cognition = CognitiveRuntime::new(CognitiveIdentity(*b"NTD97-COGNITION1"));
    let task_id = cognition
        .state_mut()
        .submit_task(
            Intent::new("complete Android continuity acceptance"),
            graph(),
            None,
        )
        .expect("task");
    let task = cognition.state().tasks.get(&task_id).expect("task").clone();

    let mut actions = ActionFabric::new(registry);
    if let Some(adapter) = adapter {
        actions
            .register_adapter(CapabilityId("app.action".into()), adapter)
            .expect("adapter");
    }
    let plan_id = actions
        .prepare_cognitive_task(&task, payloads())
        .expect("plan");

    (
        ContinuitySupervisor::new([0x97; 32], cognition, actions),
        plan_id,
        task_id,
    )
}

#[test]
fn reboot_restore_reuses_task_and_reauthorizes_before_next_side_effect() {
    let registry = registry();
    let store = SharedStore::new();
    let action_ids = Rc::new(RefCell::new(Vec::new()));
    let adapter = RecordingAdapter {
        action_ids: Rc::clone(&action_ids),
        durable_commits: Rc::clone(&store.commits),
    };
    let (mut supervisor, plan_id, task_id) = build_supervisor(registry.clone(), Some(adapter));
    let mut store = store;

    let first = supervisor
        .resume_plan_once_durable(
            plan_id,
            &authority(),
            &mut AcceptVerifier,
            &mut store,
        )
        .expect("first action");
    assert_eq!(first.cursor, 1);
    assert_eq!(action_ids.borrow().as_slice(), &[1]);

    supervisor
        .mark_suspended_by_os(&mut store)
        .expect("suspend checkpoint");

    let mut restored =
        ContinuitySupervisor::restore(registry.clone(), &mut store).expect("restore");
    assert_eq!(
        restored.cognitive().state().identity,
        CognitiveIdentity(*b"NTD97-COGNITION1")
    );
    assert_eq!(
        restored
            .actions()
            .state()
            .plans
            .get(&plan_id.0)
            .expect("plan")
            .task_id,
        task_id
    );

    restored
        .actions_mut()
        .register_adapter(
            CapabilityId("app.action".into()),
            RecordingAdapter {
                action_ids: Rc::clone(&action_ids),
                durable_commits: Rc::clone(&store.commits),
            },
        )
        .expect("adapter");

    let disposition = restored.on_wake(
        WakeReason::Boot,
        PlatformConstraints {
            user_visible: false,
            foreground_allowed: true,
            background_execution_allowed: true,
            network_available: true,
            charging: false,
            battery_percent: 70,
            thermal_critical: false,
        },
    );
    assert_eq!(disposition, WakeDisposition::BackgroundOnce);
    assert_eq!(restored.platform().boot_count, 1);

    let denied = restored.resume_plan_once_durable(
        plan_id,
        &AuthorityGrant::new(),
        &mut AcceptVerifier,
        &mut store,
    );
    assert!(matches!(denied, Err(ContinuityError::Action(_))));
    assert_eq!(restored.platform().phase, ContinuityPhase::WaitingApproval);
    assert_eq!(action_ids.borrow().as_slice(), &[1]);

    let approval_id = restored
        .platform()
        .pending_approval
        .as_ref()
        .expect("approval")
        .id;
    restored
        .acknowledge_approval(approval_id)
        .expect("ack approval");

    let second = restored
        .resume_plan_once_durable(
            plan_id,
            &authority(),
            &mut AcceptVerifier,
            &mut store,
        )
        .expect("second action");
    assert_eq!(second.plan_status, ntd_runtime::ActionPlanStatus::Completed);
    assert_eq!(restored.platform().phase, ContinuityPhase::Completed);
    assert_eq!(action_ids.borrow().as_slice(), &[1, 2]);
}

#[test]
fn process_death_after_running_barrier_restores_unknown_action_with_same_id() {
    let registry = registry();
    let (mut supervisor, plan_id, _) = build_supervisor(registry.clone(), None);
    let mut crash_store = CrashStore::new();

    let result = supervisor.resume_plan_once_durable(
        plan_id,
        &authority(),
        &mut AcceptVerifier,
        &mut crash_store,
    );
    assert!(matches!(result, Err(ContinuityError::Store(_))));

    let crashed_bytes = crash_store
        .bytes
        .borrow()
        .clone()
        .expect("durable running checkpoint");
    let mut recovery_store = SharedStore::from_bytes(crashed_bytes);
    let action_ids = Rc::new(RefCell::new(Vec::new()));

    let mut restored =
        ContinuitySupervisor::restore(registry, &mut recovery_store).expect("restore");
    let plan = restored
        .actions()
        .state()
        .plans
        .get(&plan_id.0)
        .expect("plan");
    assert_eq!(plan.actions[0].status, ntd_runtime::ActionStatus::Running);
    assert_eq!(plan.actions[0].id.0, 1);

    restored
        .actions_mut()
        .register_adapter(
            CapabilityId("app.action".into()),
            RecordingAdapter {
                action_ids: Rc::clone(&action_ids),
                durable_commits: Rc::clone(&recovery_store.commits),
            },
        )
        .expect("adapter");

    let report = restored
        .resume_plan_once_durable(
            plan_id,
            &authority(),
            &mut AcceptVerifier,
            &mut recovery_store,
        )
        .expect("recovered action");
    assert_eq!(report.action_id.expect("action").0, 1);
    assert_eq!(action_ids.borrow().as_slice(), &[1]);
}
