#![forbid(unsafe_code)]

use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
};

use ntd_core::{ActionNode, CapabilityId, SideEffectClass, TaskGraph};
use ntd_ir::TensorOp;
use ntd_runtime::{
    verify_provider_equivalence, ActionFabric, ActionFabricError, ActionOutput, ActionPlanStatus,
    ActionStatus, ActionVerification, ActionVerifier, AdapterResult, AuthorityGrant,
    AuthorityScope, CapabilityAdapter, CapabilityDescriptor, CapabilityDomain, CapabilityError,
    CapabilityRegistry, CpuReferenceProvider, CpuTiledProvider, PrefixCache, Tensor, TypedAction,
};
use ntd_validation::{
    analyze_prefix_reuse, evaluate_device_profile, representative_device_profiles,
    run_logical_continuity_soak, verify_paged_round_trip, DeviceEvidence, EnergySampler,
    EvidenceClass, EvidenceMatrix, GeneralAgentScorecard, TraceRecorder, ValidationTargets,
};

#[derive(Debug)]
struct CountingEnergy {
    samples: Vec<u64>,
    cursor: usize,
}

impl CountingEnergy {
    fn new(samples: Vec<u64>) -> Self {
        Self { samples, cursor: 0 }
    }
}

impl EnergySampler for CountingEnergy {
    fn sample_microjoules(&mut self) -> Option<u64> {
        let value = self.samples.get(self.cursor).copied();
        self.cursor = self.cursor.saturating_add(1);
        value
    }
}

#[test]
fn latency_and_energy_trace_is_platform_sampler_ready() {
    let mut recorder = TraceRecorder::new(CountingEnergy::new(vec![100, 125, 200, 250]));
    let first = recorder.measure("reflex", || 97u32);
    let second = recorder.measure("deep", || 197u32);
    assert_eq!((first, second), (97, 197));
    assert_eq!(recorder.spans().len(), 2);
    assert_eq!(recorder.spans()[0].energy_microjoules, Some(25));
    assert_eq!(recorder.spans()[1].energy_microjoules, Some(50));
    assert_eq!(recorder.total_energy_microjoules(), Some(75));
    assert!(recorder.percentile_nanos(950).is_some());
}

#[test]
fn representative_device_matrix_uses_verified_autotune_and_thermal_fallback() {
    let profiles = representative_device_profiles();
    assert_eq!(profiles.len(), 3);
    for profile in profiles {
        let report = evaluate_device_profile(&profile).expect("matrix report");
        assert!(report.accelerator_disabled_under_heat);
        if profile.capabilities.supports_npu {
            assert_eq!(report.nominal_provider, ntd_runtime::ProviderKind::Npu);
        } else if profile.capabilities.supports_vulkan {
            assert_eq!(report.nominal_provider, ntd_runtime::ProviderKind::Vulkan);
        } else {
            assert!(!matches!(
                report.nominal_provider,
                ntd_runtime::ProviderKind::Vulkan | ntd_runtime::ProviderKind::Npu
            ));
        }
    }

    let left = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("left");
    let right = Tensor::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]).expect("right");
    assert!(verify_provider_equivalence(
        &CpuReferenceProvider,
        &CpuTiledProvider::default(),
        TensorOp::MatMul,
        &[&left, &right],
    )
    .expect("equivalence"));
}

#[test]
fn prefix_reuse_and_paging_are_lossless() {
    let cache = PrefixCache {
        tokens: vec![1, 2, 3, 4, 5],
        kv: Default::default(),
    };
    let reuse = analyze_prefix_reuse(&cache, &[1, 2, 3, 9, 10]);
    assert_eq!(reuse.reusable_tokens, 3);
    assert_eq!(reuse.reuse_permille, 600);

    let bytes = (0..4097)
        .map(|index| u8::try_from(index % 251).expect("byte"))
        .collect::<Vec<_>>();
    let paging = verify_paged_round_trip(&bytes, 257).expect("paging");
    assert_eq!(paging.logical_bytes, bytes.len());
    assert!(paging.page_count > 1);
}

#[test]
fn logical_24h_continuity_soak_preserves_identity_and_sequence() {
    let report = run_logical_continuity_soak(24, 5).expect("soak");
    assert_eq!(report.logical_minutes, 1440);
    assert_eq!(report.checkpoint_cycles, 288);
    assert_eq!(report.final_sequence, 288);
}

struct RecordingAdapter {
    calls: Rc<RefCell<Vec<u64>>>,
}

impl CapabilityAdapter for RecordingAdapter {
    fn execute(
        &mut self,
        action_id: ntd_runtime::ActionId,
        _action: &TypedAction,
    ) -> Result<AdapterResult, String> {
        self.calls.borrow_mut().push(action_id.0);
        Ok(AdapterResult::Completed {
            output: ActionOutput::text("validated", "ok"),
            rollback_token: None,
        })
    }
}

struct FailingAdapter;

impl CapabilityAdapter for FailingAdapter {
    fn execute(
        &mut self,
        _action_id: ntd_runtime::ActionId,
        _action: &TypedAction,
    ) -> Result<AdapterResult, String> {
        Err("offline".into())
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

fn descriptor(
    id: &str,
    domain: CapabilityDomain,
    side_effect: SideEffectClass,
    required_scope: &str,
) -> CapabilityDescriptor {
    let mut descriptor = CapabilityDescriptor::new(CapabilityId(id.into()), 1, domain, side_effect)
        .expect("descriptor");
    descriptor.required_scopes = vec![scope(required_scope)];
    descriptor
}

fn mixed_registry() -> CapabilityRegistry {
    let mut registry = CapabilityRegistry::new();
    registry
        .register(descriptor(
            "browser.observe",
            CapabilityDomain::Browser,
            SideEffectClass::ReadOnly,
            "browser.read",
        ))
        .expect("browser");
    registry
        .register(descriptor(
            "file.write",
            CapabilityDomain::File,
            SideEffectClass::ExternalWrite,
            "files.write",
        ))
        .expect("file");
    registry
        .register(descriptor(
            "app.action",
            CapabilityDomain::App,
            SideEffectClass::ExternalWrite,
            "app.write",
        ))
        .expect("app");
    registry
        .register(descriptor(
            "pc.observe",
            CapabilityDomain::Pc,
            SideEffectClass::ReadOnly,
            "pc.observe",
        ))
        .expect("pc");
    registry
}

fn mixed_authority() -> AuthorityGrant {
    let mut grant = AuthorityGrant::new()
        .with_scope(scope("browser.read"))
        .with_scope(scope("files.write"))
        .with_scope(scope("app.write"))
        .with_scope(scope("pc.observe"));
    grant.allow_external_write = true;
    grant
}

#[test]
fn long_horizon_browser_file_app_pc_plan_commits_all_steps() {
    let registry = mixed_registry();
    let calls = Rc::new(RefCell::new(Vec::new()));
    let mut fabric = ActionFabric::new(registry);
    for capability in ["browser.observe", "file.write", "app.action", "pc.observe"] {
        fabric
            .register_adapter(
                CapabilityId(capability.into()),
                RecordingAdapter {
                    calls: Rc::clone(&calls),
                },
            )
            .expect("adapter");
    }

    let mut nodes = Vec::new();
    let mut payloads = BTreeMap::new();
    for index in 0..128u32 {
        let id = index + 1;
        let lane = index % 4;
        let (capability, side_effect, action) = match lane {
            0 => (
                "browser.observe",
                SideEffectClass::ReadOnly,
                TypedAction::BrowserObserve {
                    target: format!("page-{id}"),
                },
            ),
            1 => (
                "file.write",
                SideEffectClass::ExternalWrite,
                TypedAction::FileWrite {
                    path: format!("/workspace/{id}.txt"),
                    bytes: id.to_le_bytes().to_vec(),
                },
            ),
            2 => (
                "app.action",
                SideEffectClass::ExternalWrite,
                TypedAction::AppAction {
                    app: "ai.ntd97.validation".into(),
                    action: "append".into(),
                    payload: id.to_le_bytes().to_vec(),
                },
            ),
            _ => (
                "pc.observe",
                SideEffectClass::ReadOnly,
                TypedAction::PcObserve {
                    peer: "paired-workstation".into(),
                    surface: format!("surface-{id}"),
                },
            ),
        };
        nodes.push(ActionNode {
            id,
            capability: CapabilityId(capability.into()),
            side_effect,
            verification_required: true,
        });
        payloads.insert(id, action);
    }

    let graph = TaskGraph { actions: nodes };
    let plan = fabric
        .prepare_plan(7001, &graph, payloads)
        .expect("prepare");
    let mut score = GeneralAgentScorecard::default();
    let mut final_status = None;
    for _ in 0..128 {
        score.attempted_steps = score.attempted_steps.saturating_add(1);
        let report = fabric
            .execute_next(plan, &mixed_authority(), &mut AcceptVerifier)
            .expect("execute");
        if report.action_status == Some(ActionStatus::Committed) {
            score.committed_steps = score.committed_steps.saturating_add(1);
        }
        final_status = Some(report.plan_status);
    }

    assert_eq!(final_status, Some(ActionPlanStatus::Completed));
    assert_eq!(score.reliability_permille(), 1000);
    let recorded = calls.borrow();
    assert_eq!(recorded.len(), 128);
    let unique = recorded.iter().copied().collect::<BTreeSet<_>>();
    assert_eq!(unique.len(), 128);
}

#[test]
fn offline_network_failure_does_not_remove_local_agent_capability() {
    let mut registry = CapabilityRegistry::new();
    registry
        .register(descriptor(
            "web.search",
            CapabilityDomain::Web,
            SideEffectClass::ReadOnly,
            "network.read",
        ))
        .expect("web");
    registry
        .register(descriptor(
            "device.observe",
            CapabilityDomain::Device,
            SideEffectClass::ReadOnly,
            "device.read",
        ))
        .expect("device");

    let local_calls = Rc::new(RefCell::new(Vec::new()));
    let mut fabric = ActionFabric::new(registry);
    fabric
        .register_adapter(CapabilityId("web.search".into()), FailingAdapter)
        .expect("web adapter");
    fabric
        .register_adapter(
            CapabilityId("device.observe".into()),
            RecordingAdapter {
                calls: Rc::clone(&local_calls),
            },
        )
        .expect("device adapter");

    let network_graph = TaskGraph {
        actions: vec![ActionNode {
            id: 1,
            capability: CapabilityId("web.search".into()),
            side_effect: SideEffectClass::ReadOnly,
            verification_required: true,
        }],
    };
    let network_plan = fabric
        .prepare_plan(
            8001,
            &network_graph,
            BTreeMap::from([(
                1,
                TypedAction::WebSearch {
                    query: "offline".into(),
                    max_results: 1,
                },
            )]),
        )
        .expect("network plan");
    let network_grant = AuthorityGrant::new().with_scope(scope("network.read"));
    let failed = fabric
        .execute_next(network_plan, &network_grant, &mut AcceptVerifier)
        .expect("retryable");
    assert_eq!(failed.action_status, Some(ActionStatus::Retryable));

    let local_graph = TaskGraph {
        actions: vec![ActionNode {
            id: 2,
            capability: CapabilityId("device.observe".into()),
            side_effect: SideEffectClass::ReadOnly,
            verification_required: true,
        }],
    };
    let local_plan = fabric
        .prepare_plan(
            8002,
            &local_graph,
            BTreeMap::from([(
                2,
                TypedAction::DeviceObserve {
                    surface: "battery".into(),
                },
            )]),
        )
        .expect("local plan");
    let local_grant = AuthorityGrant::new().with_scope(scope("device.read"));
    let local = fabric
        .execute_next(local_plan, &local_grant, &mut AcceptVerifier)
        .expect("local action");
    assert_eq!(local.plan_status, ActionPlanStatus::Completed);
    assert_eq!(local_calls.borrow().len(), 1);
}

#[test]
fn security_boundary_denies_external_write_before_adapter_execution() {
    let mut registry = CapabilityRegistry::new();
    registry
        .register(descriptor(
            "file.write",
            CapabilityDomain::File,
            SideEffectClass::ExternalWrite,
            "files.write",
        ))
        .expect("file");
    let calls = Rc::new(RefCell::new(Vec::new()));
    let mut fabric = ActionFabric::new(registry);
    fabric
        .register_adapter(
            CapabilityId("file.write".into()),
            RecordingAdapter {
                calls: Rc::clone(&calls),
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
            9001,
            &graph,
            BTreeMap::from([(
                1,
                TypedAction::FileWrite {
                    path: "/protected/result".into(),
                    bytes: b"denied".to_vec(),
                },
            )]),
        )
        .expect("plan");
    let scoped_but_read_only = AuthorityGrant::new().with_scope(scope("files.write"));
    assert!(matches!(
        fabric.execute_next(plan, &scoped_but_read_only, &mut AcceptVerifier),
        Err(ActionFabricError::Capability(
            CapabilityError::ExternalWriteNotAuthorized
        ))
    ));
    assert!(calls.borrow().is_empty());
}

#[test]
fn physical_device_evidence_is_required_for_m10_completion() {
    let targets = ValidationTargets {
        required_physical_profiles: vec![
            "mobile-4gb".into(),
            "mobile-8gb".into(),
            "mobile-12gb".into(),
        ],
        max_p95_latency_nanos: 2_000_000_000,
        max_energy_per_task_microjoules: 5_000_000,
        min_reliability_permille: 990,
        min_recovery_permille: 990,
    };
    let mut matrix = EvidenceMatrix::new();
    for profile in &targets.required_physical_profiles {
        matrix.record(DeviceEvidence {
            class: EvidenceClass::CiSurrogate,
            profile: profile.clone(),
            device_fingerprint: format!("emulator-{profile}"),
            p95_latency_nanos: 1,
            energy_per_task_microjoules: 1,
            reliability_permille: 1000,
            recovery_permille: 1000,
            sovereignty_audit_passed: true,
        });
    }
    assert!(!matrix.meets_targets(&targets));

    for (index, profile) in targets.required_physical_profiles.iter().enumerate() {
        matrix.record(DeviceEvidence {
            class: EvidenceClass::PhysicalDevice,
            profile: profile.clone(),
            device_fingerprint: format!("physical-{index}-{profile}"),
            p95_latency_nanos: 1_000_000_000,
            energy_per_task_microjoules: 1_000_000,
            reliability_permille: 995,
            recovery_permille: 995,
            sovereignty_audit_passed: true,
        });
    }
    assert!(matrix.meets_targets(&targets));
}
