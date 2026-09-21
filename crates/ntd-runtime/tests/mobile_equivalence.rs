#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use ntd_ir::{DType, Graph, IrVersion, Node, NodeId, OpKind, TensorOp, ValueDecl, ValueId, ValueType};
use ntd_runtime::{
    npu_provider, vulkan_provider, AdaptiveExecutionProvider, AutotuneTable,
    CpuReferenceMobileProvider, CpuReferenceProvider, CpuTiledProvider, DeviceCapabilities,
    GraphExecutor, ProviderKind, ProviderMeasurement, ResourceSnapshot, Tensor,
    ThermalState,
};

fn tensor_decl(id: u32, rank: u8) -> ValueDecl {
    ValueDecl {
        id: ValueId(id),
        ty: ValueType::Tensor {
            dtype: DType::F32,
            rank,
        },
    }
}

fn matmul_graph() -> Graph {
    Graph {
        version: IrVersion::CURRENT,
        inputs: vec![tensor_decl(0, 2), tensor_decl(1, 2)],
        outputs: vec![ValueId(2)],
        nodes: vec![Node {
            id: NodeId(0),
            op: OpKind::Tensor(TensorOp::MatMul),
            inputs: vec![ValueId(0), ValueId(1)],
            outputs: vec![tensor_decl(2, 2)],
        }],
    }
}

fn inputs() -> BTreeMap<ValueId, Tensor> {
    BTreeMap::from([
        (
            ValueId(0),
            Tensor::new(
                vec![2, 3],
                vec![
                    1.0, 2.0, 3.0, //
                    4.0, 5.0, 6.0,
                ],
            )
            .expect("lhs"),
        ),
        (
            ValueId(1),
            Tensor::new(
                vec![3, 2],
                vec![
                    7.0, 8.0, //
                    9.0, 10.0, //
                    11.0, 12.0,
                ],
            )
            .expect("rhs"),
        ),
    ])
}

fn snapshot(available_ram_bytes: u64, battery_percent: u8, thermal: ThermalState) -> ResourceSnapshot {
    ResourceSnapshot {
        available_ram_bytes,
        battery_percent,
        charging: false,
        thermal,
        latency_budget_ms: 100,
    }
}

fn all_ops() -> Vec<TensorOp> {
    vec![
        TensorOp::Add,
        TensorOp::Mul,
        TensorOp::MatMul,
        TensorOp::QuantizedMatMul,
        TensorOp::RmsNorm,
        TensorOp::Softmax,
        TensorOp::Gather,
        TensorOp::RotaryPosition,
        TensorOp::CausalAttention,
    ]
}

fn reference_output() -> Vec<Tensor> {
    GraphExecutor::new(CpuReferenceProvider)
        .execute(&matmul_graph(), inputs())
        .expect("reference")
}

#[test]
fn representative_mobile_profiles_preserve_graph_output() {
    const GIB: u64 = 1024 * 1024 * 1024;
    let expected = reference_output();

    let profiles = [
        DeviceCapabilities {
            logical_cores: 4,
            total_ram_bytes: 4 * GIB,
            supports_vulkan: false,
            supports_npu: false,
        },
        DeviceCapabilities {
            logical_cores: 8,
            total_ram_bytes: 8 * GIB,
            supports_vulkan: true,
            supports_npu: false,
        },
        DeviceCapabilities {
            logical_cores: 8,
            total_ram_bytes: 12 * GIB,
            supports_vulkan: true,
            supports_npu: true,
        },
    ];

    for (index, device) in profiles.into_iter().enumerate() {
        let policy = ntd_runtime::ComputePolicy::for_device(&device);
        let mut adaptive = AdaptiveExecutionProvider::new(
            device.clone(),
            snapshot(device.total_ram_bytes / 2, 75, ThermalState::Nominal),
            policy,
        );

        adaptive.register(CpuReferenceMobileProvider);
        adaptive.register(CpuTiledProvider::default());

        if device.supports_vulkan {
            adaptive.register(vulkan_provider(
                CpuReferenceProvider,
                "test-vulkan-adapter",
                64 * 1024 * 1024,
                all_ops(),
            ));
        }

        if device.supports_npu {
            adaptive.register(npu_provider(
                CpuReferenceProvider,
                "test-npu-adapter",
                64 * 1024 * 1024,
                all_ops(),
            ));
        }

        let mut autotune = AutotuneTable::default();
        autotune.record(ProviderMeasurement {
            kind: ProviderKind::CpuTiled,
            op: TensorOp::MatMul,
            latency_nanos: 300,
            verified_equivalent: true,
        });
        if device.supports_vulkan {
            autotune.record(ProviderMeasurement {
                kind: ProviderKind::Vulkan,
                op: TensorOp::MatMul,
                latency_nanos: 200,
                verified_equivalent: true,
            });
        }
        if device.supports_npu {
            autotune.record(ProviderMeasurement {
                kind: ProviderKind::Npu,
                op: TensorOp::MatMul,
                latency_nanos: 100,
                verified_equivalent: true,
            });
        }
        adaptive.set_autotune(autotune);

        let output = GraphExecutor::new(adaptive)
            .execute(&matmul_graph(), inputs())
            .expect("adaptive graph");

        assert_eq!(output, expected, "device profile {index}");
    }
}

#[test]
fn hot_pressure_forces_cpu_fallback_without_output_drift() {
    const GIB: u64 = 1024 * 1024 * 1024;
    let device = DeviceCapabilities {
        logical_cores: 8,
        total_ram_bytes: 8 * GIB,
        supports_vulkan: true,
        supports_npu: true,
    };
    let policy = ntd_runtime::ComputePolicy::for_device(&device);
    let mut adaptive = AdaptiveExecutionProvider::new(
        device,
        snapshot(3 * GIB, 80, ThermalState::Hot),
        policy,
    );

    adaptive.register(vulkan_provider(
        CpuReferenceProvider,
        "test-vulkan-adapter",
        64 * 1024 * 1024,
        all_ops(),
    ));
    adaptive.register(npu_provider(
        CpuReferenceProvider,
        "test-npu-adapter",
        64 * 1024 * 1024,
        all_ops(),
    ));
    adaptive.register(CpuTiledProvider::default());
    adaptive.register(CpuReferenceMobileProvider);

    let mut autotune = AutotuneTable::default();
    autotune.record(ProviderMeasurement {
        kind: ProviderKind::Vulkan,
        op: TensorOp::MatMul,
        latency_nanos: 20,
        verified_equivalent: true,
    });
    autotune.record(ProviderMeasurement {
        kind: ProviderKind::Npu,
        op: TensorOp::MatMul,
        latency_nanos: 10,
        verified_equivalent: true,
    });
    adaptive.set_autotune(autotune);

    assert_eq!(
        adaptive
            .selected_provider(TensorOp::MatMul)
            .expect("selected")
            .kind,
        ProviderKind::CpuTiled
    );

    let adaptive_output = GraphExecutor::new(adaptive)
        .execute(&matmul_graph(), inputs())
        .expect("adaptive");
    assert_eq!(adaptive_output, reference_output());
}

#[test]
fn runtime_detects_at_least_one_host_cpu_thread() {
    let detected = DeviceCapabilities::detect(4 * 1024 * 1024 * 1024, false, false);
    assert!(detected.logical_cores >= 1);
}
