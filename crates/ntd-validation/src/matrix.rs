#![forbid(unsafe_code)]

use ntd_ir::TensorOp;
use ntd_runtime::{
    npu_provider, plan_tensor_placement, vulkan_provider, AdaptiveExecutionProvider, AutotuneTable,
    ComputePolicy, CpuReferenceMobileProvider, CpuReferenceProvider, CpuTiledProvider,
    DeviceCapabilities, MobileExecutionProvider, ProviderKind, ProviderMeasurement,
    QuantizationProfile, ResourceSnapshot, TensorPlacementPlan, ThermalState,
};

use crate::ValidationError;

const GIB: u64 = 1024 * 1024 * 1024;
const MIB: u64 = 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepresentativeDevice {
    pub name: String,
    pub capabilities: DeviceCapabilities,
    pub nominal: ResourceSnapshot,
    pub hot: ResourceSnapshot,
}

pub fn representative_device_profiles() -> Vec<RepresentativeDevice> {
    vec![
        RepresentativeDevice {
            name: "mobile-4gb-cpu".into(),
            capabilities: DeviceCapabilities {
                logical_cores: 4,
                total_ram_bytes: 4 * GIB,
                supports_vulkan: false,
                supports_npu: false,
            },
            nominal: ResourceSnapshot {
                available_ram_bytes: 3 * GIB,
                battery_percent: 70,
                charging: false,
                thermal: ThermalState::Nominal,
                latency_budget_ms: 250,
            },
            hot: ResourceSnapshot {
                available_ram_bytes: 2 * GIB,
                battery_percent: 35,
                charging: false,
                thermal: ThermalState::Hot,
                latency_budget_ms: 400,
            },
        },
        RepresentativeDevice {
            name: "mobile-8gb-vulkan".into(),
            capabilities: DeviceCapabilities {
                logical_cores: 8,
                total_ram_bytes: 8 * GIB,
                supports_vulkan: true,
                supports_npu: false,
            },
            nominal: ResourceSnapshot {
                available_ram_bytes: 6 * GIB,
                battery_percent: 75,
                charging: false,
                thermal: ThermalState::Nominal,
                latency_budget_ms: 150,
            },
            hot: ResourceSnapshot {
                available_ram_bytes: 4 * GIB,
                battery_percent: 50,
                charging: false,
                thermal: ThermalState::Hot,
                latency_budget_ms: 300,
            },
        },
        RepresentativeDevice {
            name: "mobile-12gb-npu".into(),
            capabilities: DeviceCapabilities {
                logical_cores: 8,
                total_ram_bytes: 12 * GIB,
                supports_vulkan: true,
                supports_npu: true,
            },
            nominal: ResourceSnapshot {
                available_ram_bytes: 9 * GIB,
                battery_percent: 80,
                charging: false,
                thermal: ThermalState::Nominal,
                latency_budget_ms: 100,
            },
            hot: ResourceSnapshot {
                available_ram_bytes: 6 * GIB,
                battery_percent: 45,
                charging: false,
                thermal: ThermalState::Hot,
                latency_budget_ms: 250,
            },
        },
    ]
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceMatrixReport {
    pub profile: String,
    pub quantization: QuantizationProfile,
    pub placement: TensorPlacementPlan,
    pub nominal_provider: ProviderKind,
    pub hot_provider: ProviderKind,
    pub accelerator_disabled_under_heat: bool,
}

pub fn evaluate_device_profile(
    profile: &RepresentativeDevice,
) -> Result<DeviceMatrixReport, ValidationError> {
    let policy = ComputePolicy::for_device(&profile.capabilities);
    let placement = plan_tensor_placement(
        2 * GIB,
        &profile.capabilities,
        profile.nominal,
        policy,
    )
    .map_err(|error| ValidationError::MobileCompute(format!("{error:?}")))?;

    let mut provider = AdaptiveExecutionProvider::new(
        profile.capabilities.clone(),
        profile.nominal,
        policy,
    );
    provider.register(CpuReferenceMobileProvider);
    provider.register(CpuTiledProvider::default());

    let supported_ops = vec![TensorOp::MatMul];
    if profile.capabilities.supports_vulkan {
        provider.register(vulkan_provider(
            CpuReferenceProvider,
            "validation-vulkan",
            64 * MIB,
            supported_ops.clone(),
        ));
    }
    if profile.capabilities.supports_npu {
        provider.register(npu_provider(
            CpuReferenceProvider,
            "validation-npu",
            64 * MIB,
            supported_ops,
        ));
    }

    let mut autotune = AutotuneTable::default();
    autotune.record(ProviderMeasurement {
        kind: ProviderKind::CpuTiled,
        op: TensorOp::MatMul,
        latency_nanos: 300,
        verified_equivalent: true,
    });
    if profile.capabilities.supports_vulkan {
        autotune.record(ProviderMeasurement {
            kind: ProviderKind::Vulkan,
            op: TensorOp::MatMul,
            latency_nanos: 120,
            verified_equivalent: true,
        });
    }
    if profile.capabilities.supports_npu {
        autotune.record(ProviderMeasurement {
            kind: ProviderKind::Npu,
            op: TensorOp::MatMul,
            latency_nanos: 80,
            verified_equivalent: true,
        });
    }
    provider.set_autotune(autotune);

    let nominal_provider = provider
        .selected_provider(TensorOp::MatMul)
        .ok_or_else(|| ValidationError::MobileCompute("no nominal provider".into()))?
        .kind;
    provider.set_snapshot(profile.hot);
    let hot_provider = provider
        .selected_provider(TensorOp::MatMul)
        .ok_or_else(|| ValidationError::MobileCompute("no hot provider".into()))?
        .kind;
    let accelerator_disabled_under_heat = !matches!(
        hot_provider,
        ProviderKind::Vulkan | ProviderKind::Npu
    );

    Ok(DeviceMatrixReport {
        profile: profile.name.clone(),
        quantization: policy.quantization,
        placement,
        nominal_provider,
        hot_provider,
        accelerator_disabled_under_heat,
    })
}
