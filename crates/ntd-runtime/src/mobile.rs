#![forbid(unsafe_code)]

use std::cmp::Ordering;

use ntd_ir::TensorOp;

use crate::{CpuReferenceProvider, ExecutionProvider, Tensor, TensorError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProviderKind {
    CpuReference,
    CpuTiled,
    Vulkan,
    Npu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ThermalState {
    Nominal,
    Warm,
    Hot,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PowerClass {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceCapabilities {
    pub logical_cores: usize,
    pub total_ram_bytes: u64,
    pub supports_vulkan: bool,
    pub supports_npu: bool,
}

impl DeviceCapabilities {
    pub fn detect(
        total_ram_bytes: u64,
        supports_vulkan: bool,
        supports_npu: bool,
    ) -> Self {
        let logical_cores = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);

        Self {
            logical_cores,
            total_ram_bytes,
            supports_vulkan,
            supports_npu,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceSnapshot {
    pub available_ram_bytes: u64,
    pub battery_percent: u8,
    pub charging: bool,
    pub thermal: ThermalState,
    pub latency_budget_ms: u32,
}

impl ResourceSnapshot {
    pub fn normalized(self, device: &DeviceCapabilities) -> Self {
        Self {
            available_ram_bytes: self.available_ram_bytes.min(device.total_ram_bytes),
            battery_percent: self.battery_percent.min(100),
            charging: self.charging,
            thermal: self.thermal,
            latency_budget_ms: self.latency_budget_ms,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationProfile {
    ReferenceF32,
    BalancedF16,
    MobileI8,
    AggressiveI8,
}

impl QuantizationProfile {
    pub fn bytes_per_element(self) -> u64 {
        match self {
            Self::ReferenceF32 => 4,
            Self::BalancedF16 => 2,
            Self::MobileI8 | Self::AggressiveI8 => 1,
        }
    }

    pub fn recommended(device: &DeviceCapabilities) -> Self {
        const GIB: u64 = 1024 * 1024 * 1024;
        match device.total_ram_bytes {
            ram if ram < 6 * GIB => Self::AggressiveI8,
            ram if ram < 10 * GIB => Self::MobileI8,
            ram if ram < 16 * GIB => Self::BalancedF16,
            _ => Self::ReferenceF32,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputePolicy {
    pub reserve_ram_bytes: u64,
    pub page_bytes: u64,
    pub min_battery_for_accelerator: u8,
    pub allow_vulkan: bool,
    pub allow_npu: bool,
    pub prefer_low_power: bool,
    pub quantization: QuantizationProfile,
}

impl ComputePolicy {
    pub fn for_device(device: &DeviceCapabilities) -> Self {
        const MIB: u64 = 1024 * 1024;
        Self {
            reserve_ram_bytes: (device.total_ram_bytes / 8).max(256 * MIB),
            page_bytes: 16 * MIB,
            min_battery_for_accelerator: 15,
            allow_vulkan: true,
            allow_npu: true,
            prefer_low_power: false,
            quantization: QuantizationProfile::recommended(device),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorPlacement {
    ResidentCpu,
    PagedCpu { page_bytes: u64 },
    AcceleratorLocal { provider: ProviderKind },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorPlacementPlan {
    pub quantization: QuantizationProfile,
    pub placement: TensorPlacement,
    pub estimated_bytes: u64,
    pub page_count: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageWindow {
    pub offset: u64,
    pub len: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MobileComputeError {
    Overflow,
    InvalidPageSize,
}

pub fn page_windows(total_bytes: u64, page_bytes: u64) -> Result<Vec<PageWindow>, MobileComputeError> {
    if page_bytes == 0 {
        return Err(MobileComputeError::InvalidPageSize);
    }

    let page_count = if total_bytes == 0 {
        0
    } else {
        total_bytes
            .checked_add(page_bytes - 1)
            .ok_or(MobileComputeError::Overflow)?
            / page_bytes
    };

    let capacity = usize::try_from(page_count).map_err(|_| MobileComputeError::Overflow)?;
    let mut windows = Vec::with_capacity(capacity);
    let mut offset = 0u64;
    while offset < total_bytes {
        let remaining = total_bytes - offset;
        let len = remaining.min(page_bytes);
        windows.push(PageWindow { offset, len });
        offset = offset.checked_add(len).ok_or(MobileComputeError::Overflow)?;
    }
    Ok(windows)
}

pub fn plan_tensor_placement(
    element_count: u64,
    device: &DeviceCapabilities,
    snapshot: ResourceSnapshot,
    policy: ComputePolicy,
) -> Result<TensorPlacementPlan, MobileComputeError> {
    let snapshot = snapshot.normalized(device);
    let estimated_bytes = element_count
        .checked_mul(policy.quantization.bytes_per_element())
        .ok_or(MobileComputeError::Overflow)?;

    let resident_budget = snapshot
        .available_ram_bytes
        .saturating_sub(policy.reserve_ram_bytes);

    let accelerator = if accelerator_permitted(
        ProviderKind::Npu,
        device,
        snapshot,
        policy,
    ) {
        Some(ProviderKind::Npu)
    } else if accelerator_permitted(
        ProviderKind::Vulkan,
        device,
        snapshot,
        policy,
    ) {
        Some(ProviderKind::Vulkan)
    } else {
        None
    };

    let placement = if estimated_bytes <= resident_budget / 2 {
        if let Some(provider) = accelerator {
            TensorPlacement::AcceleratorLocal { provider }
        } else {
            TensorPlacement::ResidentCpu
        }
    } else {
        TensorPlacement::PagedCpu {
            page_bytes: policy.page_bytes,
        }
    };

    let page_count = match placement {
        TensorPlacement::PagedCpu { page_bytes } => {
            u64::try_from(page_windows(estimated_bytes, page_bytes)?.len())
                .map_err(|_| MobileComputeError::Overflow)?
        }
        TensorPlacement::ResidentCpu | TensorPlacement::AcceleratorLocal { .. } => 1,
    };

    Ok(TensorPlacementPlan {
        quantization: policy.quantization,
        placement,
        estimated_bytes,
        page_count,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderProfile {
    pub kind: ProviderKind,
    pub name: String,
    pub minimum_working_set_bytes: u64,
    pub power: PowerClass,
    pub priority: u16,
    pub supported_ops: Vec<TensorOp>,
}

impl ProviderProfile {
    pub fn supports(&self, op: TensorOp) -> bool {
        self.supported_ops.contains(&op)
    }

    pub fn all_tensor_ops(kind: ProviderKind, name: impl Into<String>) -> Self {
        Self {
            kind,
            name: name.into(),
            minimum_working_set_bytes: 0,
            power: PowerClass::Medium,
            priority: 100,
            supported_ops: vec![
                TensorOp::Add,
                TensorOp::Mul,
                TensorOp::MatMul,
                TensorOp::QuantizedMatMul,
                TensorOp::RmsNorm,
                TensorOp::Softmax,
                TensorOp::Gather,
                TensorOp::RotaryPosition,
                TensorOp::CausalAttention,
            ],
        }
    }
}

pub trait MobileExecutionProvider: ExecutionProvider + Send + Sync {
    fn profile(&self) -> ProviderProfile;
}

#[derive(Debug, Clone, Copy)]
pub struct CpuReferenceMobileProvider;

impl ExecutionProvider for CpuReferenceMobileProvider {
    fn execute(&self, op: TensorOp, inputs: &[&Tensor]) -> Result<Vec<Tensor>, TensorError> {
        CpuReferenceProvider.execute(op, inputs)
    }
}

impl MobileExecutionProvider for CpuReferenceMobileProvider {
    fn profile(&self) -> ProviderProfile {
        let mut profile =
            ProviderProfile::all_tensor_ops(ProviderKind::CpuReference, "cpu-reference");
        profile.power = PowerClass::Low;
        profile.priority = 400;
        profile
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CpuTiledProvider {
    tile: usize,
}

impl CpuTiledProvider {
    pub fn new(tile: usize) -> Self {
        Self { tile: tile.max(1) }
    }

    pub fn tile(&self) -> usize {
        self.tile
    }
}

impl Default for CpuTiledProvider {
    fn default() -> Self {
        Self::new(32)
    }
}

impl ExecutionProvider for CpuTiledProvider {
    fn execute(&self, op: TensorOp, inputs: &[&Tensor]) -> Result<Vec<Tensor>, TensorError> {
        match op {
            TensorOp::MatMul | TensorOp::QuantizedMatMul => {
                Ok(vec![tiled_matmul(inputs, self.tile)?])
            }
            _ => CpuReferenceProvider.execute(op, inputs),
        }
    }
}

impl MobileExecutionProvider for CpuTiledProvider {
    fn profile(&self) -> ProviderProfile {
        let mut profile = ProviderProfile::all_tensor_ops(ProviderKind::CpuTiled, "cpu-tiled");
        profile.power = PowerClass::Medium;
        profile.priority = 300;
        profile
    }
}

pub struct ProfiledProvider<P> {
    backend: P,
    profile: ProviderProfile,
}

impl<P> ProfiledProvider<P> {
    pub fn new(backend: P, profile: ProviderProfile) -> Self {
        Self { backend, profile }
    }
}

impl<P> ExecutionProvider for ProfiledProvider<P>
where
    P: ExecutionProvider,
{
    fn execute(&self, op: TensorOp, inputs: &[&Tensor]) -> Result<Vec<Tensor>, TensorError> {
        self.backend.execute(op, inputs)
    }
}

impl<P> MobileExecutionProvider for ProfiledProvider<P>
where
    P: ExecutionProvider + Send + Sync,
{
    fn profile(&self) -> ProviderProfile {
        self.profile.clone()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderMeasurement {
    pub kind: ProviderKind,
    pub op: TensorOp,
    pub latency_nanos: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AutotuneTable {
    measurements: Vec<ProviderMeasurement>,
}

impl AutotuneTable {
    pub fn record(&mut self, measurement: ProviderMeasurement) {
        if let Some(existing) = self
            .measurements
            .iter_mut()
            .find(|entry| entry.kind == measurement.kind && entry.op == measurement.op)
        {
            *existing = measurement;
        } else {
            self.measurements.push(measurement);
        }
    }

    pub fn latency_nanos(&self, kind: ProviderKind, op: TensorOp) -> Option<u64> {
        self.measurements
            .iter()
            .find(|entry| entry.kind == kind && entry.op == op)
            .map(|entry| entry.latency_nanos)
    }
}

pub struct AdaptiveExecutionProvider {
    device: DeviceCapabilities,
    snapshot: ResourceSnapshot,
    policy: ComputePolicy,
    autotune: AutotuneTable,
    providers: Vec<Box<dyn MobileExecutionProvider>>,
}

impl AdaptiveExecutionProvider {
    pub fn new(
        device: DeviceCapabilities,
        snapshot: ResourceSnapshot,
        policy: ComputePolicy,
    ) -> Self {
        Self {
            device,
            snapshot: snapshot.normalized(&device),
            policy,
            autotune: AutotuneTable::default(),
            providers: Vec::new(),
        }
    }

    pub fn register<P>(&mut self, provider: P)
    where
        P: MobileExecutionProvider + 'static,
    {
        self.providers.push(Box::new(provider));
    }

    pub fn set_snapshot(&mut self, snapshot: ResourceSnapshot) {
        self.snapshot = snapshot.normalized(&self.device);
    }

    pub fn set_autotune(&mut self, table: AutotuneTable) {
        self.autotune = table;
    }

    pub fn selected_provider(&self, op: TensorOp) -> Option<ProviderProfile> {
        self.ordered_provider_indices(op)
            .into_iter()
            .next()
            .map(|index| self.providers[index].profile())
    }

    fn ordered_provider_indices(&self, op: TensorOp) -> Vec<usize> {
        let mut indices = self
            .providers
            .iter()
            .enumerate()
            .filter_map(|(index, provider)| {
                let profile = provider.profile();
                self.profile_allowed(&profile, op).then_some(index)
            })
            .collect::<Vec<_>>();

        indices.sort_by(|left, right| {
            let left_profile = self.providers[*left].profile();
            let right_profile = self.providers[*right].profile();
            compare_provider(
                &left_profile,
                &right_profile,
                op,
                &self.autotune,
                self.policy.prefer_low_power,
            )
        });
        indices
    }

    fn profile_allowed(&self, profile: &ProviderProfile, op: TensorOp) -> bool {
        if !profile.supports(op) {
            return false;
        }

        let working_budget = self
            .snapshot
            .available_ram_bytes
            .saturating_sub(self.policy.reserve_ram_bytes);
        if profile.minimum_working_set_bytes > working_budget {
            return false;
        }

        match profile.kind {
            ProviderKind::CpuReference | ProviderKind::CpuTiled => true,
            ProviderKind::Vulkan | ProviderKind::Npu => accelerator_permitted(
                profile.kind,
                &self.device,
                self.snapshot,
                self.policy,
            ),
        }
    }
}

impl ExecutionProvider for AdaptiveExecutionProvider {
    fn execute(&self, op: TensorOp, inputs: &[&Tensor]) -> Result<Vec<Tensor>, TensorError> {
        let indices = self.ordered_provider_indices(op);
        if indices.is_empty() {
            return Err(TensorError::NoProvider);
        }

        let mut last_error = None;
        for index in indices {
            match self.providers[index].execute(op, inputs) {
                Ok(outputs) => return Ok(outputs),
                Err(error) => last_error = Some(error),
            }
        }

        Err(last_error.unwrap_or(TensorError::NoProvider))
    }
}

fn accelerator_permitted(
    kind: ProviderKind,
    device: &DeviceCapabilities,
    snapshot: ResourceSnapshot,
    policy: ComputePolicy,
) -> bool {
    let supported = match kind {
        ProviderKind::Vulkan => policy.allow_vulkan && device.supports_vulkan,
        ProviderKind::Npu => policy.allow_npu && device.supports_npu,
        ProviderKind::CpuReference | ProviderKind::CpuTiled => true,
    };

    if !supported {
        return false;
    }

    if matches!(kind, ProviderKind::Vulkan | ProviderKind::Npu)
        && !snapshot.charging
        && snapshot.battery_percent < policy.min_battery_for_accelerator
    {
        return false;
    }

    if matches!(kind, ProviderKind::Vulkan | ProviderKind::Npu)
        && snapshot.thermal >= ThermalState::Hot
    {
        return false;
    }

    true
}

fn compare_provider(
    left: &ProviderProfile,
    right: &ProviderProfile,
    op: TensorOp,
    autotune: &AutotuneTable,
    prefer_low_power: bool,
) -> Ordering {
    let left_latency = autotune.latency_nanos(left.kind, op);
    let right_latency = autotune.latency_nanos(right.kind, op);

    match (left_latency, right_latency) {
        (Some(left_ns), Some(right_ns)) => left_ns
            .cmp(&right_ns)
            .then_with(|| left.priority.cmp(&right.priority)),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) if prefer_low_power => left
            .power
            .cmp(&right.power)
            .then_with(|| left.priority.cmp(&right.priority)),
        (None, None) => left.priority.cmp(&right.priority),
    }
}

fn tiled_matmul(inputs: &[&Tensor], tile: usize) -> Result<Tensor, TensorError> {
    if inputs.len() != 2 {
        return Err(TensorError::Arity {
            expected: 2,
            actual: inputs.len(),
        });
    }

    let lhs = inputs[0];
    let rhs = inputs[1];
    if lhs.shape().len() != 2 || rhs.shape().len() != 2 || lhs.shape()[1] != rhs.shape()[0] {
        return Err(TensorError::ShapeMismatch);
    }

    let m = lhs.shape()[0];
    let k = lhs.shape()[1];
    let n = rhs.shape()[1];
    let output_len = m.checked_mul(n).ok_or(TensorError::InvalidShape)?;
    let mut output = vec![0.0f32; output_len];

    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            let mut block = 0usize;
            while block < k {
                let end = block.saturating_add(tile).min(k);
                for inner in block..end {
                    sum += lhs.data()[row * k + inner] * rhs.data()[inner * n + col];
                }
                block = end;
            }
            output[row * n + col] = sum;
        }
    }

    Tensor::new(vec![m, n], output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy)]
    struct FailingProvider;

    impl ExecutionProvider for FailingProvider {
        fn execute(&self, op: TensorOp, _inputs: &[&Tensor]) -> Result<Vec<Tensor>, TensorError> {
            Err(TensorError::UnsupportedOp(op))
        }
    }

    fn device() -> DeviceCapabilities {
        DeviceCapabilities {
            logical_cores: 8,
            total_ram_bytes: 8 * 1024 * 1024 * 1024,
            supports_vulkan: true,
            supports_npu: true,
        }
    }

    fn snapshot() -> ResourceSnapshot {
        ResourceSnapshot {
            available_ram_bytes: 5 * 1024 * 1024 * 1024,
            battery_percent: 80,
            charging: false,
            thermal: ThermalState::Nominal,
            latency_budget_ms: 100,
        }
    }

    fn accelerator_profile(kind: ProviderKind, priority: u16) -> ProviderProfile {
        let mut profile = ProviderProfile::all_tensor_ops(kind, "accelerator");
        profile.priority = priority;
        profile.power = PowerClass::High;
        profile
    }

    #[test]
    fn quantization_profile_scales_with_ram() {
        let mut low = device();
        low.total_ram_bytes = 4 * 1024 * 1024 * 1024;
        let mut high = device();
        high.total_ram_bytes = 16 * 1024 * 1024 * 1024;

        assert_eq!(
            QuantizationProfile::recommended(&low),
            QuantizationProfile::AggressiveI8
        );
        assert_eq!(
            QuantizationProfile::recommended(&high),
            QuantizationProfile::ReferenceF32
        );
    }

    #[test]
    fn large_tensor_is_paged_under_pressure() {
        let device = device();
        let mut snapshot = snapshot();
        snapshot.available_ram_bytes = 512 * 1024 * 1024;
        let policy = ComputePolicy::for_device(&device);

        let plan = plan_tensor_placement(
            900 * 1024 * 1024,
            &device,
            snapshot,
            policy,
        )
        .expect("plan");

        assert!(matches!(plan.placement, TensorPlacement::PagedCpu { .. }));
        assert!(plan.page_count > 1);
    }

    #[test]
    fn hot_device_rejects_accelerators() {
        let device = device();
        let mut snapshot = snapshot();
        snapshot.thermal = ThermalState::Hot;
        let policy = ComputePolicy::for_device(&device);
        let mut adaptive = AdaptiveExecutionProvider::new(device, snapshot, policy);
        adaptive.register(ProfiledProvider::new(
            CpuReferenceProvider,
            accelerator_profile(ProviderKind::Npu, 1),
        ));
        adaptive.register(CpuTiledProvider::default());

        assert_eq!(
            adaptive
                .selected_provider(TensorOp::MatMul)
                .expect("provider")
                .kind,
            ProviderKind::CpuTiled
        );
    }

    #[test]
    fn autotune_selects_fastest_verified_provider() {
        let device = device();
        let snapshot = snapshot();
        let policy = ComputePolicy::for_device(&device);
        let mut adaptive = AdaptiveExecutionProvider::new(device, snapshot, policy);
        adaptive.register(CpuTiledProvider::default());
        adaptive.register(ProfiledProvider::new(
            CpuReferenceProvider,
            accelerator_profile(ProviderKind::Vulkan, 50),
        ));

        let mut table = AutotuneTable::default();
        table.record(ProviderMeasurement {
            kind: ProviderKind::CpuTiled,
            op: TensorOp::MatMul,
            latency_nanos: 500,
        });
        table.record(ProviderMeasurement {
            kind: ProviderKind::Vulkan,
            op: TensorOp::MatMul,
            latency_nanos: 100,
        });
        adaptive.set_autotune(table);

        assert_eq!(
            adaptive
                .selected_provider(TensorOp::MatMul)
                .expect("provider")
                .kind,
            ProviderKind::Vulkan
        );
    }

    #[test]
    fn provider_failure_falls_back_without_changing_semantics() {
        let device = device();
        let snapshot = snapshot();
        let policy = ComputePolicy::for_device(&device);
        let mut adaptive = AdaptiveExecutionProvider::new(device, snapshot, policy);
        adaptive.register(ProfiledProvider::new(
            FailingProvider,
            accelerator_profile(ProviderKind::Npu, 1),
        ));
        adaptive.register(CpuTiledProvider::default());
        adaptive.register(CpuReferenceMobileProvider);

        let lhs = Tensor::new(vec![1, 2], vec![1.0, 2.0]).expect("lhs");
        let rhs = Tensor::new(vec![2, 2], vec![3.0, 4.0, 5.0, 6.0]).expect("rhs");

        let adaptive_output = adaptive
            .execute(TensorOp::MatMul, &[&lhs, &rhs])
            .expect("adaptive");
        let reference_output = CpuReferenceProvider
            .execute(TensorOp::MatMul, &[&lhs, &rhs])
            .expect("reference");

        assert_eq!(adaptive_output, reference_output);
    }

    #[test]
    fn page_windows_cover_exact_length() {
        let windows = page_windows(10, 4).expect("windows");
        assert_eq!(
            windows,
            vec![
                PageWindow { offset: 0, len: 4 },
                PageWindow { offset: 4, len: 4 },
                PageWindow { offset: 8, len: 2 },
            ]
        );
    }
}
