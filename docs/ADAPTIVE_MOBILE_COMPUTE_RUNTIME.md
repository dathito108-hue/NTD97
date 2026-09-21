# Adaptive Mobile Compute Runtime

Status: Major Block C canonical contract.

This block makes NTD97's existing native graph path resource-aware without changing model identity or allowing platform/vendor APIs to enter the sovereign runtime.

## 1. Architectural boundary

The frozen dependency rule remains unchanged:

```text
platform shell / hardware adapter
        -> reports device + resource telemetry
        -> supplies Vulkan/NPU ExecutionProvider implementation
        -> ntd-runtime adaptive provider router
        -> unchanged NTD97 IR semantics
```

`ntd-runtime` contains no Android, Vulkan SDK, vendor NPU SDK, GGUF, ONNX or hosted-model dependency. A platform implementation may wrap a real accelerator through the exported provider adapters; the runtime selects it only after semantic equivalence has been verified.

## 2. Device and resource model

`DeviceCapabilities` records portable CPU thread count, total RAM and whether the platform reports Vulkan/NPU availability. CPU parallelism can be detected through Rust `available_parallelism`; RAM and accelerator facts are supplied by the platform shell because those APIs remain outside the core.

`ResourceSnapshot` carries:

- currently available RAM;
- battery percentage and charging state;
- thermal state: Nominal, Warm, Hot or Critical;
- latency budget.

`ComputePolicy` converts those facts into reserved-RAM, paging, accelerator, power and quantization policy.

## 3. CPU execution

`CpuReferenceMobileProvider` keeps the correctness baseline.

`CpuTiledProvider` provides a cache-friendlier tiled MatMul/QuantizedMatMul path while delegating all other IR operations to the same canonical CPU semantics. Its observable output is required to match the reference provider.

## 4. Vulkan and NPU boundary

`vulkan_provider(...)` and `npu_provider(...)` wrap replaceable platform implementations behind the existing `ExecutionProvider` interface. The adapter declares supported operations, minimum working set and power/priority metadata.

No accelerator is selected merely because it exists. The runtime also checks:

- device support;
- policy permission;
- RAM working-set fit;
- battery threshold when not charging;
- thermal pressure;
- verified semantic equivalence for the requested operation.

## 5. Verified autotuning and fallback

`AutotuneTable` stores per-operation provider latency measurements plus an explicit `verified_equivalent` bit.

`verify_provider_equivalence` executes a candidate beside the trusted reference path and compares outputs. Accelerator measurements that are not equivalence-verified are ignored for routing.

`AdaptiveExecutionProvider` ranks eligible providers using:

1. verified measured latency and the current latency budget;
2. configured priority;
3. optional low-power preference.

If the selected provider fails at execution time, the same operation is retried on the next eligible provider. If no provider is eligible, execution fails closed with `TensorError::NoProvider`.

## 6. RAM, quantization and placement

`QuantizationProfile` supplies portable device-level recommendations:

- ReferenceF32;
- BalancedF16;
- MobileI8;
- AggressiveI8.

The default profile becomes more compact as total RAM decreases.

`plan_tensor_placement` estimates tensor footprint after the selected quantization policy and chooses:

- accelerator-local placement when safe;
- resident CPU placement;
- paged CPU placement when memory pressure requires it.

This is a placement policy contract, not a source-format quantizer. Imported weights still become native NTD97 tensors before execution.

## 7. Paging / mapped-byte contract

`ByteRegion`, `SliceByteRegion`, `PageWindow` and `PagedByteReader` provide deterministic bounded page access without platform APIs in core.

A platform may back `ByteRegion` with memory-mapped storage or another safe file-backed region. NTD97 core owns page boundaries and execution policy; the shell owns the OS mapping mechanism.

## 8. Representative-device equivalence

The integration suite runs the same NTD97 MatMul graph across simulated mobile profiles:

- 4 GB CPU-only;
- 8 GB with Vulkan capability;
- 12 GB with Vulkan + NPU capability.

All adaptive outputs must equal the CPU reference output. A Hot thermal profile must reject accelerators and fall back to CPU without output drift.

## 9. Invariants preserved

- one NTD97 IR graph and one model identity;
- no source-model runtime backend;
- no accelerator may redefine operation semantics;
- hardware choice is runtime policy, not model architecture;
- provider failure does not silently corrupt output;
- platform-specific APIs remain outside `ntd-runtime`.

## 10. Scope boundary

This block establishes adaptive compute selection, portable CPU optimization, accelerator adapter boundaries, verified autotuning, paging/placement policy and resource-pressure fallback.

Actual Android/Vulkan/vendor-NPU device implementations are platform adapters that plug into this contract; they are not dependencies of the sovereign core. Cognitive planning, long-term memory, tools, Android lifecycle continuity and 3D embodiment remain later major blocks.
