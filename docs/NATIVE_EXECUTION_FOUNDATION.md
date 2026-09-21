# Native Intelligence Execution Foundation

Status: canonical major implementation block following NCC97 Native IR Serialization.

This block establishes the first complete native execution path for NTD97. It extends the frozen architecture; it does not add a parallel model backend.

## 1. End-to-end path

The implemented path is:

```text
NCC97 capsule
  -> capsule compatibility + integrity verification
  -> NIR97 graph decode + structural validation
  -> native tensor descriptor table
  -> embedded/content-addressed NTP97 tensor shard resolution
  -> descriptor/binding/integrity validation
  -> tensor materialization + deterministic dequantization
  -> NTD97 IR graph executor
  -> CPU reference execution provider
  -> deterministic observable tensor result
```

GGUF, ONNX, TFLite, SafeTensors, hosted AI services and source-model runtimes are not part of this path.

## 2. NTP97 native tensor shard

Native tensor payloads use the `NTP97\0` framing.

Each shard records:

- tensor ID;
- optional graph-input `ValueId` binding;
- dtype;
- shape/rank;
- element count;
- payload length;
- quantization metadata;
- SHA-256 binding to its canonical tensor descriptor;
- native payload bytes.

Encoding is deterministic and host-ABI independent.

The descriptor table remains the canonical `NTS97\0` contract introduced earlier. Loading requires each descriptor and shard to agree exactly.

## 3. Content-addressed store

The runtime-facing capsule loader resolves external NCC97 chunks by their existing SHA-256 content identity.

External content is accepted only when:

- the requested digest exists;
- its logical byte length matches the NCC97 index;
- recomputed SHA-256 matches the indexed digest.

This allows Thin capsules and deduplicated native intelligence without weakening the Full-capsule integrity model.

The initial in-memory store is a replaceable reference implementation. The `ContentStore` contract can later be backed by mmap files, encrypted local storage or mobile storage APIs without changing NTD97 IR semantics.

## 4. Quantization metadata and tensor materialization

The first native quantization profiles are:

- unquantized native tensors;
- symmetric I8 with positive finite scale;
- affine I8 with positive finite scale and signed I8 zero point.

Quantization metadata is stored as deterministic IEEE-754 bit patterns rather than host-language serialized structs.

The reference tensor loader materializes all current NTD97 IR dtypes into an F32 reference representation. This is deliberately a correctness baseline, not the final mobile memory format or optimized compute path.

## 5. CPU reference provider

The CPU reference provider defines executable correctness behavior for:

- Add;
- Mul;
- MatMul;
- QuantizedMatMul after native dequantization;
- RmsNorm;
- Softmax.

At the Major Block A boundary, `Gather` and `RotaryPosition` were intentionally left unsupported rather than guessed from a source format. Major Block B subsequently defines and implements their canonical generative semantics and adds `CausalAttention` under IR 0.2.

Optimized CPU, Vulkan and NPU providers must preserve observable semantics against the expanded native reference path.

## 6. NTD97 IR graph executor

The graph executor:

- runs the canonical IR structural validator before execution;
- binds typed graph inputs by `ValueId`;
- executes only NTD97 tensor operations through an `ExecutionProvider`;
- checks output arity and declared rank;
- rejects unsupported state/memory/control/tool nodes rather than bypassing their future authority contracts;
- returns only declared graph outputs.

The executor has no dependency on NCC97 or an external model format.

## 7. Dependency boundary

The architecture freeze remains intact:

```text
ntd-ir
  ^
  |
ntd-capsule

ntd-core     ntd-ir
    ^          ^
     \        /
      ntd-runtime
```

`ntd-capsule` uses `ntd-runtime` only as a development dependency for the end-to-end integration test. Production code does not introduce a reverse dependency.

## 8. Deterministic acceptance test

The integration test creates a Thin NCC97 capsule containing:

- a NIR97 graph;
- a canonical tensor descriptor table;
- one external content-addressed quantized weight shard;
- one embedded F32 bias shard.

It then reopens and integrity-verifies the capsule, resolves native tensor content, materializes the tensors, executes `QuantizedMatMul -> Add` through the CPU provider twice, and requires identical output.

This proves the complete native path without invoking any third-party model runtime.

## 9. Scope boundary

This block is an execution foundation, not yet a complete language model or AGI.

It does not yet provide:

- tokenizer-driven autoregressive generation;
- attention/KV-cache semantics;
- optimized/mobile kernels;
- Vulkan/NPU providers;
- source intelligence import/assimilation;
- cognitive planning/reasoning loops;
- sovereign long-term memory;
- tool/device/web execution;
- Android continuity or 3D embodiment.

Those capabilities must build on this native path rather than introducing alternate backends.
