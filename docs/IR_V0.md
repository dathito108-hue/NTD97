# NTD97 IR v0 Contract

Status: IR version 0.1 contract.

NTD97 IR is the native semantic boundary between assimilated intelligence and the sovereign execution runtime.

## 1. Versioning

Current contract:

```text
major = 0
minor = 1
```

Compatibility rule:

- same major version is required;
- a reader may accept an equal or older minor version;
- an unknown major version is rejected before execution.

The 0.x family remains experimental, but compatibility is still checked explicitly.

## 2. Graph model

A graph contains:

- version;
- declared graph inputs;
- ordered nodes;
- declared graph outputs.

Each node contains:

- stable `NodeId`;
- operation kind;
- input `ValueId` references;
- newly defined output values and their types.

Phase 003A uses a forward, topologically ordered graph contract. A node may reference graph inputs or values defined by earlier nodes.

## 3. Value types

IR v0 defines:

- scalar values;
- tensor values with dtype + rank;
- bytes;
- opaque handles.

Initial dtypes:

- F32;
- F16;
- BF16;
- I8;
- U8;
- I32;
- I64;
- Bool.

Shape dimensions and tensor layouts are deliberately deferred to the binary/tensor contract in the following phases.

## 4. Operation families

### Tensor

Initial semantic identifiers:

- Add;
- Mul;
- MatMul;
- QuantizedMatMul;
- RmsNorm;
- Softmax;
- Gather;
- RotaryPosition.

These are semantic operation identities, not optimized kernels.

### State

- Read;
- Write;
- Checkpoint.

### Memory

- Retrieve;
- Store;
- Forget.

### Control

- Select;
- Merge;
- Barrier.

### Tool

- Invoke a named capability;
- Observe;
- Verify.

Tool operations describe typed execution intent. They do not bypass capability authorization or verification.

## 5. Structural validation

The current validator rejects:

- unsupported IR version;
- duplicate node IDs;
- duplicate value definitions;
- a node input that is not yet defined;
- a declared graph output that does not exist.

Type inference, shape inference, operator-specific arity rules and side-effect validation are planned extensions. They are not silently assumed in v0.1.

## 6. Design constraint

IR operations describe **what NTD97 means to execute**, not how one source ecosystem encoded the computation.

A GGUF, SafeTensors, ONNX or other importer must normalize into these semantics or reject unsupported content.

This rule is what prevents source formats from becoming permanent architectural backends.

## 7. Runtime contract

The runtime must reject incompatible IR before any operation executes.

Future hardware providers implement IR semantics through native CPU/GPU/NPU kernels while preserving the same observable graph behavior.

## 8. Binary serialization binding

Phase 003C binds this semantic IR to the deterministic `NIR97\0` Graph-section encoding documented in `NCC97_IR_SERIALIZATION_V0.md`.

Serialization is subordinate to IR semantics: the wire format preserves the graph contract and decoded graphs must pass the same structural validator before use.

Tensor execution kernels and source-model importers remain outside the IR contract.
