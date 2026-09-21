# NTD97 IR v0 Contract

Status: IR version 0.2 contract.

NTD97 IR is the native semantic boundary between assimilated intelligence and the sovereign execution runtime.

## 1. Versioning

Current contract:

```text
major = 0
minor = 2
```

Compatibility rule:

- same major version is required;
- a reader may accept an equal or older minor version;
- an unknown major version is rejected before execution.

IR 0.2 is an additive extension of 0.1. It adds the native `CausalAttention` tensor semantic; existing 0.1 graphs remain readable.

## 2. Graph model

A graph contains a version, declared inputs, topologically ordered nodes and declared outputs. Nodes use stable `NodeId` and `ValueId` identities. A node may reference graph inputs or values defined by earlier nodes.

## 3. Value types

IR v0 defines scalar values, tensors with dtype + rank, bytes and opaque handles.

Current dtypes:

- F32;
- F16;
- BF16;
- I8;
- U8;
- I32;
- I64;
- Bool.

Concrete tensor dimensions remain runtime/capsule data rather than source-format ABI.

## 4. Tensor semantics

Current tensor operation identities:

- Add;
- Mul;
- MatMul;
- QuantizedMatMul;
- RmsNorm;
- Softmax;
- Gather;
- RotaryPosition;
- CausalAttention.

These are semantic operation identities, not optimized kernels.

Reference generative semantics in IR 0.2:

- `Gather(table, indices)` selects rows from a rank-2 table using a rank-1 integer-token sequence and produces `[sequence, width]`.
- `RotaryPosition(values, positions)` applies canonical rotary pair rotation to rank-3 `[sequence, heads, width]` values with an explicit rank-1 position sequence.
- `CausalAttention(query, key, value)` consumes equal rank-3 `[sequence, heads, width]` tensors, applies scaled dot-product attention with a causal prefix mask, and returns the same shape.

Hardware providers may optimize these operations, but their observable result must preserve the same NTD97 semantics.

## 5. State, memory, control and tool families

State:

- Read;
- Write;
- Checkpoint.

Memory:

- Retrieve;
- Store;
- Forget.

Control:

- Select;
- Merge;
- Barrier.

Tool:

- Invoke a named capability;
- Observe;
- Verify.

Tool operations describe typed intent and do not bypass capability authorization or verification.

## 6. Structural validation

The validator rejects unsupported IR versions, duplicate node IDs, duplicate value definitions, inputs that are not yet defined and graph outputs that do not exist.

Provider-specific shape/arity validation is performed again at execution time. Source runtimes cannot supply hidden semantics.

## 7. Design constraint

IR describes what NTD97 executes, not how a source ecosystem encoded it. GGUF, SafeTensors, ONNX, TFLite or another importer must normalize supported content into these semantics or reject it. They never become permanent execution backends.

## 8. Runtime and serialization binding

The runtime rejects incompatible IR before execution. CPU/GPU/NPU providers implement the same semantic graph contract.

NIR97 serialization remains deterministic and subordinate to IR semantics. IR 0.2 assigns the additive tensor opcode for `CausalAttention` while preserving every 0.1 opcode.
