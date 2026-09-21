# NTD97 IR v0 Contract

Status: IR version 0.3 contract.

NTD97 IR is the native semantic boundary between assimilated intelligence and the sovereign execution runtime.

## 1. Versioning

Current contract:

```text
major = 0
minor = 3
```

Compatibility rule:

- same major version is required;
- a reader may accept an equal or older minor version;
- an unknown major version is rejected before execution.

IR 0.3 is an additive extension of 0.2. It adds transformer-lowering semantics for `Silu`, `Reshape` and `Transpose`, extends RMSNorm with an optional learned scale input, and permits grouped-query causal attention. Existing 0.1/0.2 graphs remain readable.

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
- CausalAttention;
- Silu;
- Reshape;
- Transpose.

These are semantic operation identities, not optimized kernels.

Reference generative semantics in IR 0.3:

- `Gather(table, indices)` selects rows from a rank-2 table using a rank-1 integer-token sequence and produces `[sequence, width]`.
- `RotaryPosition(values, positions)` applies canonical rotary pair rotation to rank-3 `[sequence, heads, width]` values with an explicit rank-1 position sequence.
- `CausalAttention(query, key, value)` consumes rank-3 tensors with equal sequence/head-width; K/V share their head count and Q heads may be an integer multiple of KV heads for grouped-query attention. It applies scaled dot-product attention with a causal prefix mask and returns the Q shape.
- `RmsNorm(values[, weight])` preserves the original one-input semantic and optionally applies a learned rank-1 scale over the final dimension.
- `Silu(values)` applies the SiLU activation elementwise.
- `Reshape(values, shape)` changes tensor shape without changing element order; the explicit shape tensor must preserve element count.
- `Transpose(values, permutation)` applies an explicit rank permutation and rejects duplicate/out-of-range axes.

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

NIR97 serialization remains deterministic and subordinate to IR semantics. IR 0.3 preserves every prior tensor opcode and assigns additive opcodes 10–12 to `Silu`, `Reshape` and `Transpose`.
