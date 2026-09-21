# NTD97 IR v0 Contract

Status: IR version 0.4 contract.

NTD97 IR is the native semantic boundary between assimilated intelligence and the sovereign execution runtime.

## 1. Versioning

Current contract:

```text
major = 0
minor = 4
```

Compatibility rule:

- same major version is required;
- a reader may accept an equal or older minor version;
- an unknown major version is rejected before execution.

IR 0.4 is an additive extension of 0.3. It adds dynamic `PositionIds`, model-supplied RMSNorm epsilon, and dynamic reshape semantics (`0` copies the corresponding source dimension and one `-1` may infer the remaining dimension). IR 0.3 already added `Silu`, `Reshape`, `Transpose`, learned RMSNorm scale and grouped-query causal attention. Existing 0.1–0.3 graphs remain readable.

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
- Transpose;
- PositionIds.

These are semantic operation identities, not optimized kernels.

Reference generative semantics in IR 0.4:

- `Gather(table, indices)` selects rows from a rank-2 table using a rank-1 integer-token sequence and produces `[sequence, width]`.
- `RotaryPosition(values, positions)` applies canonical rotary pair rotation to rank-3 `[sequence, heads, width]` values with an explicit rank-1 position sequence.
- `CausalAttention(query, key, value)` consumes rank-3 tensors with equal sequence/head-width; K/V share their head count and Q heads may be an integer multiple of KV heads for grouped-query attention. It applies scaled dot-product attention with a causal prefix mask and returns the Q shape.
- `RmsNorm(values[, weight[, epsilon]])` preserves older forms, optionally applies a learned rank-1 scale, and accepts a positive scalar epsilon from the lowered model.
- `Silu(values)` applies the SiLU activation elementwise.
- `Reshape(values, shape)` changes tensor shape without changing element order; positive dimensions are explicit, `0` copies the corresponding source dimension, and one `-1` may infer the remaining dimension.
- `PositionIds(tokens)` deterministically emits `[0, 1, ..., sequence-1]` for the current rank-1 token window.\n- `Transpose(values, permutation)` applies an explicit rank permutation and rejects duplicate/out-of-range axes.

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
