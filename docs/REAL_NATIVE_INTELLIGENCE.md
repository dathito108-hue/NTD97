# M11 Real Native Intelligence

Status: in progress.

M11 converts the existing native execution substrate into a path capable of accepting real external model intelligence and ending in source-runtime-independent NTD97 assets.

## Non-negotiable invariant

External model formats are import sources only:

```text
GGUF/source
  -> bounded parse
  -> architecture + tokenizer + tensor inspection
  -> explicit conversion plan
  -> NTD97 IR lowering
  -> native tensor transcode/sharding
  -> semantic-equivalence validation
  -> signed NCC97 package
  -> atomic activation
```

The canonical runtime must never execute the GGUF file through llama.cpp or another permanent source-model backend.

## GGUF v3 intake

The clean-room Rust intake currently reads:

- GGUF magic and v3 header;
- tensor and metadata counts with hard bounds;
- all standard scalar metadata types;
- homogeneous arrays;
- strings;
- `general.alignment`;
- tensor names, dimensions, GGML type and data offsets;
- `general.architecture`;
- model name when present;
- `tokenizer.ggml.model`;
- `tokenizer.ggml.tokens`;
- BOS/EOS/unknown token IDs.

The reader rejects malformed headers, future versions, duplicate metadata/tensor names, nested arrays, invalid booleans, invalid UTF-8 metadata, invalid tensor rank/dimensions, non-power-of-two alignment, misaligned tensor offsets and out-of-range special-token IDs.

## Conversion planning

`GgufConversionPlan` classifies F32/F16 tensors as directly representable by the current NTD97 tensor payload and marks other GGML tensor encodings for native transcode.

A parsed model is **not activation-ready** merely because its file structure is valid. The plan retains a hard semantic-equivalence blocker until architecture lowering and native execution have been verified.

Use:

```bash
cargo run -p ntd-assimilation --bin gguf-intake -- /path/to/model.gguf
```

The command exits nonzero while blockers remain.

## NTD97 IR 0.3 transformer semantics

M11 adds the minimum generic semantics required to lower modern decoder-only transformer graphs without hiding source-runtime behavior inside adapters:

- `Silu`;
- `Reshape`;
- `Transpose`;
- learned-scale RMSNorm;
- grouped-query causal attention where query-head count is an integer multiple of KV-head count.

These remain provider-neutral NTD97 operations. Mobile providers must either execute them equivalently or be rejected by the existing provider-verification/autotune boundary.

## Completion gate

M11 is not complete until a representative real model follows the complete path:

```text
real GGUF
  -> native conversion
  -> signed NCC97
  -> source file/runtime no longer required
  -> local NTD97 generation
  -> deterministic/equivalence evidence
  -> Android load + generation
```

Parsing a model, listing its tensors, or producing a capsule that does not preserve its executable semantics is insufficient.
