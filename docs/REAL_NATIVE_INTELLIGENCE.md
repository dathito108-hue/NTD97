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

`GgufConversionPlan` classifies F32/F16/BF16 tensors as directly representable, Q4_0/Q8_0 as supported native transcodes, and every other GGML tensor encoding as unsupported until an explicit decoder exists. Q4_0/Q8_0 are currently decoded into NTD97-owned F32 payloads first so semantic correctness is established before mobile requantization.

A parsed model is **not activation-ready** merely because its file structure is valid. The plan retains a hard semantic-equivalence blocker until architecture lowering and native execution have been verified.

Use:

```bash
cargo run -p ntd-assimilation --bin gguf-intake -- /path/to/model.gguf
```

The command exits nonzero while blockers remain.

## NTD97 IR 0.4 transformer semantics

M11 adds the minimum generic semantics required to lower modern decoder-only transformer graphs without hiding source-runtime behavior inside adapters:

- `Silu`;
- `Reshape`;
- `Transpose`;
- dynamic `PositionIds`;
- dynamic reshape dimensions using `0`/one `-1`;
- learned-scale RMSNorm with model-supplied epsilon;
- grouped-query causal attention where query-head count is an integer multiple of KV-head count.

These remain provider-neutral NTD97 operations. Mobile providers must either execute them equivalently or be rejected by the existing provider-verification/autotune boundary.

## Canonical LLaMA lowering and native package

The current M11 path now lowers a strict LLaMA-family subset into NIR97. It requires canonical configuration metadata and canonical tensor roles, validates every required tensor shape, rejects unsupported RoPE scaling/custom frequency semantics, and rejects unconsumed model tensors rather than silently dropping source behavior.

The lowering path emits:

- a full-context NIR97 transformer graph;
- native NTP97 tensor shards and a canonical tensor descriptor table;
- an NCC97 tokenizer section;
- a forge-native intelligence candidate;
- an Ed25519-signed NCC97 package that is reloaded through the canonical native generative loader before verification succeeds.

A deterministic tiny-LLaMA GGUF fixture exercises:

```text
GGUF v3
  -> parse
  -> tensor transcode
  -> LLaMA lowering
  -> GraphGenerator execution
  -> NativeCandidate
  -> isolated regression sandbox
  -> signed NCC97
  -> native generative loader
```

This proves the native conversion plumbing and graph execution contract. It does **not** yet prove source-equivalent text behavior for a real model.

## Remaining activation blockers

M11 deliberately remains in progress because:

- GGUF tokenizer source metadata is now preserved for native lowering, including SentencePiece scores/token types, GPT-2 merge ranks, pre-tokenizer identity and add-BOS/add-EOS flags; however, the current NCC97/runtime vocabulary tokenizer is not yet a source-equivalent SentencePiece/GPT-2 BPE implementation;
- a representative real GGUF has not yet passed source-vs-NIR97 semantic-equivalence testing;
- common K-quant families such as Q4_K/Q5_K/Q6_K are not yet decoded;
- large-file import still needs a streaming/mapped path rather than whole-file memory loading;
- Android has not yet loaded and generated with the converted real native package.

The conversion plan therefore keeps activation blocked even when structural parsing/lowering/package verification succeeds.

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
