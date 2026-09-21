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

GGUF intake now also supports a safe file-backed `GgufByteSource`. Metadata and tensor tables are parsed through bounded range reads, while tensor payloads are fetched only when the lowering/transcode path consumes that tensor. The compatibility `&[u8]` path remains a wrapper over the same source contract, and the `gguf-intake` CLI no longer loads the complete source file with `fs::read`.

For canonical LLaMA/SPM tokenizers, omitted `add_space_prefix`, `add_bos_token` and `add_eos_token` keys are resolved using the source-runtime defaults `true / true / false`. Explicit GGUF metadata always overrides those defaults. The resolved values and whether defaults were inherited are surfaced by `gguf-intake` for auditability. Other tokenizer families do not inherit these LLaMA/SPM defaults.

## Conversion planning

`GgufConversionPlan` classifies F32/F16/BF16 tensors as directly representable and Q4_0/Q8_0/Q4_K/Q5_K/Q6_K as supported native transcodes. These quantized formats are decoded clean-room into NTD97-owned F32 payloads first so semantic correctness is established before mobile requantization; every other GGML tensor encoding remains fail-closed until an explicit decoder exists.

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
- an NCC97 generative manifest carrying the token-input binding, distribution-output index and vocabulary size required to bootstrap generation from the package alone;
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

The synthetic fixture proves the native conversion plumbing and graph execution contract.

The pinned real `stories260K.gguf` reference now goes further: a branch-only evidence gate
verifies the exact GGUF SHA, lowers it to streamed NTP97 shards, packages and activates a
signed Thin NCC97 asset, compares seven source tokenizer traces against the native LLaMA
SPM tokenizer, and greedily generates the full 128-token GGUF context through lazy native
tensor resolution. The normalized source and NTD97 outputs are both 322 bytes with
SHA-256 `594a911ebb2ecfeb608919bf157887e82d0090507fa187d45b2b7e23e5e8f583`.
The source runtime exists only inside the validation job and is not a production dependency.

## Remaining activation blockers

M11 deliberately remains in progress because:

- LLaMA-style SentencePiece metadata lowers into NCC97 tokenizer v0.2 and executes natively with score-ordered BPE merges, U+2581 space normalization, byte fallback, explicit policy overrides and canonical source defaults; the pinned real `stories260K` tokenizer now matches the source tokenizer ID-for-ID over the fixed differential corpus;
- canonical GPT-2 (`tokenizer.ggml.pre="gpt-2"`) lowers into NCC97 tokenizer v0.3 and executes natively with Unicode-category pre-tokenization, GPT-2 byte-to-Unicode mapping, ranked BPE merges and source BOS/EOS policy; representative real GPT-2 differential evidence and non-canonical BPE pre-tokenizers remain separate support work;
- representative real LLaMA source-vs-NIR97 execution is now proven for pinned `stories260K.gguf` over its full declared 128-token context, including signed Thin NCC97 activation and lazy shard execution;
- file-backed intake emits each converted tensor immediately as a content-addressed NTP97 shard; Thin NCC97 packages now sign external tensor references, verify each shard by length/hash before activation, persist through the canonical native asset store, and execute through a lazy file-backed `ValueId` resolver that releases graph values after their last use. Retained model-weight RAM therefore no longer grows with total model size, while peak import/activation memory remains bounded by the largest tensor transcode/encode/resolve operation;
- Android has not yet loaded and generated with the converted real native package.

The generic conversion plan remains conservative for arbitrary source artifacts even when structural parsing/lowering/package verification succeeds. The first pinned real-model target and its recorded equivalence evidence are documented in `docs/M11_STORIES260K_REFERENCE.md`.

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
