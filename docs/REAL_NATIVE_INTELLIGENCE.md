# M11 Real Native Intelligence

Status: complete.

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

## Acceptance result

M11 is complete for the supported canonical LLaMA path represented by the pinned
`stories260K.gguf` artifact.

Recorded evidence now covers:

- seven fixed source-vs-NTD97 tokenizer differential cases with ID-for-ID equality;
- full 128-step source-vs-NIR97 greedy generation equivalence for the GGUF-declared context;
- normalized source/native text equality at 322 bytes with SHA-256 `594a911ebb2ecfeb608919bf157887e82d0090507fa187d45b2b7e23e5e8f583`;
- streamed conversion into 52 NTP97 shards;
- Ed25519-signed Thin NCC97 package verification;
- canonical native asset-store commit and lazy file-backed activation;
- Android debug evidence packaging with no GGUF/source checkpoint/source runtime in the device execution path;
- Android x86_64 emulator verification of the signed package and all external shards;
- 16-token greedy Android generation matching the host-native reference exactly:
  `403,407,261,378,432,383,286,261,376,298,315,421,395,317,426,338`;
- Android generated text: 57 bytes, SHA-256 `1e454937e49b36d9d8cba4a122bd493f1348a90626a41d6061789374fcc021c8`;
- existing Android lifecycle/reboot validation remaining green after the real-model probe.

The Android evidence package capsule SHA-256 is
`8938b069b9891c2bea96f206419bad9990a9a1738a423e4116fe80472d2bda71`.

Canonical GPT-2 execution remains implemented, but representative real GPT-2 differential
evidence is separate support expansion rather than an M11 exit blocker. Non-canonical
BPE pre-tokenizers and unsupported architectures remain fail-closed.

The generic conversion plan remains conservative for arbitrary source artifacts even
when structural parsing succeeds; M11 completion does not generalize the pinned
`stories260K` equivalence result to unrelated models.

## Completion gate

M11 completion requires a representative real model to follow the complete path:

```text
real GGUF
  -> native conversion
  -> signed NCC97
  -> source file/runtime no longer required
  -> local NTD97 generation
  -> deterministic/equivalence evidence
  -> Android load + generation
```

The pinned `stories260K` path now satisfies this gate end-to-end. Parsing a model, listing its tensors, or producing a capsule that does not preserve its executable semantics remains insufficient for any future model-family support claim.
