# Native Generative Intelligence Runtime

Status: Major Block B canonical contract.

This block turns the generic native execution substrate into the first NTD97-owned autoregressive text-generation runtime. It extends the frozen architecture and does not introduce a source-model backend.

## 1. Canonical generative path

```text
Full/Thin NCC97
  -> capsule integrity + NIR97 validation
  -> native tokenizer descriptor
  -> native tensor loading/materialization
  -> bounded token context
  -> NTD97 IR graph execution
  -> native distribution output
  -> deterministic/seeded sampler
  -> next token
  -> repeat until EOS or budget
```

The runtime consumes only NTD97-owned state after loading. GGUF, ONNX, TFLite, SafeTensors and hosted model services are not execution backends.

## 2. Tokenizer contract

Native tokenizer state is stored in the canonical Tokenizer NCC97 section using the existing descriptor-frame contract.

Format identifier: `ntd97.tokenizer.vocab.v1`.

The native payload records an ordered byte-token vocabulary plus optional BOS, EOS and unknown token IDs. Duplicate/empty tokens and out-of-range special IDs are rejected.

The reference `VocabularyTokenizer` performs deterministic longest-prefix tokenization over UTF-8 bytes, excludes special tokens from normal user-text matching and can decode while omitting special tokens.

## 3. Generative tensor semantics

IR 0.2 adds `CausalAttention` and the CPU reference provider now implements all generative tensor primitives required by the native contract:

- Gather;
- RotaryPosition;
- CausalAttention;
- Softmax;
- existing matrix, norm, arithmetic and quantized matrix operations.

`CausalAttention` is the correctness baseline for scaled dot-product causal attention. Optimized mobile providers must preserve the same observable semantics.

## 4. Autoregressive generator

`GraphGenerator` owns the source-independent decode loop. It receives:

- a validated NTD97 IR graph;
- static native tensors already materialized through the provider contract;
- the graph input used for the current token context;
- the declared distribution output;
- vocabulary size and generation policy.

At every step it builds the bounded token window, executes the same NTD97 graph, reads the final vocabulary distribution, samples one token and appends it to the sequence.

EOS and maximum-new-token limits are explicit. Context truncation is left-bounded and deterministic.

## 5. Sampling

The native sampler supports:

- greedy decoding;
- deterministic seeded stochastic decoding;
- temperature;
- optional top-k filtering;
- logits or probability distributions.

Sampling uses no third-party RNG/runtime dependency. Identical seed, step and distribution produce identical selection.

## 6. NTD97-owned cache representation

`KvCache` stores per-layer key/value tensors and supports deterministic sequence append, left truncation and reset. `PrefixCache` records token prefixes and computes the reusable common prefix.

The reference `GraphGenerator` may still recompute the bounded context for correctness. Optimized providers in the mobile-compute block may consume the same NTD97-owned cache state without changing model identity or token semantics.

## 7. Source-independent package acceptance test

The integration test builds a self-contained Full `.ncc97` package containing:

- a NIR97 generation graph;
- a native F32 transition tensor;
- canonical tensor descriptors;
- a native tokenizer descriptor.

The package is reopened through NCC97 integrity verification, converted to runtime-native tensors, and passed to `GraphGenerator`. The golden model deterministically generates:

```text
prompt: a
generated: b c EOS
decoded generated text: bc
```

The generation is executed twice and must be identical.

Although the golden model is intentionally tiny, it exercises the same native package/token/graph/sampler path that larger assimilated NTD97 models must use.

## 8. Import boundary

Source importers remain outside the runtime. A supported importer may parse external model artifacts, but successful assimilation must finish by producing NTD97 IR, native tokenizer state and NCC97 native tensors. After commit, the runtime requires none of the source parser or source runtime.

## 9. Scope boundary

This block establishes native generation correctness. It does not yet provide optimized mobile kernels, provider autotuning, memory mapping, device thermal/battery policy, cognitive planning, sovereign long-term memory, tool execution, Android continuity or 3D embodiment.

Those layers must build on this generative path rather than introducing another model identity.
