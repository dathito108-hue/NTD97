# M11 stories260K Real-Model Reference

This file pins the first representative external model used for M11 real-model validation.

## Source artifact

- repository: `ggml-org/tiny-llamas`
- artifact: `stories260K.gguf`
- GGUF version: 3
- remote size: `1,185,376` bytes
- SHA-256: `047bf46455a544931cff6fef14d7910154c56afbc23ab1c5e56a72e69912c04b`
- architecture: `llama`
- tokenizer model: `llama` / SentencePiece-style BPE
- vocabulary: 512 tokens
- tensors: 48
- tensor encoding: F32
- embedding length: 64
- feed-forward length: 172
- block count: 5
- attention heads: 8
- KV heads: 4
- RoPE dimension count: 8
- RMSNorm epsilon: `1e-5`
- declared GGUF context length: 128
- original llama2.c checkpoint max context: 512

The upstream llama2.c-to-GGUF converter intentionally writes tokenizer tokens, scores,
token types and BOS/EOS/UNK IDs, but does not write
`tokenizer.ggml.add_space_prefix`, `tokenizer.ggml.add_bos_token` or
`tokenizer.ggml.add_eos_token`.

For the upstream LLaMA/SPM tokenizer type, llama.cpp initializes the source semantics to:

```text
add_space_prefix = true
add_bos          = true
add_eos          = false
```

and optional GGUF keys override those values only when present.

NTD97 therefore resolves those exact defaults only for `tokenizer.ggml.model = "llama"`.
Explicit GGUF metadata remains authoritative. Other tokenizer models do not inherit
these defaults.

## Expected NTD97 intake evidence

For the pinned artifact, `gguf-intake` should report the resolved tokenizer policy as:

```text
architecture=llama
model_name=llama
tokenizer=llama
tokenizer_add_space_prefix=true
tokenizer_add_bos_token=true
tokenizer_add_eos_token=false
tokenizer_policy_source=canonical-llama-spm-defaults
vocabulary_size=512
tensor_count=48
direct_tensor_count=48
transcode_tensor_count=0
unsupported_tensor_count=0
```

## Recorded source-equivalence evidence

The pinned artifact now has a reproducible source-equivalence gate.

Source oracle inputs are pinned independently:

- llama2.c repository commit: `350e04fe35433e6d2941dce5a1f53308f87058eb`;
- `stories260K.bin` SHA-256: `b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696`;
- `tok512.bin` SHA-256: `037cb335abb25d1fa9e8ecae30ed2a3a8ace9302862ebcdc05d51a6bbb10c312`.

Tokenizer differential evidence uses seven fixed prompts covering empty input, ordinary
words, punctuation, whitespace-prefix behavior and repeated spaces. The canonical
llama2.c tokenizer trace SHA-256 is
`a0f85845416e94154c7bdad1f98c288949752c0d816983a29b49cf3f9aeb56f6`.
NTD97 produces identical token IDs for every case.

The original source checkpoint has a larger maximum context than the pinned GGUF, so
generation equivalence is compared over the GGUF-declared 128-step context rather than
against the upstream 200-step golden. The source output is 323 bytes including the
terminal presentation newline added by `run.c`; after applying the same final-newline
normalization used by upstream `test_all.py`, both source and NTD97 produce:

```text
generated tokens = 128
text bytes       = 322
text SHA-256     = 594a911ebb2ecfeb608919bf157887e82d0090507fa187d45b2b7e23e5e8f583
```

The NTD97 side of the gate executes:

```text
pinned GGUF
  -> file-backed GGUF parse
  -> streamed LLaMA lowering
  -> 52 native NTP97 shards (48 source tensors + 4 canonical NIR97 constants)
  -> NIR97 graph round trip
  -> signed Thin NCC97 package
  -> package + shard verification
  -> canonical native asset-store commit
  -> Thin activation
  -> lazy file-backed tensor resolution
  -> CPU-reference native generation
  -> source-compatible LLaMA SPM decode
```

Both `stories260k_tokenizer_equivalence=PASS` and
`stories260k_source_equivalence=PASS` are required by the evidence workflow.

This satisfies the representative real LLaMA source-vs-NIR97 equivalence gate for the
pinned SHA. It does not authorize unsupported architectures/tokenizers or silently
generalize equivalence to arbitrary source artifacts.

## Recorded Android integration evidence

The same pinned model is exported by NTD97 as a signed Thin NCC97 package and 52
content-addressed NTP97 shards. The host exporter first verifies the package and records
a deterministic 16-token native reference. Only native artifacts are copied into debug
APK assets; the GGUF file, llama2.c checkpoint and source runtime are excluded from the
Android execution path.

Recorded Android evidence:

```text
capsule SHA-256 = 8938b069b9891c2bea96f206419bad9990a9a1738a423e4116fe80472d2bda71
shards          = 52
signature       = ok
activation      = ok
generated tokens = 16
token IDs       = 403,407,261,378,432,383,286,261,376,298,315,421,395,317,426,338
text bytes      = 57
text SHA-256    = 1e454937e49b36d9d8cba4a122bd493f1348a90626a41d6061789374fcc021c8
android_real_model = PASS
```

The x86_64 API 35 emulator verifies the package signature and shard hashes on-device,
activates the Thin capsule through the file-backed resolver, generates the reference
tokens through the JNI/native path, and then completes the existing lifecycle/reboot
probe successfully.

This closes the M11 Android integration gate for the pinned real model. Other model
families and tokenizer variants retain independent fail-closed support/evidence
requirements.

The source runtime is used only as a validation oracle. It is not linked into NCC97,
the canonical NTD97 runtime, or the shipped Android dependency graph.
