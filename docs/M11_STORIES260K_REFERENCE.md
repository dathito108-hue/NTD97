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
- training context in the pinned artifact: 128

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

The conversion plan must still remain activation-blocked until source-vs-NIR97
semantic equivalence is recorded. Structural compatibility is not numerical proof.

## Next evidence gate

The next M11 gate for this exact SHA is:

1. source tokenizer IDs for fixed prompts;
2. source logits for fixed token prefixes;
3. NTD97 tokenizer IDs for the same prompts;
4. NIR97 logits for the same prefixes;
5. declared absolute/relative tolerances and max observed error;
6. signed Thin NCC97 generation from the converted package;
7. Android load/generation using that same signed package.

The source runtime may be used only to produce validation evidence. It must not become
part of the canonical NTD97 runtime or shipped Android dependency.
