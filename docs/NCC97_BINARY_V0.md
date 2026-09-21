# NCC97 Binary Capsule v0.1

Status: Phase 003B canonical wire-format contract.

NCC97 is the native durable intelligence container for NTD97.

This phase defines a deterministic binary representation with no external serialization/runtime dependency.

## 1. Encoding

All integer fields use little-endian encoding.

The file layout is:

```text
+----------------------+ 0
| Fixed Header         | 112 bytes
+----------------------+ 112
| Fixed Manifest       | 56 bytes
+----------------------+
| Chunk Index          | 56 bytes per entry
+----------------------+
| alignment padding    | to 8 bytes
+----------------------+
| Embedded payloads    | each starts at 8-byte aligned offset
+----------------------+
```

There is no required trailing padding after the final payload.

## 2. Header

The v0.1 header is exactly 112 bytes.

Fields:

- magic: `NCC97\0`;
- header length;
- NCC97 major/minor;
- NTD97 IR major/minor;
- reserved feature flags;
- manifest offset/length;
- chunk-index offset/length;
- payload-area offset;
- chunk count;
- reserved word;
- SHA-256 metadata root;
- reserved tail bytes.

Unknown non-zero reserved fields are rejected in v0.1.

The reader validates the NCC97/IR compatibility contract before exposing chunks.

## 3. Manifest

The v0.1 manifest is exactly 56 bytes.

It stores:

- capsule kind;
- manifest flags;
- 16-byte capsule identity;
- optional 32-byte base capsule root.

Capsule kinds:

- `Full`: self-contained native capsule; external chunks are forbidden;
- `Thin`: may reference immutable chunks from a content-addressed vault;
- `State`: mutable/state-oriented capsule, optionally linked to a base root.

## 4. Chunk index

Each chunk-index entry is exactly 56 bytes.

Fields:

- section kind;
- flags;
- reserved word;
- logical byte length;
- embedded absolute file offset, or zero for an external chunk;
- SHA-256 content hash.

The hash is also the content identity for the chunk in this phase.

External chunk references carry hash + logical length but no embedded bytes.

## 5. Integrity model

For every embedded chunk:

```text
chunk_hash = SHA256(payload_bytes)
```

The header metadata root is:

```text
metadata_root = SHA256(manifest_bytes || chunk_index_bytes)
```

Reading a capsule verifies:

1. magic and fixed-layout constraints;
2. NCC97 + NTD97 IR compatibility;
3. manifest/index ranges;
4. metadata root;
5. every embedded chunk payload hash;
6. capsule-kind rules such as Full capsules forbidding external chunks.

An external chunk cannot be payload-verified until the content-addressed store resolves it. Its expected SHA-256 hash remains part of the capsule metadata.

## 6. Determinism

Given the same:

- capsule kind;
- capsule ID;
- base root;
- ordered chunk set;
- chunk bytes/hashes;

the writer must produce byte-identical NCC97 output.

Wall-clock timestamps, random padding and host-dependent serialization are not part of v0.1.

## 7. Streaming / mmap contract

The reader operates on an immutable byte slice and returns embedded payloads as borrowed slices.

This is intentionally compatible with a future memory-mapped file implementation: the binary semantics do not require copying embedded tensor payloads into new buffers.

The current Phase 003B implementation does not add an OS-specific mmap dependency.

## 8. Scope boundary

Phase 003B does not:

- serialize the complete NTD97 IR graph itself;
- define tensor layouts/quantization payloads;
- import GGUF/SafeTensors/ONNX/TFLite;
- execute any graph;
- resolve external content-addressed chunks;
- perform live intelligence assimilation.

It establishes the durable container those later systems must use.

## 9. Acceptance criteria

Phase 003B passes when:

1. deterministic writer exists;
2. zero-copy-style reader over `&[u8]` exists;
3. Full/Thin/State primitives exist;
4. known SHA-256 vectors pass;
5. full capsule round-trip passes;
6. external thin-capsule reference round-trip passes;
7. state capsule base-root round-trip passes;
8. payload tampering is detected;
9. metadata tampering is detected;
10. truncation is rejected;
11. CI format, clippy and tests pass.
