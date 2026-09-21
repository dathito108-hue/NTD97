# NTD97 Cognitive Capsule — .ncc97

Status: format contract v0.1 draft.

## Purpose

The Cognitive Capsule is NTD97's native portable intelligence container.

It is intended to replace "one model file equals one intelligence" with a richer representation that can be loaded on constrained devices and exported for backup.

## Design goals

- streamable;
- memory-map friendly;
- chunk-addressable;
- resumable;
- integrity checked;
- device-profile aware;
- extensible without breaking old readers;
- able to embed or reference large immutable chunks;
- able to separate model weights from mutable memory;
- deterministic manifest representation.

## Logical layout

```text
NCC97 Header
Manifest
Chunk Index
  |- GRAPH
  |- TENSORS
  |- TOKENIZER
  |- CODECS
  |- ROUTER
  |- ADAPTERS
  |- CAPABILITIES
  |- MEMORY_SCHEMA
  |- MEMORY_STATE (optional)
  |- DEVICE_PROFILES
  |- PROVENANCE
  |- SIGNATURES
Chunk Payloads / External Chunk References
```

## Required header fields

- magic: `NCC97\0`;
- format major/minor;
- manifest offset/length;
- chunk index offset/length;
- feature flags;
- capsule identifier;
- root content hash.

## Tensor storage

Tensor payloads are stored as independent chunks.

A tensor descriptor records:

- logical name;
- shape;
- dtype/quantization family;
- block layout;
- byte order;
- chunk references;
- optional calibration metadata;
- execution hints.

The format does not mandate one quantization algorithm. Runtime providers advertise supported layouts and conversion paths.

## Import adapters

Phase targets include import from:

- GGUF;
- SafeTensors;
- ONNX;
- TFLite.

Import means parsing and normalizing supported content into NTD97 graph/tensor/tokenizer abstractions. Unsupported operators or metadata must fail explicitly rather than silently degrading semantics.

## Export / backup modes

### Thin capsule

Contains manifests, mutable state, adapters and references to immutable tensor chunks already stored in a content-addressed vault.

### Full capsule

Contains all required chunks and can restore the intelligence on another compatible NTD97 installation.

### State capsule

Contains only mutable memory, learned adapters, capability packages and configuration bound to a known base capsule hash.

## Integrity and provenance

Every chunk has a cryptographic content hash.

The manifest records:

- source format;
- importer version;
- licenses/provenance where available;
- transformation history;
- quantization/conversion history;
- signer metadata where used.

This is required for safe capability acquisition and reproducible restore.

## Versioning rule

Readers must reject incompatible major versions.

Unknown optional sections are skipped through length-prefixed descriptors. Unknown required features cause a clear compatibility error.
