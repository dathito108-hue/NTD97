# Capability Forge + Native Assimilation

Major Block H adds the canonical boundary for turning supported external capabilities or intelligence sources into NTD97-owned native assets.

## Invariants

- NTD97 remains one sovereign intelligence identity.
- An external source may be parsed by an importer, but no source runtime becomes a permanent backend.
- Importers must terminate in a validated `NativeCandidate`.
- Capability candidates become canonical capability descriptors plus native adapter assets.
- Intelligence candidates become NTD97 IR plus supported NCC97-native sections.
- Source provenance and license data are preserved in the native capsule.
- Assimilation commits only after isolated native validation succeeds.
- Committed assets are signed and versioned; rollback activates an older verified native version.
- Source bytes are never stored in the committed native package.

## Pipeline

```text
SourcePackage
  -> license / size / source-digest policy
  -> ImporterRegistry discovery
  -> SourceImporter
  -> NativeCandidate
  -> NativeValidationSandbox
  -> NCC97 sections
  -> Ed25519 signature
  -> NativeAssetStore atomic commit
```

The importer registry maps supported media types to one importer identity and rejects ambiguous registrations.

## Provenance and licensing

Each source carries:

- source URI;
- SHA-256 source digest;
- SPDX-style license identifier;
- license notice;
- attribution.

Forge policy allowlists accepted license identifiers and enforces a maximum source payload size. The source digest is rechecked immediately before assimilation.

## Native candidates

Two candidate classes are supported.

### Capability

A capability candidate contains:

- stable asset ID and monotonic version;
- canonical `CapabilityDescriptor`;
- native adapter format and bytes;
- required regression cases.

The resulting NCC97 package contains `Capabilities`, `Adapters`, `Provenance`, `AssimilationLog` and `Signatures` sections.

### Intelligence

An intelligence candidate contains:

- stable asset ID and monotonic version;
- validated NTD97 IR graph;
- optional supported native NCC97 sections;
- required regression cases.

The resulting package includes the canonical `Graph` section plus provenance, assimilation log and signature sections.

## Sandbox and regression validation

The canonical in-core sandbox does not execute source code. It validates only NTD97-native candidate semantics and reports that no network or external write was used.

Current regression probes include:

- NTD97 IR encode/decode semantic round-trip;
- generated native adapter SHA-256 integrity.

Forge refuses a sandbox report that is not isolated, reports network access/external writes, or omits any required regression.

## Signature and commit

Assimilated native packages are signed with Ed25519. Verification binds the ordered native section kinds, lengths and SHA-256 digests. The asset ID, version and asset kind are also bound through the signed assimilation log.

`NativeAssetStore`:

- accepts only packages signed by its trusted verification key;
- rejects duplicate/non-monotonic versions;
- stages an entire batch before mutating active state;
- switches the active version only after full validation;
- supports rollback only to a previously stored, reverified native version.

## Acceptance

Major Block H is accepted when:

- capability source -> signed native NCC97 -> reloadable capability descriptor passes;
- external intelligence source -> NTD97 IR -> signed native NCC97 -> reloadable graph passes;
- disallowed license is rejected before commit;
- duplicate version does not mutate active state;
- untrusted/tampered package is rejected;
- rollback restores a prior verified native version;
- canonical Rust CI passes;
- Android APK/lifecycle regression remains green.

No third-party AI backend or cloud fallback is introduced by this block.
