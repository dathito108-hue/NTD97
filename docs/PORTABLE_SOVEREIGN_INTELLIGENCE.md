# Portable Sovereign Intelligence

Major Block I makes NTD97-native intelligence portable independently of the app installation while preserving user ownership and offline recovery.

## Invariants

- NTD97 remains the owner of its model intelligence, capability assets, memory and continuity state.
- Backup encryption is performed with a user-owned key; no hosted service or account is required.
- Restored assets remain canonical NCC97 capsules.
- Restore is staged and atomic from the destination store's perspective.
- A wrong key, corrupted object, missing Thin-backup object or invalid capsule fails closed.
- Network access is not required for Full-backup restore.

## Portable asset classes

`PortableAsset` classifies native NCC97 material as:

- **Model** — Full/Thin NCC97 intelligence/model assets;
- **Capability** — Full/Thin native capability assets;
- **Memory** — State NCC97 memory assets;
- **State** — State NCC97 identity/continuity/learned-state assets.

Asset identity is stable through `asset_id + version + SHA-256 capsule digest`.

## Sovereign object store

`SovereignObjectStore` stores content-addressed encrypted objects.

For each object:

1. SHA-256 of plaintext becomes the content digest.
2. logical length and digest are bound as AEAD associated data.
3. a deterministic XChaCha20-Poly1305 nonce is derived from the digest under the NTD97 object-nonce domain.
4. ciphertext is stored under the digest.
5. read/import decrypts and revalidates digest and length before accepting the object.

Identical native content therefore deduplicates to one encrypted object. The deterministic nonce can only repeat for the same content digest, so the same key does not intentionally reuse a nonce across distinct plaintext objects.

## Backup forms

### Full

A Full backup includes:

- encrypted manifest;
- every encrypted object referenced by the manifest.

It is self-contained and can restore to an empty destination store offline.

### Thin

A Thin backup includes:

- encrypted manifest;
- content digests only.

The referenced encrypted objects must already exist in the destination/user-owned object store. Missing objects fail closed.

### State

A State backup selects only:

- Memory assets;
- State assets.

Model and Capability assets are intentionally excluded.

## Manifest and encryption

Portable backups use deterministic PSB97/PSM97 framing.

The encrypted manifest binds:

- backup kind;
- backup ID;
- canonical ordered asset entries;
- asset class;
- asset ID;
- version;
- logical length;
- content digest.

Both manifest and content objects use XChaCha20-Poly1305 with domain-separated associated data.

`BackupKey`:

- is 256 bits;
- never prints key material through `Debug`;
- is zeroized on drop.

## Restore

`restore_backup` restores into a cloned/staged object store first.

Only after all of the following succeed is the destination store replaced:

- embedded object import;
- AEAD authentication;
- manifest decryption;
- canonical manifest validation;
- object availability;
- object digest/length integrity;
- NCC97 asset validation.

This prevents partial recovery state after a failed restore.

## Acceptance

Major Block I is accepted when:

- Full backup restores model, capability, memory and state into a fresh destination store;
- Thin backup reuses content-addressed encrypted objects without duplication;
- State backup excludes model/capability assets;
- a different user key cannot restore the backup;
- corrupted ciphertext is rejected and does not partially mutate the destination;
- Thin restore fails closed when a referenced object is missing;
- canonical Rust CI passes;
- Android APK/lifecycle regression remains green.

No third-party AI backend, cloud backup dependency or source-model runtime is introduced by this block.
