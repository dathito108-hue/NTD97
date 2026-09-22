# Paired PC Fabric — PCF97

Major Block G extends NTD97 with trusted-computer capabilities while preserving the architecture freeze: the phone remains the owner of NTD97 identity, cognition, memory and task state. A paired PC is an execution/observation capability node, not a second intelligence backend.

## Boundaries

- no hosted AI backend or cloud fallback;
- no cognition or sovereign-memory migration to the PC;
- every remote operation remains a typed NTD97 action;
- ActionFabric authority and verification remain the governing phone-side contract;
- remote side effects retain stable ActionId identity across retries.

## Pairing and secure session

The fabric pins Ed25519 peer identities and establishes mutually authenticated X25519 sessions. Directional traffic keys are derived with HKDF and PCF97 frames are protected with ChaCha20-Poly1305. Session sequencing rejects replayed encrypted frames and pairing rejects an identity that does not match the pinned peer record.

## Typed protocol

PCF97 carries canonical messages for:

- remote capability discovery;
- typed action requests;
- typed action results;
- artifact descriptors, pulls and chunks.

Requests include a canonical digest. Results bind request identity, capability/version and request digest, then carry their own result digest. Phone-side verification rejects a result bound to a different request.

## Desktop capability agent

The desktop agent exposes installed handlers through explicit capability descriptors. The current canonical handlers cover:

- system observation;
- governed process execution;
- governed artifact read;
- governed artifact write.

Execution policy constrains allowed programs, working roots and captured output size. Artifact policy constrains allowed roots, write permission and maximum byte size. Requests outside policy fail closed.

## Artifact integrity

Large returned artifacts use deterministic descriptors and chunk transfer. The receiver enforces transfer identity/offset sequencing and validates the final SHA-256 digest before materializing bytes as an NTD97 action result.

## Continuity and idempotency

TAF97 minor version 0.2 serializes the additive PC action variants. MCS97 minor version 0.2 accepts the updated action checkpoint. Stable ActionId request IDs let the desktop agent cache completed requests so a retry/cold continuation does not replay a committed remote side effect.

## Production Android integration

M13 binds the pre-existing PCF97 fabric into the same production Android ActionFabric used by web, files, apps and device actions:

- `pc.observe`, `pc.execute`, `pc.artifact.read` and `pc.artifact.write` remain canonical `TypedAction` variants and preserve stable TAF97 ActionIds;
- a strict app-private pairing profile pins the peer alias, socket address, local signing seed, remote peer id and remote Ed25519 verify key;
- each connection performs the canonical PCF97 handshake with fresh OS CSPRNG entropy and bounded TCP read/write timeouts;
- remote capability discovery must match the expected id, version, side-effect class, verification bit and authority scope before registration;
- the Android verifier accepts PC results only after the bridge marks them as authenticated following PCF97 request/result binding and secure-session verification;
- read-only PC observation/artifact read never grants write authority; process execution and artifact write require the same sovereign ExternalWrite approval/reconfirm path as other M13 writes;
- missing or invalid pairing profiles fail closed and do not fall back to a cloud or mock execution backend.

## Phone-side pairing profile lifecycle

M13 also exposes a user-facing Android provisioning surface for the production PCF97 profile already consumed by ActionFabric:

- Java UI never writes `.pcp97` files directly; provisioning, describe, list and revoke all cross the native bridge and reuse the canonical Rust validator;
- the phone signing seed is generated from the OS CSPRNG inside native code and remains only in app-private storage;
- the UI receives only a public receipt containing the phone peer id and Ed25519 verify key for exchange with the desktop;
- alias replacement is create-only: an existing alias must be explicitly revoked before a new identity can be installed, including under file-race conditions;
- staged profile bytes are synced before commit, committed inside the same app-private directory without replacement, reloaded and revalidated before success is returned;
- list returns only profiles that still pass the canonical loader, while revoke rejects symlink/path escape and syncs the profile directory after removal;
- provisioning a profile does not grant any PC execution authority; `pc.execute` and `pc.artifact.write` remain governed ExternalWrite actions requiring sovereign approval.

## Acceptance

The block is covered by regression tests for:

- mutual authentication and rejection of an unpaired identity;
- encrypted-frame replay rejection;
- canonical request/result encoding;
- result/request mismatch rejection;
- chunked artifact integrity and tamper rejection;
- desktop policy denial for unallowlisted execution and path escape;
- stable ActionId remote-side-effect idempotency;
- TAF97 paired-PC action checkpoint round-trip;
- production TCP handshake codec and authenticated loopback action execution;
- strict Android pairing-profile parsing and pinned-identity validation;
- Android default-authority denial for PC execution and fail-closed missing-profile behavior;
- native provision/describe/list/revoke round-trip without exposing `local_seed`;
- invalid socket address and mismatched pinned identity rejection;
- emulator create/list/public-identity/revoke lifecycle against the real app-private profile path.

Canonical Rust verification and the Android native/APK/lifecycle regression gate both passed before merge.
