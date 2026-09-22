# Tool, Internet + Device Action Fabric

Status: Major Block E canonical contract.

This block gives the existing NTD97 cognitive task model a typed, governed execution fabric for web, browser, file, device and app capabilities. It does not embed Android, browser-engine, filesystem, network or hosted-AI implementations into the sovereign core.

## 1. Execution boundary

```text
CognitiveTask / TaskGraph
  -> capability registry
  -> typed action payload
  -> authority gate
  -> platform/tool adapter
  -> output
  -> verifier
  -> commit / retry / rollback / fail
  -> resumable TAF97 journal
```

`ntd-runtime` owns the action semantics and state machine. Platform shells and tool integrations implement `CapabilityAdapter` and provide the actual operating-system or network calls.

## 2. Capability registry

`CapabilityRegistry` binds each `CapabilityId` to a versioned `CapabilityDescriptor` containing:

- capability version;
- domain;
- side-effect class;
- verification requirement;
- rollback support;
- resumability;
- required authority scopes.

Action plans pin the descriptor version. A TAF97 checkpoint cannot silently resume against a different version of the same capability ID.

Domains are:

- Web;
- Browser;
- File;
- Device;
- App;
- Custom.

## 3. Typed actions

`TypedAction` defines source-independent runtime requests for:

- web search;
- web fetch;
- browser observation;
- browser interaction;
- file read;
- file write;
- device observation;
- device interaction;
- app action;
- custom capability payloads.

The registry rejects domain mismatches before adapter execution.

These are portable action contracts, not implementations of HTTP, WebView, Android Accessibility, Storage Access Framework or another platform API.

## 3.1 Production WebSearch normalization

The Android WebSearch platform boundary remains provider-independent. NTD97 does not ship a search-provider SDK, hostname or provider-specific JSON schema.

A user configuration supplies:

- a public-HTTPS endpoint template containing `{query}` and optionally `{count}`;
- a dot-separated path to the result array;
- relative title and URL paths;
- an optional snippet path.

The Android boundary fetches the provider JSON through the same hardened public-HTTPS transport, maps it to bounded canonical `title / URL / snippet` items, drops unsafe/non-public result URLs and deduplicates URLs before data crosses into native cognition.

The native bridge receives a binary normalized-search response rather than raw provider JSON. It materializes an `ActionValue::TextList`, binds the result count and canonical SHA-256 digest into evidence, and the production verifier recomputes both before commit. Endpoint evidence exposes only the final HTTPS source host, not the configured path/query.

Clearing the configuration disables WebSearch immediately and restores fail-closed behavior.

## 4. Authority model

`AuthorityGrant` contains the scopes granted for the current execution attempt plus explicit switches for external writes and irreversible actions.

The gate is deny-by-default:

- all descriptor scopes must be present;
- `ExternalWrite` additionally requires explicit external-write authority;
- `Irreversible` additionally requires explicit irreversible authority.

Authority grants are deliberately **not serialized into TAF97**. After cold reconstruction, the platform/user must provide current authority again before a tool action can run.

## 5. Side effects and verification

The fabric reuses the canonical `SideEffectClass` from `ntd-core`:

- ReadOnly;
- Reversible;
- ExternalWrite;
- Irreversible.

`ActionVerifier` receives the capability descriptor, typed request and adapter output and returns:

- Accept;
- Retry;
- Reject.

Only accepted actions advance the plan cursor.

A rejected action is rolled back when the capability explicitly supports rollback and returns a rollback token. Otherwise the action fails closed.

`rollback_plan` refuses to claim success when a committed side effect cannot actually be undone.

## 6. Adapter contract

`CapabilityAdapter` is the platform/tool boundary. It supports:

- execute;
- optional resume;
- optional rollback.

The same stable `ActionId` is reused for retries and cold resume. Side-effecting adapters must treat `ActionId` as their idempotency key so process death or transport failure cannot create duplicate effects when an operation is retried.

Adapter errors become `Retryable` action state rather than leaving the journal stuck in `Running`.

An adapter that returns `Suspended` for a descriptor that is not marked resumable fails the plan.

## 7. Cognitive-task binding

`ActionFabric::prepare_cognitive_task` consumes the `CognitiveTask` created by Major Block D. The cognitive task continues to own its `Intent`, task ID and `TaskGraph`; the tool fabric does not create a second agent/task identity.

Every `ActionNode` must have exactly one typed payload and its declared side-effect and verification contract must agree with the registered capability.

Plan preparation is transactional: validation completes before any plan/action ID cursor is committed.

## 8. Action journal

Each `PlannedAction` stores:

- stable ActionId;
- TaskGraph node ID;
- capability ID + pinned version;
- side-effect / verification contract;
- typed action payload;
- status;
- attempt count;
- verified output;
- resume token;
- rollback token;
- last error.

Statuses are:

- Prepared;
- Running;
- Suspended;
- Retryable;
- Committed;
- RolledBack;
- Failed.

Plan state tracks the cognitive task ID, current cursor and Ready/Suspended/Completed/Failed/RolledBack status.

## 9. TAF97 continuity format

Action execution state is serialized through deterministic `TAF97\0`.

Current version:

```text
major = 0
minor = 1
```

TAF97 serializes only portable action state. It does not serialize adapters or authority grants.

Decode validates capability IDs, pinned versions, domains, side-effect contracts, verification contracts, plan cursors, action identities and suspended-state resume tokens against the current registry.

## 10. NCC97 State binding

A TAF97 checkpoint can be embedded in the existing NCC97 `ContinuityState` section and bound to a known base root.

The acceptance path is:

```text
CognitiveTask
  -> multi-surface ActionFabric plan
  -> WebSearch suspends
  -> TAF97 checkpoint
  -> NCC97 State / integrity verify
  -> restore ActionFabric
  -> authority re-evaluated
  -> resume same ActionId
  -> FileWrite
  -> DeviceObserve
  -> AppAction
  -> verified Completed plan
```

A separate acceptance test forces verifier rejection of an external write and proves that the adapter rollback path executes.

## 11. Android/device boundary

`DeviceObserve`, `DeviceInteract` and `AppAction` are the portable bridge contracts for future Android shell implementations. The shell may map them to user-permitted Android APIs, accessibility/automation surfaces, intents or app-specific integrations where allowed by the OS and user authorization.

No Android permission, AccessibilityService, Intent, WebView or vendor SDK type enters `ntd-runtime`.

## 12. Scope boundary

This block establishes typed tool semantics, governance, side-effect handling, verification, rollback, retry and action continuity.

It does not yet provide the Android lifecycle scheduler, notification/approval UI, reboot wake path, voice/media surfaces or 3D embodiment. Those belong to Major Block F and must reconstruct this same SIK97 + TAF97 identity/state rather than invent another agent.
