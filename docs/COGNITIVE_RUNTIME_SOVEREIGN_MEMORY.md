# Cognitive Runtime + Sovereign Memory

Status: Major Block D canonical contract.

This block gives one NTD97 identity a persistent cognitive state machine above the native inference/runtime substrate. It does not introduce another model service, planner backend, or platform-specific lifecycle dependency.

## 1. Cognitive identity

`CognitiveIdentity([u8; 16])` is the stable runtime identity carried by SIK97 cognitive state.

The live state contains:

- logical tick;
- persistent goals;
- active/planned/completed/failed tasks;
- `Intent` + `TaskGraph` from `ntd-core`;
- world facts with revisions;
- episodic, semantic and procedural memory;
- learned delta descriptors/payloads and activation state.

Cold reconstruction restores these fields into the same `CognitiveRuntime`; it does not construct a replacement identity.

## 2. Adaptive reasoning loop

`CognitiveRuntime::advance` uses the existing NTD97 `CognitiveSignals -> ReasoningBudget` policy.

Reference iteration budgets are:

- Reflex: 1 iteration;
- Standard: 2 iterations;
- Deep: 4 iterations;
- Recovery: 3 iterations.

Response length is still independent from reasoning depth.

The same runtime therefore supports fast one-step reactions and bounded multi-step reasoning without switching models.

## 3. Planner / executor / verifier split

The cognitive loop has explicit interfaces:

- `CognitivePlanner` chooses the next directive;
- `CognitiveExecutor` performs a `Work` directive;
- `CognitiveVerifier` accepts, retries or rejects the observation.

Internal directives for recall, memory write, world-state update and completion are committed only after verification.

A verification retry is recorded as episodic memory and does not advance the committed task step. A rejection fails the task and linked goal.

This is the boundary that future native generative planning and M5 tool execution plug into. External AI services are not part of the contract.

## 4. Persistent goals, tasks and world state

Goals and tasks use monotonic IDs and bounded `Intent.max_steps`.

World facts are keyed, revisioned and timestamped by logical cognitive tick. State validation rejects:

- missing goal references;
- empty objectives/keys/namespaces;
- zero or regressed next IDs;
- tasks beyond their declared step budget;
- invalid world revisions.

## 5. Sovereign memory

`SovereignMemory` owns three native memory classes:

- Episodic: events, retries, completions and failures;
- Semantic: durable knowledge/facts;
- Procedural: reusable methods/workflows.

The memory API provides:

- `store` with tags and bounded importance;
- deterministic `retrieve` with text/tag/kind filtering and stable ranking;
- `forget` by native memory ID.

Retrieval updates recall metadata while preserving deterministic ordering for equal scores.

## 6. Learned delta hook

`LearnedDelta` stores a native namespace, generation, payload and active flag.

`DeltaActivationHook` is invoked before a delta becomes active. If activation fails, live state is not marked active.

This is only the runtime hook. Training/assimilation, signing, regression and rollback remain the later Capability Forge / assimilation block.

## 7. SIK97 checkpoint

Mutable cognitive state is serialized through the deterministic `SIK97\0` checkpoint format.

Current version:

```text
major = 0
minor = 1
```

The checkpoint encodes:

- cognitive identity;
- logical tick and monotonic ID cursors;
- goals;
- tasks, intents and task graphs;
- world facts;
- sovereign memory records and recall metadata;
- learned deltas.

Encoding first validates runtime invariants. Decoding rejects incompatible major versions, malformed booleans/enums, non-canonical ordering, duplicate/regressed IDs, invalid UTF-8, trailing bytes and invalid reconstructed state.

NCC97 supplies cryptographic chunk integrity; SIK97 supplies deterministic cognitive-state semantics.

## 8. NCC97 State capsule binding

A State `.ncc97` capsule can embed the SIK97 checkpoint in the existing `ContinuityState` section and bind it to a known base capsule root.

The acceptance test proves:

```text
live CognitiveRuntime
  -> deterministic SIK97 checkpoint
  -> NCC97 State capsule
  -> NCC97 integrity verification
  -> SIK97 decode
  -> same CognitiveIdentity + task + world + memory
```

Memory is currently part of the atomic SIK97 continuity snapshot. Future backup/dedup work may also materialize separate `MemoryState` chunks without changing runtime memory identity.

## 9. Cold-process continuation acceptance

The end-to-end runtime test deliberately stops after a verification retry, serializes the complete state, reconstructs a new `CognitiveRuntime`, then continues the same task under Recovery reasoning.

The restored runtime must preserve:

- identity;
- task ID and verification failure count;
- learned-delta activation;
- recalled memory metadata;
- linked goal;
- world state.

It then finishes the task, marks the goal completed and records completion memory.

## 10. Scope boundary

This block establishes persistent cognition and sovereign memory. It does not yet execute web/browser/device actions, manage Android process scheduling, render the 3D assistant, or perform live intelligence assimilation.

Those later blocks must consume this state machine rather than creating a second agent identity.
