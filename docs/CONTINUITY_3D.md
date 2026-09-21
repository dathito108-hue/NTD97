# NTD97 3D Embodiment and Continuous Agent Contract

Status: canonical Phase 002 contract.

## 1. 3D assistant

NTD97 has an interactive 3D character as a first-class assistant surface.

The embodiment is driven by cognitive state but isolated from the reasoning kernel.

### Required 3D channels

- idle/active/thinking/acting states;
- facial expression;
- eye/gaze target;
- lip synchronization;
- gesture/action animation;
- voice output;
- touch/click interaction;
- compact status/progress representation.

### Surfaces

The same embodiment controller may drive:

- full in-app 3D scene;
- floating assistant overlay when the user grants the required OS permission;
- compact bubble or PiP-style surface where appropriate;
- voice/notification fallback when rendering cannot remain visible.

The renderer must adapt FPS, geometry, effects and update frequency to battery and thermal state.

## 2. 24/7 agent semantics

NTD97 is designed to remain **logically alive 24/7**, including after the app UI is closed.

Mobile operating systems can suspend or terminate processes. Therefore the architecture guarantees continuity of committed cognitive/task state, not impossible unrestricted CPU residency.

### Continuity states

```text
INTERACTIVE
ACTIVE_EXECUTION
CHECKPOINTED
SUSPENDED_BY_OS
WAITING_CONDITION
WAITING_APPROVAL
RECONSTRUCTING
VERIFYING_RESUME
COMPLETED
FAILED_RECOVERABLE
```

### Persistent sovereign checkpoint

A checkpoint includes enough state to recreate the same agent mission:

- identity/capsule root;
- goal;
- task graph;
- current action cursor;
- bounded reasoning state reference;
- world-state version;
- memory transaction cursor;
- capability leases;
- pending approval;
- retry/backoff data;
- completion/verification predicates.

No sensitive free-form reasoning transcript is required to resume. The runtime persists structured state needed for correctness.

## 3. Android execution strategy

The Android implementation is layered:

### Visible active work

When a long-running task is legitimately noticeable to the user, NTD97 can use an appropriate foreground execution mode with a persistent notification.

### Deferred/persistent work

Tasks that do not need continuous CPU are checkpointed and scheduled through platform-supported persistent work mechanisms with constraints and retry/backoff.

### Process death

Every externally visible side effect is bracketed by durable execution state so a new process can determine whether an action:

- was never started;
- is in-flight/unknown;
- completed and awaits verification;
- failed and can retry.

### Reboot

A reboot bootstrap reconstructs only durable eligible work, restores the canonical SIK97 graph, verifies capsule/memory integrity, and resumes according to permissions and OS execution rules.

### Doze/battery/thermal pressure

The scheduler lowers wake frequency, reasoning budget, render cost and non-urgent work. Correctness and committed state are preserved.

## 4. Floating assistant limitations

A floating 3D surface is permission- and platform-dependent.

On Android, application overlays require explicit user authorization and other apps can hide overlays on sensitive screens. NTD97 therefore never treats the overlay as the only communication channel.

## 5. Independence

The 24/7 agent does not require:

- a cloud heartbeat;
- hosted inference;
- third-party push messaging;
- a remote scheduler;
- an account login.

Local platform scheduling and persisted NTD97 state are sufficient for the sovereign baseline.

## 6. Acceptance test

The Phase M5 continuity target is demonstrated when:

1. a task is started;
2. NTD97 commits a checkpoint;
3. the UI is closed;
4. the process is terminated;
5. the process later restarts through an eligible platform event;
6. SIK97 reconstructs the same identity and task graph;
7. the checkpoint is verified;
8. execution resumes without duplicating an already-completed side effect.

A second test repeats the sequence across device reboot.
