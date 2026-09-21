# NTD97 Canonical Architecture

Status: Phase 003A architecture-frozen sovereign contract.

## 1. Goal

NTD97 is not defined as a chat application with tools attached. The core abstraction is an **intelligence runtime that compiles intent into verified actions**.

The conversational interface is only one interaction surface. The interactive 3D assistant is another first-class surface; neither defines the intelligence itself.

The runtime is designed around five priorities:

1. deep reasoning when the task actually needs it;
2. low time-to-first-action;
3. minimal unnecessary language generation;
4. broad interaction through capability adapters;
5. portable and recoverable intelligence state.

## 2. NTD97 model identity

**NTD97 itself is the model and AGI identity.**

External model files are not peers, selectable backends, or permanent dependencies. They are intelligence sources. A successful import is assimilated into the canonical NTD97 representation:

```text
External intelligence source
        |
Parse + inspect + provenance
        |
Semantic / graph normalization
        |
        NTD97 IR
        |
Compatibility + integrity validation
        |
Atomic assimilation commit
        |
.ncc97 native intelligence state
        |
NTD97 identity / memory / capabilities
```

After a successful assimilation commit, normal NTD97 execution must not require the source model runtime, cloud service, vendor SDK, or source file format parser.

"Immediately becomes NTD97 intelligence" means that a validated import is committed as native NTD97 state in the same ingestion transaction. Unvalidated or unsupported material is rejected or quarantined rather than silently becoming trusted intelligence.

Provenance is retained for auditability, but provenance does not create a runtime dependency.

## 2.1 Sovereign Intelligence Kernel — SIK97

SIK97 is the canonical intelligence kernel. It owns:

- reasoning budget and scheduling;
- world-state and goal-state lifecycle;
- memory coordination;
- task graph execution state;
- capability routing;
- checkpoint/recovery state;
- device-resource budgeting.

NTD97 IR is the kernel-facing intermediate representation for cognition and execution. It is designed so that imported ecosystems do not dictate NTD97's architecture.

## 2.2 Single-runtime rule

There is one canonical cognitive runtime.

Imported weights may come from different ecosystem formats, but after import they are represented through the NTD97 execution interface. The application must not accumulate unrelated permanent model/backend stacks.

Fast and deep behavior are produced through **adaptive compute budgets** inside the same cognitive loop:

- reflex budget: direct intent/action mapping, shallow planning, cached skills;
- standard budget: bounded planning + verification;
- deep budget: iterative decomposition, counter-checking, tool-assisted reasoning;
- recovery budget: alternative strategy after verified failure.

The budget can escalate automatically based on uncertainty, task complexity, tool failure, or verification failure.

## 3. Core subsystems

### 3.1 Interaction Fabric

Normalizes inputs from:

- text and voice;
- screen state and images;
- Android/iOS app surfaces;
- sensors where permission exists;
- paired desktop agents;
- future embodied/robotic adapters.

It outputs a typed observation stream, never raw UI-specific control logic.

### 3.2 Intent Compiler

Transforms user intent into a compact machine-oriented representation:

- objective;
- constraints;
- completion criteria;
- privacy/security scope;
- expected output;
- urgency;
- acceptable resource budget.

The default behavior is not to draft a long answer. If the objective is executable, it creates a task graph.

### 3.3 Adaptive Cognitive Loop

Maintains one reasoning state and chooses a compute budget.

The loop can:

- infer;
- retrieve memory;
- inspect capabilities;
- decompose;
- simulate;
- execute;
- observe;
- verify;
- revise.

User-visible language is produced only when needed for interaction, clarification, consent, or result reporting.

### 3.4 Task Graph IR

All multi-step work is compiled into a typed DAG/state machine with:

- preconditions;
- capability requirements;
- side-effect class;
- rollback information;
- completion predicates;
- checkpoints;
- verification steps.

This creates a stable boundary between intelligence and execution.

### 3.5 Executor + Verifier

The executor performs only typed actions declared by capabilities.

The verifier checks observable success rather than trusting the planner's assumption.

Examples:

- file exists and hash matches;
- web request returned expected schema;
- app UI reached expected state;
- remote command returned expected exit status;
- game/app interaction produced the allowed target state.

### 3.6 Capability Graph

A capability is a versioned contract:

```text
Capability
  id
  version
  input schema
  output schema
  permissions
  side-effect class
  cost model
  availability predicate
  executor
  verifier
  rollback
```

Initial capability families:

- web/browser;
- files and storage;
- Android app interaction;
- notifications/background jobs;
- media;
- shell/terminal where the OS permits it;
- paired PC control;
- external APIs;
- user-installed skills.

Interaction with apps or games must use platform-permitted interfaces and must not rely on bypassing anti-cheat, DRM, authentication, or security controls.

### 3.7 Capability Forge

When no suitable capability exists, NTD97 may enter an acquisition workflow:

1. search existing installed capabilities;
2. search trusted documentation/registries/repositories;
3. inspect license and provenance;
4. synthesize or adapt an implementation;
5. build in an isolated workspace;
6. run contract, security, and regression tests;
7. package as a versioned capability;
8. install only under the required permission scope;
9. preserve rollback information.

Capability growth is therefore **extensible but reversible**. Core runtime replacement is not the normal mechanism for learning a new skill.

### 3.8 Memory Fabric

Memory is separated into:

- working context;
- episodic task history;
- semantic knowledge;
- procedural skills;
- user-approved durable preferences;
- capability state.

Mutable memory is logically separate from immutable model weights so that backups and updates do not require duplicating the full model.

### 3.9 Network Fabric

Internet access is exposed as capabilities:

- HTTP/API access;
- search;
- browser automation;
- downloads/uploads;
- streaming;
- secure paired-device transport.

The planner sees cost, latency, trust and connectivity metadata before selecting a network path.

### 3.10 Paired PC Agent

Desktop interaction uses a mutually authenticated peer agent.

The mobile runtime sends typed tasks rather than arbitrary hidden control messages. The PC returns structured observations, logs, and artifacts.

This supports:

- code/build jobs;
- desktop apps;
- large-model or high-compute assist;
- file exchange;
- browser sessions;
- hardware unavailable on the phone.

### 3.11 Intelligence Assimilation Engine

The assimilation engine accepts supported model, knowledge and skill sources and converts them into canonical NTD97 state.

The pipeline is:

1. identify source type and version;
2. parse structure, tensors/graph/tokenizer/knowledge/skills as applicable;
3. preserve source provenance and license metadata;
4. normalize supported semantics into NTD97 IR;
5. convert or repack native tensor and codec chunks;
6. bind learned adapters, semantic knowledge and procedural skills to NTD97 namespaces;
7. validate deterministic compatibility/integrity checks;
8. atomically commit the new native intelligence state;
9. update capsule root/hash and rollback metadata.

Assimilation never means blindly copying arbitrary executable code into the trusted core.

### 3.12 Embodied 3D Assistant

The 3D assistant is a first-class interaction surface driven by NTD97 state.

It contains separate layers for:

- avatar/mesh and animation;
- facial expression and gaze;
- lip-sync and speech;
- gesture/action state;
- interaction hit targets;
- contextual status;
- render-budget adaptation.

The character can appear:

- inside the NTD97 app as a full 3D scene;
- as a user-authorized floating surface where the mobile OS permits overlays;
- through compact bubble/PiP/notification/voice surfaces when a full overlay is unavailable or inappropriate.

The 3D renderer is deliberately isolated from the cognitive kernel. Rendering can be suspended under thermal/battery pressure without stopping cognition or task continuity.

### 3.13 Continuous Agent Runtime

NTD97 targets **24/7 logical availability**, not the false assumption that a mobile OS will permit one process to consume CPU continuously forever.

The canonical continuity contract is:

```text
ACTIVE
  -> checkpoint
  -> UI closed / process killed / reboot / Doze
  -> persisted sovereign state
  -> eligible platform wake
  -> reconstruct same SIK97 graph
  -> verify checkpoint
  -> resume bounded work
```

The state required to resume includes:

- active goals;
- task graph cursor;
- tool/capability state;
- pending approvals;
- world-state snapshot;
- memory transaction state;
- next eligible wake condition;
- verification requirements.

On Android, implementation may combine user-visible foreground service execution for appropriate active work, persistent scheduled work for deferred tasks, boot/package restart handling, notifications and OS-approved wake mechanisms. NTD97 must remain correct when Android delays or stops execution.

Therefore "works after exiting the app" means the agent's mission and state survive the UI lifecycle; it does not promise unrestricted background CPU in defiance of the operating system.

## 4. Mobile execution substrate

The portable core is implemented in Rust.

Platform shells bind through a narrow C ABI/JNI/Swift FFI boundary.

Execution providers are selected dynamically by NTD97 itself:

1. device accelerator provider if compatible;
2. Vulkan/Metal compute provider;
3. optimized CPU provider;
4. optional paired-PC provider when explicitly allowed.

A remote/cloud provider is never required for the sovereign baseline.

The runtime owns tensor layout, scheduling, KV/state management, paging, quantization metadata, and device profiles.

Target mobile optimizations include:

- mmap/zero-copy loading;
- chunked capsule streaming;
- quantized tensor blocks;
- memory-pressure-aware paging;
- prefix/state reuse;
- speculative internal execution where supported;
- thermal-aware scheduling;
- foreground/background execution policy;
- battery-aware compute budgets.

## 5. Compatibility strategy

"Compatible with current phones" is treated as graceful adaptation rather than identical performance on every device.

The canonical runtime must:

- detect architecture and memory limits;
- select a supported execution provider;
- choose a device profile;
- downgrade context/precision/parallelism safely;
- remain functional through CPU fallback when practical;
- never assume a particular vendor NPU.

Android/ARM64 is the first implementation target. The core architecture remains platform-neutral so iOS and desktop shells can share the same capsule and task representation.

## 6. Concise interaction contract

Default user-facing policy:

- execute when sufficiently specified;
- ask only when a missing fact blocks safe execution;
- report the outcome in the shortest useful form;
- expand reasoning/explanation only when requested.

Internally, reasoning depth is independent of response length.

## 7. Sovereign invariants

These additional invariants define the dependency boundary:

- NTD97 is one independent model/AGI identity;
- no third-party AI API, hosted inference endpoint or cloud control plane is required for core cognition;
- no external model remains a permanent runtime backend after successful assimilation;
- imported intelligence becomes native NTD97 state only after validation and atomic commit;
- NTD97 can boot offline, load native intelligence, restore memory and continue local tasks;
- internet is a knowledge/tool sensor, not the brain;
- paired PCs are optional execution bodies, not owners of identity or memory;
- the 3D avatar is an interface and may be throttled without losing agent state;
- UI exit, process death and reboot cannot erase a committed goal/task checkpoint.

## 8. General invariants

These are architecture-level constraints:

- one canonical runtime;
- imported formats do not become architectural forks;
- every side effect passes through a typed capability;
- every task has observable completion criteria;
- learned capabilities are versioned and rollback-capable;
- portable intelligence has integrity/provenance metadata;
- mobile resource pressure can reduce compute but cannot silently corrupt state;
- internet/PC connectivity is optional, never assumed;
- execution and verification are separate concepts.


## 9. Phase 003A architecture freeze

The dependency and responsibility boundaries are now frozen in `docs/ARCHITECTURE_FREEZE.md`.

NTD97 IR v0 is specified in `docs/IR_V0.md`.

Future binary format, importers, execution kernels, Android shells and 3D components must conform to those boundaries rather than redefining the sovereign execution path.
