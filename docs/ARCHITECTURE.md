# NTD97 Canonical Architecture

Status: Phase 001 canonical foundation.

## 1. Goal

NTD97 is not defined as a chat application with tools attached. The core abstraction is an **intelligence runtime that compiles intent into verified actions**.

The conversational interface is only one interaction surface.

The runtime is designed around five priorities:

1. deep reasoning when the task actually needs it;
2. low time-to-first-action;
3. minimal unnecessary language generation;
4. broad interaction through capability adapters;
5. portable and recoverable intelligence state.

## 2. Single-runtime rule

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

## 4. Mobile execution substrate

The portable core is implemented in Rust.

Platform shells bind through a narrow C ABI/JNI/Swift FFI boundary.

Execution providers are selected dynamically:

1. device accelerator provider if compatible;
2. Vulkan/Metal compute provider;
3. optimized CPU provider;
4. optional paired-PC or remote provider when explicitly allowed.

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

## 7. Invariants

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
