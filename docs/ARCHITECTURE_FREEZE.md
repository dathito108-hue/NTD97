# NTD97 Architecture Freeze — v0

Status: canonical contract for Phase 003A.

This document freezes the dependency boundaries required before NCC97 binary implementation begins.

## 1. Frozen identity

NTD97 is one sovereign mobile AGI identity.

The stable native system is:

```text
NTD97 Identity
  |
  +-- SIK97 cognitive/runtime state
  +-- NTD97 IR
  +-- NCC97 native intelligence state
  +-- sovereign memory
  +-- installed capabilities
```

External model formats, cloud APIs, desktop computers and rendering engines are not part of the identity.

## 2. Frozen execution path

The canonical native execution path is:

```text
NCC97
  -> manifest compatibility gate
  -> NTD97 IR
  -> SIK97 / NTD97 runtime
  -> hardware provider
  -> observable result
```

No imported model runtime may sit beside this path as a permanent alternate backend.

## 3. Frozen dependency direction

At the Rust package level:

```text
ntd-ir
  ^
  |
ntd-capsule

ntd-core     ntd-ir
    ^          ^
     \        /
      ntd-runtime

future importer ----> ntd-ir + ntd-capsule
future platform ----> ntd-runtime
future 3D UI -------> runtime events / state
```

Rules:

- `ntd-ir` has no dependency on runtime, platform, importer or UI crates.
- `ntd-capsule` may know the IR version contract but never depends on runtime execution.
- `ntd-runtime` consumes core and IR contracts; platform-specific APIs do not enter it.
- importers convert source ecosystems into IR/capsule state and do not become execution backends.
- Android/iOS/Desktop shells depend inward on the runtime; the sovereign core never depends outward on a platform shell.
- the 3D assistant consumes observable runtime state; cognition never requires the renderer to exist.

## 4. Frozen responsibility boundaries

### NTD97 IR owns

- portable operation semantics;
- graph value/node identity;
- tensor/state/memory/control/tool operation classes;
- structural graph validation;
- IR version compatibility.

### NCC97 owns

- durable native intelligence packaging;
- section/chunk addressing;
- capsule/IR compatibility declaration;
- integrity/provenance/assimilation metadata;
- backup/restore representation.

NCC97 does **not** own reasoning policy or execute operations.

### SIK97 / ntd-runtime owns

- reasoning budget;
- cognitive scheduling;
- graph execution orchestration;
- world/goal/task state;
- resource policy;
- checkpoint/resume orchestration;
- provider selection.

The runtime does **not** parse arbitrary external model formats.

### Importers own

- source parsing;
- semantic normalization;
- source-to-IR conversion;
- source provenance capture;
- assimilation candidate creation.

Importers cannot directly mutate live sovereign state without an assimilation transaction.

### Platform shells own

- lifecycle;
- permissions;
- notifications;
- platform scheduling;
- sensors/UI bridges;
- rendering surfaces.

Platform shells cannot redefine model semantics.

## 5. Change-control rule

After Phase 003A, changing one of the following requires an explicit architecture decision and migration impact review:

- NTD97 identity boundary;
- canonical execution path;
- package dependency direction;
- IR major-version semantics;
- NCC97 major-version compatibility rules;
- separation between importer and runtime;
- separation between cognition and UI/3D renderer.

Minor additive IR operations may be introduced under the normal IR compatibility rules.

## 6. What Phase 003A intentionally does not implement

- NCC97 binary wire layout;
- mmap reader/writer;
- tensor payload serialization;
- GGUF parsing;
- tensor execution kernels;
- CPU/Vulkan/NPU inference;
- live assimilation transaction engine.

Those belong to later phases and must conform to this frozen contract.

## 7. Acceptance criteria

Phase 003A passes only if:

1. `ntd-ir` exists as an independent crate;
2. IR v0 has explicit versioning;
3. graph structural validation rejects dangling/duplicate value definitions;
4. NCC97 declares the IR compatibility contract;
5. the runtime has an explicit IR compatibility gate;
6. the workspace builds without third-party runtime dependencies;
7. canonical docs match the code contract;
8. CI format, clippy and unit tests pass.
