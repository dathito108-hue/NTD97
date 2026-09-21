# NTD97 Roadmap

The roadmap is milestone-based. Architecture changes should be made only when an invariant cannot be met.

## M0 — Clean Foundation

- repository bootstrap;
- canonical architecture;
- Rust workspace;
- typed core contracts;
- minimal CI;
- no inherited AMPER code.

## M1 — NCC97 Native Intelligence Format

- binary header and manifest;
- chunk index;
- mmap/streaming reader;
- integrity verification;
- thin/full/state export;
- first GGUF + SafeTensors import experiments.

Exit: a capsule can be created, validated, opened and round-tripped.

## M2 — Adaptive Mobile Runtime

- device capability detection;
- tensor store;
- CPU reference provider;
- Vulkan provider interface;
- memory/thermal/battery budget manager;
- execution graph loader.

Exit: deterministic test graph executes through the same provider contract used by mobile.

## M3 — Cognitive Loop

- intent IR;
- adaptive reasoning budget;
- task graph compiler;
- executor/verifier split;
- short-response policy;
- persistent checkpoints.

Exit: one runtime can switch from low-latency reflex behavior to deeper iterative planning without changing backend identity.

## M4 — Tool + Internet Fabric

- capability registry;
- typed web/search/browser/file actions;
- permission scopes;
- side-effect classification;
- verification and rollback contracts.

Exit: tasks can discover and use tools from one capability graph.

## M5 — Android Interaction Fabric

- Android app shell;
- screen/observation pipeline;
- permitted AccessibilityService automation;
- foreground/background task continuity;
- notifications and user approvals;
- media/voice interfaces.

Exit: NTD97 can perform verified multi-app tasks on a supported Android phone.

## M6 — Paired PC Fabric

- mutual authentication;
- encrypted session;
- typed remote capabilities;
- artifact transfer;
- desktop observation/execution agent.

Exit: the phone can delegate a typed task to a trusted PC and verify the returned result.

## M7 — Capability Forge

- capability search/discovery;
- provenance/license capture;
- isolated build/test sandbox;
- generated adapter/skill packages;
- signature/version/rollback;
- regression suite.

Exit: a missing capability can be acquired or developed without mutating the core runtime irreversibly.

## M8 — Memory + Portable Intelligence

- episodic/semantic/procedural stores;
- adapter/delta learning hooks;
- thin/full/state capsule backup;
- restore across devices;
- encrypted user-owned state.

Exit: intelligence state can be backed up and restored independently of the app install.

## M9 — Performance Convergence

- latency tracing;
- prefix/state reuse;
- memory paging;
- quantization profiles;
- provider autotuning;
- thermal-aware execution;
- power-aware background scheduling.

Exit: device profiles automatically select the fastest verified configuration that fits resource constraints.

## M10 — General Mobile Agent Validation

- long-horizon task suites;
- app/browser/file/PC mixed tasks;
- failure recovery;
- offline degradation;
- security boundaries;
- reproducible benchmarks.

Exit: measurable capability, latency, reliability and recovery targets are met on a representative device matrix.
