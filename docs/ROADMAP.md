# NTD97 Roadmap

The roadmap is milestone-based. Architecture changes should be made only when an invariant cannot be met.

## M0 — Clean Foundation

- repository bootstrap;
- canonical architecture;
- Rust workspace;
- typed core contracts;
- minimal CI;
- no inherited AMPER code.

## M1 — NCC97 Native Intelligence + Assimilation Format

### Phase 003A — Architecture Freeze + NTD97 IR v0

- freeze identity and dependency boundaries;
- establish independent `ntd-ir` crate;
- define IR 0.1 version contract;
- define initial tensor/state/memory/control/tool operation families;
- structural graph validation;
- bind NCC97 compatibility to IR version;
- runtime IR compatibility gate.

Exit: architecture/IR contracts compile, match canonical documentation and pass CI.

### Phase 003B — NCC97 Binary Capsule

- deterministic binary header and manifest encoding;
- chunk index and section descriptors;
- content hashes and integrity verification;
- reader/writer round-trip;
- mmap/streaming-friendly access contract;
- thin/full/state capsule primitives.

Exit: a native capsule can be deterministically written, reopened, validated and round-tripped without implementing any external-model importer yet.

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
- persistent checkpoints;
- persistent world/goal state;
- cold-process reconstruction of the same cognitive identity.

Exit: one runtime can switch from low-latency reflex behavior to deeper iterative planning without changing backend identity.

## M4 — Tool + Internet Fabric

- capability registry;
- typed web/search/browser/file actions;
- permission scopes;
- side-effect classification;
- verification and rollback contracts.

Exit: tasks can discover and use tools from one capability graph.

## M5 — Android Interaction + 3D Embodiment + 24/7 Continuity

- Android app shell;
- screen/observation pipeline;
- permitted AccessibilityService automation;
- foreground/background task continuity;
- notifications and user approvals;
- media/voice interfaces;
- interactive 3D avatar scene;
- adaptive 3D renderer with expression/gaze/lip-sync/gesture state;
- user-authorized floating assistant surface where the OS permits it;
- persistent active-task service policy;
- scheduled/retry wake path;
- reboot/process-death restoration;
- state reconstruction and checkpoint verification.

Exit: NTD97 can perform verified multi-app tasks, present an interactive 3D assistant, survive UI exit/process death/reboot at the logical task level, and resume eligible work under Android execution rules.

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

## M8 — Sovereign Memory + Portable Intelligence

- episodic/semantic/procedural stores;
- adapter/delta learning hooks;
- thin/full/state capsule backup;
- restore across devices;
- encrypted user-owned state;
- content-addressed NTD97-owned memory format;
- full sovereign offline restore test.

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

## M10 — Sovereign General Mobile Agent Validation

- long-horizon task suites;
- app/browser/file/PC mixed tasks;
- failure recovery;
- offline degradation;
- security boundaries;
- reproducible benchmarks;
- 24/7 continuity soak tests;
- offline boot + restore + local task suite;
- no-third-party-AI-dependency audit.

Exit: measurable capability, latency, reliability and recovery targets are met on a representative device matrix.
