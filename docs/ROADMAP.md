# NTD97 Roadmap

The roadmap is milestone-based. Architecture changes should be made only when an invariant cannot be met.

From the Native Intelligence Execution Foundation onward, tightly coupled work is delivered as **major implementation blocks**. A block is merged only when it provides a meaningful end-to-end capability rather than a sequence of small phase-only changes.

## M0 — Clean Foundation — complete

- repository bootstrap;
- canonical architecture;
- Rust workspace;
- typed core contracts;
- minimal CI;
- no inherited AMPER code.

## M1 — Native Intelligence Format + Execution Foundation

### Completed foundation contracts

- Architecture Freeze + NTD97 IR v0;
- NCC97 deterministic binary capsule;
- NCC97 Native IR Serialization;
- SHA-256 capsule/chunk integrity;
- Full/Thin/State capsule primitives;
- deterministic NIR97 graph encode/decode;
- tensor descriptor and tokenizer/codec framing.

### Major Block A — Native Intelligence Execution Foundation

Goal: prove that native NTD97 intelligence can be loaded and executed end-to-end without a source-model runtime.

Delivered contract:

- deterministic `NTP97` native tensor payload framing;
- content-addressed tensor shard resolution;
- embedded/external NCC97 tensor loading;
- tensor descriptor-to-shard integrity binding;
- graph-input-to-tensor binding validation;
- quantization metadata for native/unquantized, symmetric I8 and affine I8;
- reference tensor materializer;
- CPU reference provider;
- NTD97 IR graph executor;
- NCC97 native graph/tensor program loader;
- deterministic `.ncc97 -> native tensors -> IR execution -> output` integration test.

Exit: a deterministic test graph executes entirely through NTD97-owned formats and provider contracts. No GGUF/ONNX/TFLite runtime is required or allowed in the canonical execution path.

## M2 — Major Block B: Native Generative Intelligence Runtime

Goal: turn the generic native tensor/IR executor into a usable local generative model runtime while preserving NTD97 identity.

Delivered contract:

- NTD97 IR 0.2 additive `CausalAttention` semantic;
- CPU reference Gather, RotaryPosition and CausalAttention;
- canonical `ntd97.tokenizer.vocab.v1` tokenizer payload inside NCC97 descriptor framing;
- deterministic longest-prefix native vocabulary tokenizer;
- source-independent `GraphGenerator` autoregressive decode loop;
- greedy and seeded stochastic sampling with temperature/top-k;
- logits/probability distribution contracts;
- bounded token context;
- NTD97-owned `KvCache` and `PrefixCache` representations;
- Full `.ncc97` native generative package loader;
- deterministic source-independent golden text generation test;
- importer boundary preserved: external formats must terminate at NTD97 IR + NCC97 state.

Exit: an NTD97-native text model accepts tokens/text, executes locally and generates deterministic/reference output without a third-party model backend.
## M3 — Major Block C: Adaptive Mobile Compute Runtime

Goal: make the same native model path practical across current phones.

Delivered contract:

- portable CPU thread detection plus platform-supplied RAM/Vulkan/NPU capability reporting;
- `ResourceSnapshot` for available RAM, battery, charging, thermal state and latency budget;
- `ComputePolicy` for RAM reserve, paging, accelerator permission, power preference and quantization profile;
- `CpuReferenceMobileProvider` correctness baseline;
- cache-friendlier `CpuTiledProvider` for MatMul/QuantizedMatMul;
- replaceable Vulkan and NPU provider adapters behind the same `ExecutionProvider` semantics;
- per-operation provider capability and working-set metadata;
- `AutotuneTable` with explicit semantic-equivalence verification;
- latency-budget-aware provider ranking;
- battery/thermal/RAM gating for accelerators;
- deterministic provider fallback on execution failure;
- quantization profiles for F32/F16/I8 mobile placement policy;
- tensor placement planning for resident, accelerator-local or paged CPU storage;
- deterministic `ByteRegion` / `PagedByteReader` paging contract suitable for platform-backed mapped storage;
- representative 4 GB / 8 GB / 12 GB device-profile graph-equivalence tests;
- hot-device fallback test proving accelerator rejection without output drift.

Exit: the same NTD97 graph can select the fastest registered **verified** provider that fits current resource constraints, while preserving CPU-reference semantics and falling back deterministically under pressure.
## M4 — Major Block D: Cognitive Runtime + Sovereign Memory

Goal: build the persistent intelligence loop on top of native inference.

Delivered contract:

- persistent `CognitiveIdentity` and logical tick;
- native goals, tasks, world facts and monotonic state IDs;
- persisted `Intent` + `TaskGraph` task representation from `ntd-core`;
- adaptive Reflex / Standard / Deep / Recovery iterative reasoning in one runtime;
- explicit `CognitivePlanner` / `CognitiveExecutor` / `CognitiveVerifier` split;
- verified internal directives for recall, memory write, world update, work and completion;
- episodic, semantic and procedural `SovereignMemory`;
- deterministic store / retrieve / forget memory semantics;
- verification retry/failure/completion events recorded into episodic memory;
- learned-delta state plus pre-activation hook;
- deterministic SIK97 cognitive checkpoint format;
- strict restored-state validation and incompatible-state rejection;
- NCC97 State-capsule binding through `ContinuityState`;
- cold-process reconstruction test that resumes the same task and identity after checkpoint restore.

Exit: one local NTD97 identity can reason, remember, checkpoint and recover without switching to another model service.
## M5 — Major Block E: Tool, Internet + Device Action Fabric

Goal: give cognition typed, governed real-world capabilities.

Delivered contract:

- versioned `CapabilityRegistry` with Web / Browser / File / Device / App / Custom domains;
- typed WebSearch/WebFetch/BrowserObserve/BrowserInteract/FileRead/FileWrite/DeviceObserve/DeviceInteract/AppAction requests;
- authority scopes with deny-by-default external-write and irreversible gates;
- canonical side-effect contract reused from `ntd-core`;
- direct binding from `CognitiveTask` / `TaskGraph` into action plans;
- transactional plan preparation and stable action IDs;
- capability-version pinning across checkpoint/restore;
- replaceable `CapabilityAdapter` execution boundary with no platform API in sovereign core;
- Accept / Retry / Reject verification semantics;
- rollback tokens and fail-closed rollback behavior;
- resumable actions and adapter failure recovery;
- stable ActionId idempotency contract across retry/cold resume;
- deterministic `TAF97` action-journal checkpoint;
- authority is intentionally re-evaluated after restore rather than persisted;
- NCC97 State-capsule binding through `ContinuityState`;
- end-to-end WebSearch -> FileWrite -> DeviceObserve -> AppAction test across TAF97 restore;
- verifier-rejection test proving external-write rollback;
- transient-adapter failure test proving retry reuses the same ActionId.

Exit: NTD97 can materialize a cognitive TaskGraph into governed typed actions, execute and verify multi-step work across registered local/internet/device adapters, and resume interrupted actions without changing task identity.
## M6 — Major Block F: Android Continuity + Interactive 3D Assistant — complete

Goal: make NTD97 a persistent mobile assistant interface under operating-system limits.

Delivered contract:

- inward-dependent `ntd-mobile-shell` crate;
- deterministic MCS97 envelope combining SIK97 + TAF97 + capability snapshot + lifecycle metadata;
- self-contained cold restore without rebuilding capability registry in Android;
- task/action linkage verification across cognition and tool state;
- durable approval and retry/backoff metadata;
- OS-aware wake policy for interactive/foreground/persistent/approval/verify/suspend paths;
- process-death/reboot acceptance proving committed side effects are not replayed;
- renderer-independent avatar mode/expression/gaze/lip-sync/gesture/progress state;
- thermal/battery-aware 60/30/15/5 FPS render policy;
- local voice/media state machine;
- Android device-protected AtomicFile continuity store;
- Android JobScheduler + reboot receiver + foreground service paths;
- generic/private approval notifications;
- in-app OpenGL ES avatar and permission-gated floating overlay;
- local PCM AudioRecord/AudioTrack bridge;
- concrete local `NtdNativeRuntimeHost` bound to the sovereign Rust runtime through `ntd-android-bridge` JNI;
- no ServiceLoader dependency, hosted runtime, third-party AI backend or cloud fallback in the canonical Android path;
- Android SDK 35 / NDK 27 / Gradle 8.9 / JDK 17 build gate;
- native packaging for `arm64-v8a`, `armeabi-v7a` and `x86_64`;
- debug APK assembly + artifact publication;
- API 35 x86_64 emulator validation covering native host attachment, avatar JNI, PCM bridge, notification channels, foreground-service request, overlay service, persisted JobScheduler work, reboot, and post-reboot cold start.

Acceptance result: Rust `fmt -> clippy -D warnings -> workspace tests` PASS and Android build/lifecycle gate PASS. UI exit, process death and reboot preserve the logical task/continuity contract through platform-approved mechanisms. Broader Android-version and representative-device soak/matrix validation remains part of M10 performance/general-agent validation.

## M7 — Major Block G: Paired PC Fabric — complete

Goal: let phone-owned NTD97 delegate governed work to a trusted computer without moving cognition, memory or identity off the phone.

Delivered contract:

- additive `Pc` capability domain and typed PC observe/execute/artifact actions;
- TAF97 0.2 persistence for paired-PC actions and MCS97 0.2 compatibility;
- pinned Ed25519 peer identities;
- mutually authenticated X25519 session establishment;
- HKDF-derived directional session keys;
- ChaCha20-Poly1305 encrypted PCF97 frames with strict replay sequencing;
- canonical capability, request, result and artifact wire messages;
- request/result digest binding and returned-result verification;
- SHA-256 verified chunked artifact transfer;
- governed system observation, process execution and file artifact handlers;
- execution program allowlists, working-root policy and artifact-root/write/size policy;
- stable ActionId idempotency preventing remote side-effect replay;
- ActionFabric-compatible `PairedPcAdapter`;
- bounded length-prefixed TCP frame transport;
- Rust 1.80-compatible pinned crypto dependency graph;
- regression tests for unpaired identity rejection, encrypted-frame replay, artifact tampering, policy denial, result/request mismatch and TAF97 PC-action restore.

Acceptance result: Rust `fmt -> clippy -D warnings -> workspace tests` PASS and Android build/lifecycle regression PASS. The phone can delegate typed work to an authenticated PC capability node while NTD97 identity, cognition and sovereign memory remain phone-owned.

Exit: complete.

## M8 — Major Block H: Capability Forge + Native Assimilation

- capability discovery;
- provenance/license capture;
- isolated build/test sandbox;
- generated adapters/skills;
- source intelligence import;
- semantic normalization into NTD97 IR/NCC97;
- assimilation validation + atomic commit;
- signature/version/rollback;
- regression suites.

Exit: new supported intelligence and capabilities become NTD97-native assets and no source runtime is required after successful assimilation.

## M9 — Major Block I: Portable Sovereign Intelligence

- encrypted user-owned memory/state;
- thin/full/state capsule backup;
- content-addressed deduplication;
- restore across devices;
- capability/model/memory migration;
- offline boot + restore;
- sovereign recovery tests.

Exit: model intelligence, learned state and memory can be backed up and restored independently of the app installation.

## M10 — Major Block J: Performance Convergence + General Mobile Agent Validation

- latency and energy tracing;
- prefix/state reuse;
- memory paging;
- quantization/provider autotuning;
- thermal-aware execution;
- long-horizon task suites;
- browser/app/file/PC mixed tasks;
- offline degradation;
- failure recovery;
- security-boundary tests;
- 24/7 logical continuity soak tests;
- representative-device matrix;
- no-third-party-AI-dependency audit.

Exit: measurable capability, latency, reliability, recovery and sovereignty targets are met on representative mobile hardware.
