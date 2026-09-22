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
- canonical `ntd97.tokenizer.v2` payload with backward decoding of `ntd97.tokenizer.vocab.v1`;
- deterministic native vocabulary tokenizer, LLaMA-style SentencePiece score-ordered BPE with byte fallback, and canonical GPT-2 Unicode pre-tokenization + byte-level ranked BPE;
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

## M8 — Major Block H: Capability Forge + Native Assimilation — complete

Goal: allow supported external capabilities and intelligence sources to be discovered, validated and converted into signed NTD97-native assets without retaining the source runtime.

Delivered contract:

- media-type importer registry with deterministic capability/intelligence discovery;
- source digest, URI, attribution and license provenance capture;
- explicit allowlisted-license and source-size policy gates;
- source importer boundary that must terminate in a validated `NativeCandidate`;
- generated native adapter/skill assets using the existing CapabilityDescriptor contract;
- source intelligence normalization into NTD97 IR graphs;
- native-only isolated regression sandbox with no network or external writes;
- graph semantic round-trip and adapter-integrity regression probes;
- NCC97 packaging using canonical Graph / Capabilities / Adapters / Provenance / Signatures / AssimilationLog sections;
- Ed25519-signed native packages with trusted-signer verification;
- monotonic asset versioning;
- all-or-none batch commit semantics;
- explicit rollback to a previously verified native version;
- source bytes and source runtime excluded from committed native assets;
- end-to-end tests for capability assimilation, intelligence assimilation, provenance/license denial, signature tampering, version conflict and rollback;
- Rust 1.80-compatible pinned signing dependency graph.

Acceptance result: Rust `fmt -> clippy -D warnings -> workspace tests` PASS and Android build/lifecycle regression PASS. Supported imported intelligence and capabilities become signed NTD97-native NCC97 assets; successful runtime use no longer depends on the source runtime.

Exit: complete.

## M9 — Major Block I: Portable Sovereign Intelligence — complete

Goal: make native NTD97 model intelligence, capabilities, memory and state portable across installations/devices without depending on the original app instance or network service.

Delivered contract:

- inward-dependent `ntd-portability` crate;
- canonical portable asset classes for Model / Capability / Memory / State;
- strict NCC97 capsule-kind validation per asset class;
- user-owned 256-bit backup key with redacted debug output and zeroization on drop;
- XChaCha20-Poly1305 encryption for content objects and backup manifests;
- SHA-256 content addressing;
- deterministic per-content object nonce derivation bound to digest and logical length;
- encrypted sovereign object store with integrity verification on insert/import/read;
- Full portable backup containing all referenced encrypted objects;
- Thin portable backup containing only encrypted manifest references to content-addressed objects;
- State portable backup containing only Memory / State assets;
- deterministic PSB97/PSM97 backup and encrypted-manifest framing;
- canonical asset ordering and duplicate rejection;
- all-or-none staged restore into a destination sovereign object store;
- cross-device restore using the same user-owned key;
- model / capability / memory / state migration in one full backup;
- offline restore path with no network dependency;
- wrong-key rejection;
- corrupted-object rejection without partial destination mutation;
- missing Thin-backup object fail-closed behavior;
- content-addressed deduplication across repeated backups.

Acceptance result: Rust `fmt -> clippy -D warnings -> workspace tests` PASS and Android build/lifecycle regression PASS. Full backup restores model, capability, memory and identity/state assets into a fresh destination store; Thin and State semantics, cryptographic failure paths and sovereign recovery all pass deterministic tests.

Exit: complete.

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
- no-third-party-AI-dependency audit;
- NDE97 physical-device evidence records;
- CI-attested Android evidence APK;
- emulator-rejecting physical evidence collector.

Current status: CI validation foundation is implemented and green. Final M10 completion remains blocked until distinct physical 4 GB / 8 GB / 12 GB device records satisfy the declared latency, energy, reliability, recovery and sovereignty targets.

Exit: measurable capability, latency, reliability, recovery and sovereignty targets are met on representative mobile hardware.


## Post-M10 roadmap lock

From M11 onward NTD97 is developed only as large end-to-end capability blocks. Small phases may exist locally while implementing a block, but they are not standalone roadmap milestones and are not merged merely because an interface or mock passes.

M10 physical-device evidence remains a parallel hardware-validation gate. It does not block implementation of the intelligence/product blocks below, and it cannot substitute for them.

A post-M10 block is complete only when its production path, deterministic regression coverage, failure behavior, sovereignty boundary and mobile integration are all present.

## M11 — Major Block K: Real Native Intelligence — complete

Goal: replace toy/reference intelligence with importable, executable, source-runtime-independent model intelligence that actually exercises the NTD97 generative stack.

Required contract:

- clean-room GGUF v3 intake in Rust with no llama.cpp/third-party model runtime dependency;
- safe file-backed GGUF byte source with bounded metadata/table parsing and per-tensor range reads;
- content-addressed NTP97 tensor shard staging with signed Thin NCC97 external references, releasing converted tensor payloads after each shard is persisted;
- bounded parsing of typed metadata, arrays, tensor tables, alignment and tokenizer metadata;
- fail-closed rejection of malformed, unsupported-version, duplicate or structurally invalid inputs;
- explicit conversion plan separating directly materializable tensors from tensors requiring native transcode;
- additive NTD97 IR transformer semantics required for production lowering;
- learned-weight RMSNorm, SiLU, reshape/transpose and grouped-query causal attention;
- canonical Llama-family architecture/tensor-name lowering into NIR97;
- GGML quantized-tensor transcode into an NTD97-owned tensor representation supported by mobile execution;
- native tokenizer conversion into the canonical NCC97 tokenizer section;
- content-addressed tensor shard emission and graph/tensor binding validation;
- source-vs-NIR97 semantic equivalence fixtures before activation;
- signed NCC97 native package production through the existing assimilation/forge boundary;
- CLI/import API suitable for large model files without retaining the source runtime after successful commit;
- end-to-end test: GGUF fixture -> NTD97 IR/NCC97 -> native generator -> deterministic text.

Current implementation:

- bounded clean-room GGUF v3 parser/intake and conversion planning;
- fail-closed tokenizer/tensor-table validation;
- NTD97 IR 0.4 transformer semantics: SiLU, dynamic Reshape, Transpose, PositionIds, learned/model-epsilon RMSNorm and grouped-query causal attention;
- strict canonical LLaMA-family metadata/tensor-role lowering into a full-context NIR97 graph;
- exact GGUF tensor-boundary validation;
- direct F32/F16/BF16 materialization plus clean-room Q4_0/Q8_0/Q4_K/Q5_K/Q6_K native transcode to NTD97-owned F32;
- native tokenizer section, tensor descriptor table and NTP97 tensor shard emission;
- canonical generative manifest carrying token-input, distribution-output and vocabulary bootstrap metadata;
- preservation and validation of GGUF tokenizer semantic metadata;
- resolved canonical LLaMA/SPM policy semantics for omitted optional GGUF flags, with explicit metadata overrides and CLI-visible provenance;
- pinned `stories260K.gguf` real-model reference profile with source artifact hashes and reproducible same-step evidence;
- native LLaMA-style SentencePiece execution carried through GGUF -> NCC97 v0.2 -> runtime, including score-ordered merges, U+2581 space normalization, byte fallback and source add-space/BOS/EOS policy;
- real `tok512` source-vs-NTD97 tokenizer differential PASS over seven fixed prompt cases;
- real `stories260K` GGUF -> streamed NTP97 -> signed Thin NCC97 -> lazy native generation equivalence PASS for all 128 declared context steps, with byte-identical normalized source/native text;
- Android host exporter producing the same real model as a signed Thin NCC97 package with 52 NTP97 shards and a deterministic 16-token native reference;
- Android JNI verification of package signature plus external shard integrity, Thin activation and lazy native generation;
- Android x86_64 emulator generation PASS with exact 16-token host-reference equality and preserved lifecycle/reboot validation;
- native canonical GPT-2 Unicode pre-tokenization + byte-level ranked BPE carried through GGUF -> NCC97 v0.3 -> runtime;
- file-backed bounded GGUF source reads, streamed content-addressed tensor staging and signed Thin NCC97 packaging;
- external Thin tensor references covered by the native package signature and length/hash verification;
- lazy `ValueId` tensor resolution from verified NTP97 shards with graph-value release after final use;
- Thin-package commit/rollback through the canonical native asset store;
- Forge-native intelligence candidate creation;
- signed Full/Thin NCC97 verification through the canonical native package boundary;
- deterministic tiny-LLaMA GGUF -> NIR97/NCC97 -> GraphGenerator regression path;
- unsupported source semantics, unconsumed tensors and unsupported GGML types remain fail-closed.

Acceptance result:

- pinned `stories260K.gguf` source tokenizer differential PASS;
- pinned `stories260K.gguf` 128-step source-vs-NIR97 generation equivalence PASS;
- signed Thin NCC97 package + 52 content-addressed NTP97 shards verified and activated natively;
- Android x86_64 emulator signature/shard verification + 16-token real native generation PASS;
- no GGUF/source checkpoint/llama.cpp runtime/hosted AI backend required after native package creation;
- representative real GPT-2 differential validation and additional pre-tokenizers such as Qwen2/LLaMA3 remain future support expansion and continue fail-closed until independently proven.

Exit: complete. At least one real supported external model can be imported once, converted to signed NTD97-native NCC97 assets, then loaded and used for local text generation on Android without the source model runtime or a hosted AI backend.

## M12 — Major Block L: End-to-End Native Assistant Loop — complete

Goal: make the Android product use the real native intelligence path rather than lifecycle/test harnesses.

Required contract:

- chat composer, conversation surface and token streaming;
- prompt/chat-template compilation into the native tokenizer path;
- user input -> native generation -> adaptive cognition -> memory -> task graph -> response;
- Reflex/Standard/Deep/Recovery budgets driven by real model inference;
- sovereign conversation/context memory with bounded context construction;
- interruption/cancellation without committing partial invalid state;
- checkpointable generation/task state across UI exit and process death;
- final answer synthesis from verified action results;
- Android JNI surface for submit/cancel/stream/status rather than test-only lifecycle calls;
- end-to-end offline chat and reasoning acceptance tests.

Current implementation:

- Android chat composer, transcript surface, local token streaming and user cancellation through production JNI submit/stream/status APIs;
- bounded canonical prompt compilation preserving complete recent dialogue pairs;
- verified real native model loading from signed Thin NCC97 + NTP97 shards with no hosted fallback;
- partial cancelled/failed generations remain uncommitted;
- sovereign NCS97 conversation state binding each active turn to cognitive task identity and native model identity;
- token-level NCS97 checkpoint/restore across Activity destruction/process continuity using Android AtomicFile storage;
- completed dialogue stored as episodic sovereign memory and restored with the cognitive checkpoint;
- native model-logit preflight computes entropy/top-margin uncertainty and feeds canonical Reflex/Standard/Deep/Recovery budget selection;
- prior paused/failed conversation state escalates the next preflight through the existing Recovery signal;
- budget-dependent sovereign memory recall is compiled back into the final bounded prompt while recent dialogue is preserved ahead of lower-ranked memory;
- selected reasoning budget, complexity/uncertainty signals and retained-memory count are persisted in cognitive world state and survive NCS97 restore;
- the selected budget now drives the canonical CognitiveRuntime planner/executor/verifier loop before answer generation: Reflex=1, Standard=2, Deep=4, Recovery=3 verified NIR97 forward iterations;
- each reasoning iteration extends a private scratch-token prefix from real native logits; scratch tokens are discarded before answer streaming while verified iteration evidence is persisted in NCS97 cognitive state;
- Android real-model acceptance requires exact budget-to-iteration equality before and after NCS97 restore, adaptive reasoning selection, nonzero memory recall on an overlapping follow-up turn, cancellation and lifecycle/reboot regression;
- strict native action protocol parsing now materializes canonical non-empty TaskGraph + TypedAction payloads only for allowlisted read-only actions; malformed, non-canonical, write or irreversible requests fail closed;
- canonical action protocol text is persisted beside the cognitive task graph in NCS97 so typed payloads can be deterministically reconstructed after process death;
- a hidden native planning pass runs after verified reasoning and records direct/actions/invalid outcome without exposing planner scratch output in the transcript;
- Android production action execution currently supports model-grounded `device.observe` through the existing ActionFabric, a local ResourceSnapshot adapter and evidence-checking verifier;
- generic verified-action coordination prepares the cognitive task through ActionFabric, refuses suspended/retryable/failed plans, and permits answer synthesis only from Completed plans whose actions are all Committed;
- verified action results replace the active answer prompt only after evidence collection, while direct/invalid planner output creates no task-graph side effects;
- Android acceptance checks fail-closed planner invariants and preserves planner/action evidence across NCS97 restore.
- governed live-device queries now use constrained native-logit candidate scoring over allowlisted read-only observation surfaces; DIRECT is not admissible when current device evidence is required;
- Android device evidence is surface-scoped before verified synthesis so battery/thermal/memory turns do not inflate the bounded prompt with unrelated fields;
- verified-action synthesis provenance is persisted inside NCS97 and exposed through the production JNI chat contract;
- the final M12 Android acceptance turn requires a current-battery request to produce a non-empty verified TaskGraph, committed action evidence, verified synthesis provenance and a completed native response in the same request.

Acceptance result:

- real-model Android governed turn PASS on the canonical native stack;
- constrained native-logit planner status = actions;
- non-empty TaskGraph action count = 1;
- committed verified action count = 1;
- verified synthesis provenance = ready;
- native synthesized response streamed 4 tokens and terminated with COMPLETE;
- final request status = COMPLETE;
- adaptive reasoning loop, sovereign memory recall, NCS97 restore, cancellation and lifecycle/reboot regression remained green in the same emulator acceptance run;
- no external AI backend or source model runtime participates in the Android assistant loop after native package creation.

Broader production Android adapters remain future task-surface expansion rather than an M12 exit blocker; M12's governed-action acceptance is intentionally scoped to the verified read-only `device.observe` path.

Exit: complete. The canonical APK can hold a useful local conversation, reason through the native model, remember relevant state, execute a governed verified local action, synthesize a response from verified evidence and resume interrupted work without any external AI backend.

## M13 — Major Block M: Real-World Capability Adapters

Goal: replace mock capability adapters with production adapters while preserving the existing authority/verification model.

Required contract:

- real HTTP/WebSearch/WebFetch boundary;
- browser observe/interact adapter;
- scoped Android file read/write through platform storage APIs;
- Android device observation/control where platform policy permits;
- app launch/intents/accessibility-assisted interaction behind explicit user authority;
- verified download/upload/artifact handling;
- network/offline transitions and resumable actions;
- side-effect receipts, rollback where feasible and idempotent cold resume;
- paired-PC fabric integrated into the same production task graph;
- real mixed Web -> File -> App/Device -> PC acceptance tasks.

Current implementation:

- production Android `web.fetch` adapter uses the OS HTTPS stack rather than a hosted AI/tool backend;
- HTTPS boundary is GET-only, timeout-bounded and response-size-bounded, re-validates redirects, rejects user-info/non-443 endpoints and blocks loopback/link-local/site-local/multicast/IPv6-ULA destinations;
- provider-independent `web.search` reuses that hardened HTTPS transport through a runtime-configured endpoint template with canonical `{query}` and optional `{count}` placeholders; the sovereign/native core contains no search-provider SDK, provider hostname or hosted AI/tool dependency, and missing/invalid configuration fails closed;
- search endpoint configuration is persisted only in Android app-private preferences and remains replaceable without changing `TypedAction::WebSearch`, `CapabilityDescriptor`, TaskGraph identity or verification semantics;
- production `browser.observe` / `browser.interact` use an app-owned WebView platform boundary while ActionFabric remains the owner of typed actions, authority, verification and commit state;
- browser navigation and subresource access are restricted to public HTTPS, TLS errors are cancelled, file/content access and mixed content are disabled, and private/local targets fail closed;
- `browser.interact` supports only fixed `click` and `set_value` operations with no arbitrary JavaScript input; it is classified as `ExternalWrite`, requires the dedicated `browser.interact` authority scope plus explicit external-write approval, and must return a platform receipt before verifier acceptance;
- browser session loss, unsupported operations, missing targets, invalid selectors/values, transport errors and untrusted receipts fail closed rather than synthesizing success;
- native ActionFabric retains ActionId, authority, verification and commit ownership while Java supplies only the Android HTTPS/browser platform boundaries;
- production `file.read` / `file.write` adapters are restricted to an app-private capability root, reject absolute/traversal/symlink paths and cap evidence/write size;
- app-private file writes use sync + atomic rename and emit rollback tokens restoring the previous bytes or removing newly-created files;
- native action protocol now represents `file.write` as a reversible side effect instead of a read-only action;
- Android production TaskGraph execution accepts only allowlisted `device.observe`, `web.fetch`, `file.read` and `file.write` capabilities; unsupported domains remain fail-closed;
- Android emulator acceptance performs a real HTTPS fetch, rejects a private-network HTTPS target, verifies app-private write/read content and verifies rollback to the absent state;
- native action protocol represents `device.interact` and `app.action` as `ExternalWrite` side effects rather than read-only operations;
- production Android `device.interact` currently supports explicit-authority clipboard `set_text` with platform read-back verification and a native evidence receipt, without persisting the previous clipboard content;
- production Android `app.action` currently supports exact-package `launch` through an explicit Android Intent and emits a native evidence receipt without broad package-query permission;
- both app/device write capabilities require dedicated scopes plus `allow_external_write=true`; default grants are proven to deny execution before the side effect;
- Android emulator acceptance verifies both default authority rejection and explicit-authority clipboard/app-launch execution.
- native `artifact.download` is a first-class reversible/resumable Web-domain action persisted in TAF97 0.3 rather than a loose `web.fetch + file.write` composition;
- Android artifact downloads use the hardened HTTPS boundary, stage bytes under the app-private capability root, persist hash + rollback state in the ActionFabric resume token, then hash-verify and atomic-rename only during resume;
- completed artifact downloads emit SHA-256/path/byte-count receipts, support rollback to the prior file/absent state and preserve staged state across action checkpoints;
- Android emulator acceptance requires real HTTPS artifact staging, suspended action state, verified resume/commit and rollback.
- canonical chat ExternalWrite approval is persisted in sovereign cognitive/NCS97 state and emits a dedicated approval-required event, separate from legacy continuity approval;
- clipboard writes and exact-package app launches remain suspended until explicit user approval; approval derives exact authority scopes from the persisted canonical native action protocol, while deny performs no side effect;
- interrupted `approved-executing` state restores as `reconfirm` and never replays an ExternalWrite automatically;
- Android chat UI checkpoints pending approval, routes Approve/Deny through the chat request ID and resumes verified synthesis only after native receipt verification;
- emulator acceptance proves pending clipboard approval survives NCS97 restore, deny preserves prior clipboard state, and approve produces the exact requested write plus verified synthesis.
- canonical chat now persists ExternalWrite approval state inside sovereign cognition/NCS97, emits a dedicated approval-required chat event and never reuses the legacy continuity approval state machine;
- explicit clipboard/app-launch actions are validated from canonical native action protocol, suspended before side effects, and only receive exact ExternalWrite authority after user approval;
- interrupted `approved-executing` state restores as `reconfirm` rather than replaying an external write automatically;
- Android chat UI checkpoints pending approvals, separates chat Approve/Deny from continuity approval, resumes verified synthesis after approval and cancels denied tasks without side effects;
- emulator acceptance proves clipboard state is unchanged before approval and after deny, pending approval survives NCS97 restore, and approved execution produces verified action evidence before generation resumes;
- emulator acceptance now also requires unconfigured WebSearch to fail closed, a real runtime-configured WebSearch HTTPS boundary, public browser observation, private-target rejection, default browser-interaction authority denial and a receipt-verified approved browser interaction.

Still required before M13 completion:

- user-facing WebSearch endpoint configuration/discovery and provider-result normalization beyond the provider-independent runtime boundary;
- broader browser session persistence/navigation/form semantics beyond the initial fail-closed observe/click/set-value foundation;
- broader Android storage surfaces through explicit platform/user grants;
- connect production app/device external-write adapters to the canonical chat approval/suspend/resume flow; accessibility-assisted interaction remains fail-closed until its explicit-authority adapter is implemented;
- verified upload handling and broader resumable network transitions beyond bounded artifact downloads;
- paired-PC integration into the same production mixed TaskGraph;
- real mixed Web -> File -> App/Device -> PC acceptance on representative phone hardware.

Exit: NTD97 can complete useful multi-surface tasks on a real phone with verifiable results and no mock adapter in the canonical path.

## M14 — Major Block N: Cognitive Quality + Capability Growth

Goal: turn the runtime/capability substrate into a progressively more capable general agent.

Required contract:

- task-conditioned retrieval across semantic/procedural/episodic memory;
- long-context compression and context budgeting;
- model-grounded planning with verifier-driven revision;
- persistent learned deltas/adapters with pre-activation evaluation;
- capability-gap detection;
- capability discovery/build/test/package/install loop using the existing forge;
- signed/versioned activation with rollback to known-good state;
- benchmark corpus for reasoning, planning, memory, tool use and recovery;
- safeguards preventing unverified generated code/capability state from silently becoming active.

Exit: NTD97 can detect a missing supported capability, build or assimilate a candidate, validate it in isolation, activate it transactionally and use it in a later task while preserving rollback.

## M15 — Major Block O: Embodied Multimodal Assistant

Goal: make the 3D assistant and voice path a first-class interface to the same sovereign intelligence.

Required contract:

- native/local speech-to-text path;
- native/local text-to-speech path;
- streaming speech turn-taking and interruption;
- real character mesh/rig, skeletal animation and expression state;
- viseme-driven lip sync from actual speech output;
- gaze/gesture/progress driven by cognitive/action state;
- floating assistant interaction within Android policy;
- camera/screen/sensor perception only through explicit capability/permission boundaries;
- thermal-aware fidelity degradation without cognition loss.

Exit: the user can converse with and interact with an embodied NTD97 assistant while all reasoning, memory and governed actions remain tied to the same NTD97 identity.

## M16 — Major Block P: Production General-Agent Convergence

Goal: validate NTD97 as a durable mobile general-agent product rather than a collection of subsystem demonstrations.

Required contract:

- real-model latency/energy/RAM profiling across representative 4/8/12+ GB phones;
- CPU/Vulkan/NPU provider equivalence and autotuning on real hardware;
- long-horizon mixed-task reliability and recovery;
- multi-day logical-continuity soak under Android lifecycle constraints;
- cold boot/reboot/update/backup/restore migration tests;
- security and sovereignty boundary audits;
- no-third-party-AI-dependency audit on release artifacts;
- model/capability corruption and rollback drills;
- Android-version/device-vendor compatibility matrix;
- production signing/update path;
- measurable quality benchmark for conversation, reasoning, memory, tool use and task completion;
- portable platform boundary prepared for future non-Android shells without moving cognition out of the NTD97 core.

Exit: a release candidate demonstrates useful native intelligence, governed real-world agency, continuity, portability and measured mobile performance on representative physical hardware. Passing M16 is a product-readiness milestone; it is not by itself a scientific proof of AGI.
