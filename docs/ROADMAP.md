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

- deterministic \`NTP97\` native tensor payload framing;
- content-addressed tensor shard resolution;
- embedded/external NCC97 tensor loading;
- tensor descriptor-to-shard integrity binding;
- graph-input-to-tensor binding validation;
- quantization metadata for native/unquantized, symmetric I8 and affine I8;
- reference tensor materializer;
- CPU reference provider;
- NTD97 IR graph executor;
- NCC97 native graph/tensor program loader;
- deterministic \`.ncc97 -> native tensors -> IR execution -> output\` integration test.

Exit: a deterministic test graph executes entirely through NTD97-owned formats and provider contracts. No GGUF/ONNX/TFLite runtime is required or allowed in the canonical execution path.

## M2 — Major Block B: Native Generative Intelligence Runtime

Goal: turn the generic native tensor/IR executor into a usable local generative model runtime while preserving NTD97 identity.

Planned scope:

- canonical sequence/token semantics;
- native tokenizer contract and implementation;
- attention, positional and gather semantics required by the native model graph;
- KV/state-cache representation owned by NTD97;
- autoregressive decode loop;
- native sampling contract;
- bounded context/prefix reuse;
- deterministic tiny-model golden inference;
- source-independent native model package test;
- importer boundary for converting supported source weights/graphs into NTD97 IR + NCC97 only.

Exit: an NTD97-native text model can accept tokens, execute locally and generate deterministic/reference output without a third-party model backend.

## M3 — Major Block C: Adaptive Mobile Compute Runtime

Goal: make the same native model path practical across current phones.

Planned scope:

- device capability detection;
- optimized CPU provider;
- Vulkan provider behind the same \`ExecutionProvider\` semantics;
- replaceable NPU provider interface where platform support exists;
- memory mapping/paging and tensor placement;
- quantization profiles;
- provider autotuning;
- RAM, thermal, battery and latency budget manager;
- fallback/recovery under resource pressure;
- representative-device deterministic equivalence tests.

Exit: the same NTD97 graph can select the fastest verified local provider that fits device constraints without changing model identity.

## M4 — Major Block D: Cognitive Runtime + Sovereign Memory

Goal: build the persistent intelligence loop on top of native inference.

Planned scope:

- intent/task IR;
- adaptive reasoning budget;
- reflex vs deep iterative reasoning inside one runtime;
- executor/verifier split;
- persistent goals/world state/checkpoints;
- episodic, semantic and procedural memory;
- retrieval/write/forget semantics;
- learned adapter/delta hooks;
- cold-process reconstruction of the same cognitive identity.

Exit: one local NTD97 identity can reason, remember, checkpoint and recover without switching to another model service.

## M5 — Major Block E: Tool, Internet + Device Action Fabric

Goal: give cognition typed, governed real-world capabilities.

Planned scope:

- capability registry;
- typed web/search/browser/file actions;
- permission and authority scopes;
- side-effect classification;
- verification and rollback;
- Android observation and permitted interaction bridge;
- app/device action execution;
- failure recovery and resumable task state.

Exit: NTD97 can plan, execute and verify multi-step tasks across local tools, internet and permitted device surfaces.

## M6 — Major Block F: Android Continuity + Interactive 3D Assistant

Goal: make NTD97 a persistent mobile assistant interface under operating-system limits.

Planned scope:

- Android app shell;
- foreground/background task continuity;
- notification/approval surfaces;
- scheduled/retry wake path;
- process-death and reboot reconstruction;
- voice/media interfaces;
- interactive 3D avatar;
- expression, gaze, lip-sync and gesture state;
- user-authorized floating assistant surface where the OS permits it.

Exit: UI exit, process death and reboot do not destroy logical task identity; eligible work resumes through platform-approved mechanisms and is represented through an interactive 3D assistant.

## M7 — Major Block G: Paired PC Fabric

- mutual authentication;
- encrypted sessions;
- typed remote capabilities;
- artifact transfer;
- desktop observation/execution agent;
- returned-result verification.

Exit: the phone can delegate a typed task to a trusted PC without making the PC part of NTD97's identity.

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
