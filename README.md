# NTD97 — Mobile General Intelligence Runtime

NTD97 is a clean-room, sovereign mobile AGI model and runtime designed to make the phone itself the primary home of the intelligence.

Its target is a single portable intelligence runtime designed for current phones: deep reasoning when required, very low interaction overhead, fast task execution, broad tool use, internet access, device/PC interaction, capability acquisition, and portable intelligence backup.

## Canonical principles

- **NTD97 is the model.** NTD97 is one independent mobile AGI identity, not an app that switches among external models.
- **One runtime, one architecture.** External models, weights, knowledge and skills are import sources, not permanent parallel backends.
- **Native assimilation.** Successfully imported intelligence is converted into NTD97 IR + `.ncc97` native state and becomes part of NTD97 immediately after validation and atomic commit. The source runtime is not required afterward.
- **Sovereign-first.** Core reasoning, memory, planning, execution and recovery must work without third-party AI APIs, cloud control planes, hosted model services or mandatory accounts.
- **Act before talking.** Executable requests should normally become: understand -> plan -> act -> verify -> concise result.
- **Adaptive reasoning.** Simple work uses a low-latency reflex budget; difficult work receives progressively deeper compute inside the same cognitive runtime.
- **Mobile-first execution.** Every plan is aware of RAM, CPU/GPU/NPU availability, battery, thermal pressure, connectivity, and Android/iOS lifecycle constraints.
- **Tool-native intelligence.** Web, files, apps, device controls, automation, and paired computers are capabilities in one graph.
- **Capability growth.** Missing capabilities may be discovered, learned, built, tested in isolation, versioned, installed, and rolled back.
- **Portable intelligence.** NTD97 defines a native Cognitive Capsule format: `.ncc97`.
- **Local-first, network-capable.** Internet and remote execution extend the system but are not mandatory architectural foundations.
- **Short communication, deep understanding.** User-facing responses should be minimal unless explanation is requested.
- **Embodied 3D assistant.** A persistent interactive 3D character is a first-class interface, including an in-app scene and a user-authorized floating surface where the OS permits it.
- **24/7 logical continuity.** NTD97 preserves goals, plans and checkpoints across UI exit, process death and reboot, resuming through platform-allowed background/foreground mechanisms.

## Canonical stack

```text
Human / Sensors / 3D Assistant / App UI / Paired PC
               |
        Interaction Fabric
               |
          Intent Compiler
               |
      Adaptive Cognitive Loop
        /                 \
  Reflex Budget       Deep Budget
        \                 /
          Task Graph IR
               |
       Executor + Verifier
               |
     Capability / Tool Graph
       /    |      |      \
     Web  Device  Files  PC Link
               |
      Sovereign Intelligence Kernel (SIK97)
               |
           NTD97 IR
               |
      Adaptive Mobile Runtime
               |
  NTD97 Cognitive Capsule (.ncc97)
```

## Native intelligence format

`.ncc97` is the canonical NTD97 container. It is designed to carry or reference:

- tensor shards / quantized weights;
- execution graph;
- tokenizer and multimodal codecs;
- reasoning/router metadata;
- adapters and learned deltas;
- capability descriptors;
- memory schema and optional memory state;
- device execution profiles;
- provenance, integrity hashes, signatures, and version metadata.

Import adapters may accept formats such as GGUF, SafeTensors, ONNX and TFLite, plus structured knowledge and skill packages, then assimilate supported semantics into NTD97 IR and native capsule state. A full backup can be exported as a self-contained or deduplicated `.ncc97` capsule.

## Repository status

**Major Block F — Android Continuity + Interactive 3D Assistant (core/shell implemented, APK binding gate pending)**

NTD97 now has a portable mobile shell around the same sovereign cognition/action identity:

```text
SIK97 cognition + TAF97 actions + capability snapshot
  -> deterministic MCS97 mobile continuity
  -> OS-aware wake policy
  -> Android AtomicFile / JobScheduler / reboot / foreground paths
  -> approval + local PCM voice
  -> avatar state
  -> in-app / floating OpenGL ES 3D surface
```

Process-death/reboot tests prove a committed action is not replayed after restore. Android source fails closed when no packaged local runtime host exists; Android Gradle/native-provider packaging remains the current M6 release gate.
This repository starts from zero. No AMPER source tree or architecture is inherited.

See:

- `docs/ARCHITECTURE.md`
- `docs/ARCHITECTURE_FREEZE.md`
- `docs/IR_V0.md`
- `docs/NCC97_BINARY_V0.md`
- `docs/NCC97_IR_SERIALIZATION_V0.md`
- `docs/NATIVE_EXECUTION_FOUNDATION.md`
- `docs/NATIVE_GENERATIVE_RUNTIME.md`
- `docs/ADAPTIVE_MOBILE_COMPUTE_RUNTIME.md`
- `docs/COGNITIVE_RUNTIME_SOVEREIGN_MEMORY.md`
- `docs/TOOL_INTERNET_DEVICE_ACTION_FABRIC.md`
- `docs/ANDROID_CONTINUITY_3D_ASSISTANT.md`
- `docs/COGNITIVE_CAPSULE.md`
- `docs/SOVEREIGN_MODEL.md`
- `docs/CONTINUITY_3D.md`
- `docs/ROADMAP.md`
