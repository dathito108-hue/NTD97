# NTD97 — Mobile General Intelligence Runtime

NTD97 is a clean-room, mobile-first AGI research project.

Its target is a single portable intelligence runtime designed for current phones: deep reasoning when required, very low interaction overhead, fast task execution, broad tool use, internet access, device/PC interaction, capability acquisition, and portable intelligence backup.

## Canonical principles

- **One runtime, one architecture.** External model formats are import sources, not permanent parallel backends.
- **Act before talking.** Executable requests should normally become: understand -> plan -> act -> verify -> concise result.
- **Adaptive reasoning.** Simple work uses a low-latency reflex budget; difficult work receives progressively deeper compute inside the same cognitive runtime.
- **Mobile-first execution.** Every plan is aware of RAM, CPU/GPU/NPU availability, battery, thermal pressure, connectivity, and Android/iOS lifecycle constraints.
- **Tool-native intelligence.** Web, files, apps, device controls, automation, and paired computers are capabilities in one graph.
- **Capability growth.** Missing capabilities may be discovered, learned, built, tested in isolation, versioned, installed, and rolled back.
- **Portable intelligence.** NTD97 defines a native Cognitive Capsule format: `.ncc97`.
- **Local-first, network-capable.** Internet and remote execution extend the system but are not mandatory architectural foundations.
- **Short communication, deep understanding.** User-facing responses should be minimal unless explanation is requested.

## Canonical stack

```text
Human / Sensors / App UI / Paired PC
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

Import adapters may accept formats such as GGUF, SafeTensors, ONNX and TFLite, then normalize them into the NTD97 runtime representation. A full backup can be exported as a self-contained or deduplicated `.ncc97` capsule.

## Repository status

**Phase 001 — Clean Mobile AGI Foundation**

This repository starts from zero. No AMPER source tree or architecture is inherited.

See:

- `docs/ARCHITECTURE.md`
- `docs/COGNITIVE_CAPSULE.md`
- `docs/ROADMAP.md`
