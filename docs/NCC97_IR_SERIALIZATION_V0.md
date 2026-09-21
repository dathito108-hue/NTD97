# NCC97 Native IR Serialization v0.1

Status: Phase 003C canonical wire-format contract.

This contract defines how NTD97 IR graphs and native descriptor metadata are serialized inside NCC97 sections. It does not define execution kernels.

## 1. Graph section

A Graph section begins with the 6-byte magic `NIR97\0`.

All integers are little-endian.

The fixed graph header is 24 bytes:

- magic;
- header length;
- IR major/minor version;
- graph input count;
- graph output count;
- node count.

Graph inputs are encoded as fixed 8-byte value declarations. Graph outputs are encoded as 32-bit ValueIds. Nodes follow in declared order.

## 2. Value declaration

Each value declaration is exactly 8 bytes:

- ValueId: u32;
- value-kind tag: u8;
- dtype tag: u8;
- tensor rank: u8;
- reserved: u8, required to be zero.

Value kinds:

- scalar;
- tensor;
- bytes;
- handle.

Scalar values require rank zero. Bytes/Handle require dtype and rank zero. Unknown tags and non-canonical reserved values are rejected.

## 3. Node encoding

Each node begins with a 20-byte node header:

- NodeId: u32;
- operation family: u8;
- operation code: u8;
- reserved: u16;
- input count: u32;
- output count: u32;
- attribute byte length: u32.

The header is followed by:

1. input ValueIds;
2. output value declarations;
3. operation attribute bytes.

Operation family/code values map directly to the NTD97 IR semantic enums. Unknown family/code pairs are rejected.

IR 0.2 preserves all existing tensor opcode assignments and adds tensor opcode `9 = CausalAttention`. Readers that only understand IR 0.1 reject an IR 0.2 graph through the existing version gate rather than misinterpreting the new operation.

Only Tool::Invoke currently carries attributes: its capability string is stored as canonical UTF-8 bytes. Operations without attributes must encode an attribute length of zero.

## 4. Decode validation

A graph is not accepted merely because the byte stream parses.

After decoding, the canonical NTD97 IR structural validator runs again. Therefore a persisted graph is rejected if it contains:

- incompatible IR versions;
- duplicate node IDs;
- duplicate value definitions;
- forward references to missing values;
- missing declared graph outputs.

Trailing bytes are rejected. Reserved fields must be zero.

## 5. NCC97 integration

`push_graph_section` serializes a valid Graph and inserts it as an embedded `SectionKind::Graph` chunk.

`decode_graph_section` requires:

- the chunk kind to be Graph;
- embedded bytes to be available;
- the graph codec to parse successfully;
- structural validation to pass.

The NCC97 capsule layer verifies the Graph chunk hash before graph decoding begins.

Thus the load path is:

```text
NCC97 header/index validation
        ->
Graph chunk SHA-256 validation
        ->
NIR97 graph decoding
        ->
NTD97 IR structural validation
        ->
validated Graph
```

## 6. Tensor descriptor table

Tensor execution kernels remain out of scope, but Phase 003C defines a deterministic tensor descriptor table.

Magic: `NTS97\0`.

Each tensor descriptor records:

- stable tensor ID;
- dtype;
- rank;
- logical byte length;
- ordered shape dimensions.

The schema describes tensor identity and logical shape only. It does not choose CPU/GPU/NPU kernels, quantization kernels, memory placement, tiling, or runtime scheduling.

## 7. Tokenizer and codec descriptor frames

Tokenizer and multimodal codec metadata use a generic descriptor frame.

Magic: `NDF97\0`.

The fixed frame header records:

- frame kind: Tokenizer or Codec;
- descriptor schema version;
- UTF-8 format-name length;
- opaque payload length;
- reserved zero fields.

The format name identifies the native descriptor schema carried by the payload. Phase 003C does not embed third-party runtime implementations.

## 8. Determinism

Serialization is deterministic.

For identical IR/descriptor values and ordering, the encoder must produce byte-identical output. There are no timestamps, randomized padding, host-size integers, map-order dependencies, or native-ABI struct dumps.

A golden vector for the empty IR graph fixes the initial byte-level compatibility contract.

## 9. Scope boundary

Phase 003C does not:

- execute IR;
- define tensor kernels;
- define quantized tensor payload codecs;
- import GGUF, SafeTensors, ONNX or TFLite;
- resolve external NCC97 chunks;
- perform model assimilation;
- choose hardware providers.

These remain downstream concerns.

## 10. Acceptance criteria

Phase 003C passes when:

1. every current NTD97 IR value type is encodable/decodable;
2. every current operation family is encodable/decodable;
3. Tool::Invoke capability attributes round-trip;
4. graph encoding is deterministic;
5. graph decoding rejects non-canonical/trailing data;
6. decoded graphs are structurally validated;
7. a Graph survives `Graph -> NCC97 -> reopen -> Graph` identically;
8. a fixed golden graph vector passes;
9. tensor descriptor tables round-trip deterministically;
10. Tokenizer and Codec descriptor frames round-trip;
11. format, clippy and workspace tests pass.
