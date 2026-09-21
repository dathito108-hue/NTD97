#![forbid(unsafe_code)]

use crate::{CapsuleBuilder, ChunkStorageView, ChunkView, SectionKind};
use ntd_ir::{
    ControlOp, DType, Graph, IrVersion, MemoryOp, Node, NodeId, OpKind, StateOp, TensorOp, ToolOp,
    ValidationError, ValueDecl, ValueId, ValueType,
};

pub const IR_GRAPH_MAGIC: [u8; 6] = *b"NIR97\0";
pub const IR_GRAPH_HEADER_LEN: usize = 24;
pub const VALUE_DECL_LEN: usize = 8;
pub const NODE_HEADER_LEN: usize = 20;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IrCodecError {
    Truncated,
    InvalidMagic,
    InvalidHeader,
    InvalidValueType(u8),
    InvalidDType(u8),
    InvalidOpFamily(u8),
    InvalidOpCode { family: u8, opcode: u8 },
    InvalidUtf8,
    NonCanonicalEncoding,
    UnsupportedVersion(IrVersion),
    StructuralValidation(ValidationError),
    Overflow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphSectionError {
    WrongSection(SectionKind),
    ExternalGraph,
    Codec(IrCodecError),
}

pub fn push_graph_section(builder: &mut CapsuleBuilder, graph: &Graph) -> Result<(), IrCodecError> {
    builder.push_embedded(SectionKind::Graph, encode_graph(graph)?);
    Ok(())
}

pub fn decode_graph_section(chunk: &ChunkView<'_>) -> Result<Graph, GraphSectionError> {
    if chunk.kind != SectionKind::Graph {
        return Err(GraphSectionError::WrongSection(chunk.kind));
    }

    match chunk.storage {
        ChunkStorageView::Embedded(bytes) => decode_graph(bytes).map_err(GraphSectionError::Codec),
        ChunkStorageView::External => Err(GraphSectionError::ExternalGraph),
    }
}

pub fn encode_graph(graph: &Graph) -> Result<Vec<u8>, IrCodecError> {
    if !IrVersion::CURRENT.can_read(graph.version) {
        return Err(IrCodecError::UnsupportedVersion(graph.version));
    }

    graph
        .validate()
        .map_err(IrCodecError::StructuralValidation)?;

    let input_count = u32::try_from(graph.inputs.len()).map_err(|_| IrCodecError::Overflow)?;
    let output_count = u32::try_from(graph.outputs.len()).map_err(|_| IrCodecError::Overflow)?;
    let node_count = u32::try_from(graph.nodes.len()).map_err(|_| IrCodecError::Overflow)?;

    let mut out = Vec::new();
    out.extend_from_slice(&IR_GRAPH_MAGIC);
    push_u16(&mut out, IR_GRAPH_HEADER_LEN as u16);
    push_u16(&mut out, graph.version.major);
    push_u16(&mut out, graph.version.minor);
    push_u32(&mut out, input_count);
    push_u32(&mut out, output_count);
    push_u32(&mut out, node_count);

    for input in &graph.inputs {
        encode_value_decl(&mut out, *input);
    }

    for output in &graph.outputs {
        push_u32(&mut out, output.0);
    }

    for node in &graph.nodes {
        encode_node(&mut out, node)?;
    }

    Ok(out)
}

pub fn decode_graph(bytes: &[u8]) -> Result<Graph, IrCodecError> {
    let mut cursor = Cursor::new(bytes);

    if cursor.take(6)? != IR_GRAPH_MAGIC.as_slice() {
        return Err(IrCodecError::InvalidMagic);
    }

    if usize::from(cursor.u16()?) != IR_GRAPH_HEADER_LEN {
        return Err(IrCodecError::InvalidHeader);
    }

    let version = IrVersion {
        major: cursor.u16()?,
        minor: cursor.u16()?,
    };

    if !IrVersion::CURRENT.can_read(version) {
        return Err(IrCodecError::UnsupportedVersion(version));
    }

    let input_count = cursor.u32()?;
    let output_count = cursor.u32()?;
    let node_count = cursor.u32()?;

    let mut inputs =
        Vec::with_capacity(usize::try_from(input_count).map_err(|_| IrCodecError::Overflow)?);
    for _ in 0..input_count {
        inputs.push(decode_value_decl(&mut cursor)?);
    }

    let mut outputs =
        Vec::with_capacity(usize::try_from(output_count).map_err(|_| IrCodecError::Overflow)?);
    for _ in 0..output_count {
        outputs.push(ValueId(cursor.u32()?));
    }

    let mut nodes =
        Vec::with_capacity(usize::try_from(node_count).map_err(|_| IrCodecError::Overflow)?);
    for _ in 0..node_count {
        nodes.push(decode_node(&mut cursor)?);
    }

    if !cursor.is_finished() {
        return Err(IrCodecError::NonCanonicalEncoding);
    }

    let graph = Graph {
        version,
        inputs,
        outputs,
        nodes,
    };

    graph
        .validate()
        .map_err(IrCodecError::StructuralValidation)?;

    Ok(graph)
}

fn encode_value_decl(out: &mut Vec<u8>, value: ValueDecl) {
    push_u32(out, value.id.0);
    match value.ty {
        ValueType::Scalar(dtype) => {
            out.push(1);
            out.push(dtype_tag(dtype));
            out.push(0);
            out.push(0);
        }
        ValueType::Tensor { dtype, rank } => {
            out.push(2);
            out.push(dtype_tag(dtype));
            out.push(rank);
            out.push(0);
        }
        ValueType::Bytes => {
            out.push(3);
            out.push(0);
            out.push(0);
            out.push(0);
        }
        ValueType::Handle => {
            out.push(4);
            out.push(0);
            out.push(0);
            out.push(0);
        }
    }
}

fn decode_value_decl(cursor: &mut Cursor<'_>) -> Result<ValueDecl, IrCodecError> {
    let id = ValueId(cursor.u32()?);
    let ty_tag = cursor.u8()?;
    let dtype_tag = cursor.u8()?;
    let rank = cursor.u8()?;
    let reserved = cursor.u8()?;

    if reserved != 0 {
        return Err(IrCodecError::NonCanonicalEncoding);
    }

    let ty = match ty_tag {
        1 => {
            if rank != 0 {
                return Err(IrCodecError::NonCanonicalEncoding);
            }
            ValueType::Scalar(decode_dtype(dtype_tag)?)
        }
        2 => ValueType::Tensor {
            dtype: decode_dtype(dtype_tag)?,
            rank,
        },
        3 => {
            if dtype_tag != 0 || rank != 0 {
                return Err(IrCodecError::NonCanonicalEncoding);
            }
            ValueType::Bytes
        }
        4 => {
            if dtype_tag != 0 || rank != 0 {
                return Err(IrCodecError::NonCanonicalEncoding);
            }
            ValueType::Handle
        }
        other => return Err(IrCodecError::InvalidValueType(other)),
    };

    Ok(ValueDecl { id, ty })
}

fn encode_node(out: &mut Vec<u8>, node: &Node) -> Result<(), IrCodecError> {
    let (family, opcode, attrs) = encode_op(&node.op);
    let input_count = u32::try_from(node.inputs.len()).map_err(|_| IrCodecError::Overflow)?;
    let output_count = u32::try_from(node.outputs.len()).map_err(|_| IrCodecError::Overflow)?;
    let attr_len = u32::try_from(attrs.len()).map_err(|_| IrCodecError::Overflow)?;

    push_u32(out, node.id.0);
    out.push(family);
    out.push(opcode);
    push_u16(out, 0);
    push_u32(out, input_count);
    push_u32(out, output_count);
    push_u32(out, attr_len);

    for input in &node.inputs {
        push_u32(out, input.0);
    }

    for output in &node.outputs {
        encode_value_decl(out, *output);
    }

    out.extend_from_slice(&attrs);
    Ok(())
}

fn decode_node(cursor: &mut Cursor<'_>) -> Result<Node, IrCodecError> {
    let id = NodeId(cursor.u32()?);
    let family = cursor.u8()?;
    let opcode = cursor.u8()?;

    if cursor.u16()? != 0 {
        return Err(IrCodecError::NonCanonicalEncoding);
    }

    let input_count = cursor.u32()?;
    let output_count = cursor.u32()?;
    let attr_len = cursor.u32()?;

    let mut inputs =
        Vec::with_capacity(usize::try_from(input_count).map_err(|_| IrCodecError::Overflow)?);
    for _ in 0..input_count {
        inputs.push(ValueId(cursor.u32()?));
    }

    let mut outputs =
        Vec::with_capacity(usize::try_from(output_count).map_err(|_| IrCodecError::Overflow)?);
    for _ in 0..output_count {
        outputs.push(decode_value_decl(cursor)?);
    }

    let attrs = cursor.take(usize::try_from(attr_len).map_err(|_| IrCodecError::Overflow)?)?;
    let op = decode_op(family, opcode, attrs)?;

    Ok(Node {
        id,
        op,
        inputs,
        outputs,
    })
}

fn encode_op(op: &OpKind) -> (u8, u8, Vec<u8>) {
    let encoded = match op {
        OpKind::Tensor(op) => (1, tensor_op_tag(*op), Vec::new()),
        OpKind::State(op) => (2, state_op_tag(*op), Vec::new()),
        OpKind::Memory(op) => (3, memory_op_tag(*op), Vec::new()),
        OpKind::Control(op) => (4, control_op_tag(*op), Vec::new()),
        OpKind::Tool(ToolOp::Invoke { capability }) => {
            let bytes = capability.as_bytes().to_vec();
            (5, 1, bytes)
        }
        OpKind::Tool(ToolOp::Observe) => (5, 2, Vec::new()),
        OpKind::Tool(ToolOp::Verify) => (5, 3, Vec::new()),
    };
    encoded
}

fn decode_op(family: u8, opcode: u8, attrs: &[u8]) -> Result<OpKind, IrCodecError> {
    match family {
        1 => {
            reject_attrs(attrs)?;
            Ok(OpKind::Tensor(match opcode {
                1 => TensorOp::Add,
                2 => TensorOp::Mul,
                3 => TensorOp::MatMul,
                4 => TensorOp::QuantizedMatMul,
                5 => TensorOp::RmsNorm,
                6 => TensorOp::Softmax,
                7 => TensorOp::Gather,
                8 => TensorOp::RotaryPosition,
                9 => TensorOp::CausalAttention,
                other => {
                    return Err(IrCodecError::InvalidOpCode {
                        family,
                        opcode: other,
                    })
                }
            }))
        }
        2 => {
            reject_attrs(attrs)?;
            Ok(OpKind::State(match opcode {
                1 => StateOp::Read,
                2 => StateOp::Write,
                3 => StateOp::Checkpoint,
                other => {
                    return Err(IrCodecError::InvalidOpCode {
                        family,
                        opcode: other,
                    })
                }
            }))
        }
        3 => {
            reject_attrs(attrs)?;
            Ok(OpKind::Memory(match opcode {
                1 => MemoryOp::Retrieve,
                2 => MemoryOp::Store,
                3 => MemoryOp::Forget,
                other => {
                    return Err(IrCodecError::InvalidOpCode {
                        family,
                        opcode: other,
                    })
                }
            }))
        }
        4 => {
            reject_attrs(attrs)?;
            Ok(OpKind::Control(match opcode {
                1 => ControlOp::Select,
                2 => ControlOp::Merge,
                3 => ControlOp::Barrier,
                other => {
                    return Err(IrCodecError::InvalidOpCode {
                        family,
                        opcode: other,
                    })
                }
            }))
        }
        5 => Ok(OpKind::Tool(match opcode {
            1 => ToolOp::Invoke {
                capability: std::str::from_utf8(attrs)
                    .map_err(|_| IrCodecError::InvalidUtf8)?
                    .to_owned(),
            },
            2 => {
                reject_attrs(attrs)?;
                ToolOp::Observe
            }
            3 => {
                reject_attrs(attrs)?;
                ToolOp::Verify
            }
            other => {
                return Err(IrCodecError::InvalidOpCode {
                    family,
                    opcode: other,
                })
            }
        })),
        other => Err(IrCodecError::InvalidOpFamily(other)),
    }
}

fn reject_attrs(attrs: &[u8]) -> Result<(), IrCodecError> {
    if attrs.is_empty() {
        Ok(())
    } else {
        Err(IrCodecError::NonCanonicalEncoding)
    }
}

fn dtype_tag(dtype: DType) -> u8 {
    match dtype {
        DType::F32 => 1,
        DType::F16 => 2,
        DType::Bf16 => 3,
        DType::I8 => 4,
        DType::U8 => 5,
        DType::I32 => 6,
        DType::I64 => 7,
        DType::Bool => 8,
    }
}

fn decode_dtype(tag: u8) -> Result<DType, IrCodecError> {
    match tag {
        1 => Ok(DType::F32),
        2 => Ok(DType::F16),
        3 => Ok(DType::Bf16),
        4 => Ok(DType::I8),
        5 => Ok(DType::U8),
        6 => Ok(DType::I32),
        7 => Ok(DType::I64),
        8 => Ok(DType::Bool),
        other => Err(IrCodecError::InvalidDType(other)),
    }
}

fn tensor_op_tag(op: TensorOp) -> u8 {
    match op {
        TensorOp::Add => 1,
        TensorOp::Mul => 2,
        TensorOp::MatMul => 3,
        TensorOp::QuantizedMatMul => 4,
        TensorOp::RmsNorm => 5,
        TensorOp::Softmax => 6,
        TensorOp::Gather => 7,
        TensorOp::RotaryPosition => 8,
        TensorOp::CausalAttention => 9,
    }
}

fn state_op_tag(op: StateOp) -> u8 {
    match op {
        StateOp::Read => 1,
        StateOp::Write => 2,
        StateOp::Checkpoint => 3,
    }
}

fn memory_op_tag(op: MemoryOp) -> u8 {
    match op {
        MemoryOp::Retrieve => 1,
        MemoryOp::Store => 2,
        MemoryOp::Forget => 3,
    }
}

fn control_op_tag(op: ControlOp) -> u8 {
    match op {
        ControlOp::Select => 1,
        ControlOp::Merge => 2,
        ControlOp::Barrier => 3,
    }
}

fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], IrCodecError> {
        let end = self.offset.checked_add(len).ok_or(IrCodecError::Overflow)?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or(IrCodecError::Truncated)?;
        self.offset = end;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8, IrCodecError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, IrCodecError> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> Result<u32, IrCodecError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn is_finished(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CapsuleKind, CapsuleView};

    fn scalar(id: u32, dtype: DType) -> ValueDecl {
        ValueDecl {
            id: ValueId(id),
            ty: ValueType::Scalar(dtype),
        }
    }

    fn tensor(id: u32, dtype: DType, rank: u8) -> ValueDecl {
        ValueDecl {
            id: ValueId(id),
            ty: ValueType::Tensor { dtype, rank },
        }
    }

    fn sample_graph() -> Graph {
        Graph {
            version: IrVersion::CURRENT,
            inputs: vec![tensor(0, DType::F16, 2), scalar(1, DType::I32)],
            outputs: vec![ValueId(7)],
            nodes: vec![
                Node {
                    id: NodeId(10),
                    op: OpKind::Tensor(TensorOp::RmsNorm),
                    inputs: vec![ValueId(0)],
                    outputs: vec![tensor(2, DType::F16, 2)],
                },
                Node {
                    id: NodeId(11),
                    op: OpKind::State(StateOp::Read),
                    inputs: vec![ValueId(1)],
                    outputs: vec![ValueDecl {
                        id: ValueId(3),
                        ty: ValueType::Handle,
                    }],
                },
                Node {
                    id: NodeId(12),
                    op: OpKind::Memory(MemoryOp::Retrieve),
                    inputs: vec![ValueId(3)],
                    outputs: vec![ValueDecl {
                        id: ValueId(4),
                        ty: ValueType::Bytes,
                    }],
                },
                Node {
                    id: NodeId(13),
                    op: OpKind::Control(ControlOp::Select),
                    inputs: vec![ValueId(2), ValueId(4)],
                    outputs: vec![ValueDecl {
                        id: ValueId(5),
                        ty: ValueType::Handle,
                    }],
                },
                Node {
                    id: NodeId(14),
                    op: OpKind::Tool(ToolOp::Invoke {
                        capability: "web.search".to_owned(),
                    }),
                    inputs: vec![ValueId(5)],
                    outputs: vec![ValueDecl {
                        id: ValueId(6),
                        ty: ValueType::Bytes,
                    }],
                },
                Node {
                    id: NodeId(15),
                    op: OpKind::Tool(ToolOp::Verify),
                    inputs: vec![ValueId(6)],
                    outputs: vec![scalar(7, DType::Bool)],
                },
            ],
        }
    }

    #[test]
    fn graph_round_trips_identically() {
        let graph = sample_graph();
        let encoded = encode_graph(&graph).expect("encode");
        let decoded = decode_graph(&encoded).expect("decode");
        assert_eq!(decoded, graph);
        assert_eq!(encode_graph(&decoded).expect("re-encode"), encoded);
    }

    #[test]
    fn all_current_operation_variants_round_trip() {
        let operations = vec![
            OpKind::Tensor(TensorOp::Add),
            OpKind::Tensor(TensorOp::Mul),
            OpKind::Tensor(TensorOp::MatMul),
            OpKind::Tensor(TensorOp::QuantizedMatMul),
            OpKind::Tensor(TensorOp::RmsNorm),
            OpKind::Tensor(TensorOp::Softmax),
            OpKind::Tensor(TensorOp::Gather),
            OpKind::Tensor(TensorOp::RotaryPosition),
            OpKind::Tensor(TensorOp::CausalAttention),
            OpKind::State(StateOp::Read),
            OpKind::State(StateOp::Write),
            OpKind::State(StateOp::Checkpoint),
            OpKind::Memory(MemoryOp::Retrieve),
            OpKind::Memory(MemoryOp::Store),
            OpKind::Memory(MemoryOp::Forget),
            OpKind::Control(ControlOp::Select),
            OpKind::Control(ControlOp::Merge),
            OpKind::Control(ControlOp::Barrier),
            OpKind::Tool(ToolOp::Invoke {
                capability: "device.camera".to_owned(),
            }),
            OpKind::Tool(ToolOp::Observe),
            OpKind::Tool(ToolOp::Verify),
        ];

        let nodes = operations
            .into_iter()
            .enumerate()
            .map(|(index, op)| Node {
                id: NodeId(index as u32),
                op,
                inputs: vec![ValueId(0)],
                outputs: vec![scalar(index as u32 + 1, DType::F32)],
            })
            .collect::<Vec<_>>();

        let graph = Graph {
            version: IrVersion::CURRENT,
            inputs: vec![scalar(0, DType::F32)],
            outputs: vec![ValueId(nodes.len() as u32)],
            nodes,
        };

        let encoded = encode_graph(&graph).expect("encode");
        assert_eq!(decode_graph(&encoded).expect("decode"), graph);
    }

    #[test]
    fn all_current_dtypes_round_trip() {
        let dtypes = [
            DType::F32,
            DType::F16,
            DType::Bf16,
            DType::I8,
            DType::U8,
            DType::I32,
            DType::I64,
            DType::Bool,
        ];

        let graph = Graph {
            version: IrVersion::CURRENT,
            inputs: dtypes
                .into_iter()
                .enumerate()
                .map(|(index, dtype)| scalar(index as u32, dtype))
                .collect(),
            outputs: vec![ValueId(0)],
            nodes: Vec::new(),
        };

        let encoded = encode_graph(&graph).expect("encode");
        assert_eq!(decode_graph(&encoded).expect("decode"), graph);
    }

    #[test]
    fn graph_section_round_trips_through_ncc97_capsule() {
        let graph = sample_graph();
        let mut builder = CapsuleBuilder::new(CapsuleKind::Full, *b"NTD97-IR-CAPS-01");
        push_graph_section(&mut builder, &graph).expect("push graph");

        let bytes = builder.write().expect("write capsule");
        let capsule = CapsuleView::read(&bytes).expect("read capsule");
        let graph_chunk = capsule
            .chunks
            .iter()
            .find(|chunk| chunk.kind == SectionKind::Graph)
            .expect("graph chunk");

        assert_eq!(
            decode_graph_section(graph_chunk).expect("decode graph section"),
            graph
        );
    }

    #[test]
    fn rejects_structurally_invalid_graph_on_encode() {
        let mut graph = sample_graph();
        graph.outputs = vec![ValueId(999)];

        assert_eq!(
            encode_graph(&graph),
            Err(IrCodecError::StructuralValidation(
                ValidationError::MissingGraphOutput(ValueId(999))
            ))
        );
    }

    #[test]
    fn rejects_trailing_bytes() {
        let mut encoded = encode_graph(&sample_graph()).expect("encode");
        encoded.push(0);
        assert_eq!(
            decode_graph(&encoded),
            Err(IrCodecError::NonCanonicalEncoding)
        );
    }

    #[test]
    fn golden_vector_empty_graph_is_stable() {
        let graph = Graph {
            version: IrVersion::CURRENT,
            inputs: Vec::new(),
            outputs: Vec::new(),
            nodes: Vec::new(),
        };

        assert_eq!(
            encode_graph(&graph).expect("encode"),
            vec![
                0x4e, 0x49, 0x52, 0x39, 0x37, 0x00, 0x18, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            ]
        );
    }

    #[test]
    fn rejects_unknown_operation_code() {
        let graph = Graph {
            version: IrVersion::CURRENT,
            inputs: vec![scalar(0, DType::F32)],
            outputs: vec![ValueId(1)],
            nodes: vec![Node {
                id: NodeId(0),
                op: OpKind::Tensor(TensorOp::Add),
                inputs: vec![ValueId(0)],
                outputs: vec![scalar(1, DType::F32)],
            }],
        };

        let mut encoded = encode_graph(&graph).expect("encode");
        let node_opcode_offset = IR_GRAPH_HEADER_LEN + VALUE_DECL_LEN + 4 + 5;
        encoded[node_opcode_offset] = 0xff;

        assert_eq!(
            decode_graph(&encoded),
            Err(IrCodecError::InvalidOpCode {
                family: 1,
                opcode: 0xff,
            })
        );
    }

    #[test]
    fn fixed_lengths_are_stable() {
        assert_eq!(VALUE_DECL_LEN, 8);
        assert_eq!(NODE_HEADER_LEN, 20);
    }
}
