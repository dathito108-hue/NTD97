#![forbid(unsafe_code)]

use std::collections::BTreeSet;

pub const IR_MAJOR: u16 = 0;
pub const IR_MINOR: u16 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IrVersion {
    pub major: u16,
    pub minor: u16,
}

impl IrVersion {
    pub const CURRENT: Self = Self {
        major: IR_MAJOR,
        minor: IR_MINOR,
    };

    pub fn can_read(self, other: Self) -> bool {
        self.major == other.major && other.minor <= self.minor
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValueId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    Bf16,
    I8,
    U8,
    I32,
    I64,
    Bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    Scalar(DType),
    Tensor { dtype: DType, rank: u8 },
    Bytes,
    Handle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValueDecl {
    pub id: ValueId,
    pub ty: ValueType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorOp {
    Add,
    Mul,
    MatMul,
    QuantizedMatMul,
    RmsNorm,
    Softmax,
    Gather,
    RotaryPosition,
    CausalAttention,
    Silu,
    Reshape,
    Transpose,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateOp {
    Read,
    Write,
    Checkpoint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOp {
    Retrieve,
    Store,
    Forget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlOp {
    Select,
    Merge,
    Barrier,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolOp {
    Invoke { capability: String },
    Observe,
    Verify,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpKind {
    Tensor(TensorOp),
    State(StateOp),
    Memory(MemoryOp),
    Control(ControlOp),
    Tool(ToolOp),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    pub id: NodeId,
    pub op: OpKind,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueDecl>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Graph {
    pub version: IrVersion,
    pub inputs: Vec<ValueDecl>,
    pub outputs: Vec<ValueId>,
    pub nodes: Vec<Node>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    UnsupportedVersion(IrVersion),
    DuplicateNodeId(NodeId),
    DuplicateValue(ValueId),
    MissingInput { node: NodeId, value: ValueId },
    MissingGraphOutput(ValueId),
}

impl Graph {
    pub fn validate(&self) -> Result<(), ValidationError> {
        if !IrVersion::CURRENT.can_read(self.version) {
            return Err(ValidationError::UnsupportedVersion(self.version));
        }

        let mut node_ids = BTreeSet::new();
        let mut values = BTreeSet::new();

        for input in &self.inputs {
            if !values.insert(input.id) {
                return Err(ValidationError::DuplicateValue(input.id));
            }
        }

        for node in &self.nodes {
            if !node_ids.insert(node.id) {
                return Err(ValidationError::DuplicateNodeId(node.id));
            }

            for input in &node.inputs {
                if !values.contains(input) {
                    return Err(ValidationError::MissingInput {
                        node: node.id,
                        value: *input,
                    });
                }
            }

            for output in &node.outputs {
                if !values.insert(output.id) {
                    return Err(ValidationError::DuplicateValue(output.id));
                }
            }
        }

        for output in &self.outputs {
            if !values.contains(output) {
                return Err(ValidationError::MissingGraphOutput(*output));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(id: u32) -> ValueDecl {
        ValueDecl {
            id: ValueId(id),
            ty: ValueType::Scalar(DType::F32),
        }
    }

    #[test]
    fn validates_forward_graph() {
        let graph = Graph {
            version: IrVersion::CURRENT,
            inputs: vec![scalar(0), scalar(1)],
            outputs: vec![ValueId(2)],
            nodes: vec![Node {
                id: NodeId(0),
                op: OpKind::Tensor(TensorOp::Add),
                inputs: vec![ValueId(0), ValueId(1)],
                outputs: vec![scalar(2)],
            }],
        };

        assert_eq!(graph.validate(), Ok(()));
    }

    #[test]
    fn rejects_dangling_input() {
        let graph = Graph {
            version: IrVersion::CURRENT,
            inputs: vec![scalar(0)],
            outputs: vec![ValueId(2)],
            nodes: vec![Node {
                id: NodeId(0),
                op: OpKind::Tensor(TensorOp::Add),
                inputs: vec![ValueId(0), ValueId(99)],
                outputs: vec![scalar(2)],
            }],
        };

        assert_eq!(
            graph.validate(),
            Err(ValidationError::MissingInput {
                node: NodeId(0),
                value: ValueId(99),
            })
        );
    }

    #[test]
    fn rejects_duplicate_node_id() {
        let graph = Graph {
            version: IrVersion::CURRENT,
            inputs: vec![scalar(0)],
            outputs: vec![ValueId(2)],
            nodes: vec![
                Node {
                    id: NodeId(7),
                    op: OpKind::State(StateOp::Read),
                    inputs: vec![ValueId(0)],
                    outputs: vec![scalar(1)],
                },
                Node {
                    id: NodeId(7),
                    op: OpKind::State(StateOp::Write),
                    inputs: vec![ValueId(1)],
                    outputs: vec![scalar(2)],
                },
            ],
        };

        assert_eq!(
            graph.validate(),
            Err(ValidationError::DuplicateNodeId(NodeId(7)))
        );
    }

    #[test]
    fn rejects_missing_graph_output() {
        let graph = Graph {
            version: IrVersion::CURRENT,
            inputs: vec![scalar(0)],
            outputs: vec![ValueId(99)],
            nodes: Vec::new(),
        };

        assert_eq!(
            graph.validate(),
            Err(ValidationError::MissingGraphOutput(ValueId(99)))
        );
    }

    #[test]
    fn rejects_duplicate_value_definition() {
        let graph = Graph {
            version: IrVersion::CURRENT,
            inputs: vec![scalar(0)],
            outputs: vec![ValueId(0)],
            nodes: vec![Node {
                id: NodeId(0),
                op: OpKind::State(StateOp::Checkpoint),
                inputs: vec![ValueId(0)],
                outputs: vec![scalar(0)],
            }],
        };

        assert_eq!(
            graph.validate(),
            Err(ValidationError::DuplicateValue(ValueId(0)))
        );
    }
}
