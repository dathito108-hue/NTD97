#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use ntd_ir::{Graph, NodeId, OpKind, ValidationError, ValueDecl, ValueId, ValueType};

use crate::tensor::{ExecutionProvider, Tensor, TensorError};

#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionError {
    InvalidGraph(ValidationError),
    MissingGraphInput(ValueId),
    MissingValue {
        node: NodeId,
        value: ValueId,
    },
    UnsupportedNode(NodeId),
    Provider {
        node: NodeId,
        source: TensorError,
    },
    OutputArity {
        node: NodeId,
        expected: usize,
        actual: usize,
    },
    OutputType {
        node: NodeId,
        value: ValueId,
    },
    OutputRank {
        node: NodeId,
        value: ValueId,
        expected: usize,
        actual: usize,
    },
}

pub struct GraphExecutor<P> {
    provider: P,
}

impl<P> GraphExecutor<P>
where
    P: ExecutionProvider,
{
    pub fn new(provider: P) -> Self {
        Self { provider }
    }

    pub fn execute(
        &self,
        graph: &Graph,
        inputs: BTreeMap<ValueId, Tensor>,
    ) -> Result<Vec<Tensor>, ExecutionError> {
        graph.validate().map_err(ExecutionError::InvalidGraph)?;

        let mut values = BTreeMap::new();
        for input in &graph.inputs {
            let tensor = inputs
                .get(&input.id)
                .ok_or(ExecutionError::MissingGraphInput(input.id))?;
            validate_decl(NodeId(u32::MAX), *input, tensor)?;
            values.insert(input.id, tensor.clone());
        }

        for node in &graph.nodes {
            let op = match &node.op {
                OpKind::Tensor(op) => *op,
                _ => return Err(ExecutionError::UnsupportedNode(node.id)),
            };

            let mut node_inputs = Vec::with_capacity(node.inputs.len());
            for value_id in &node.inputs {
                node_inputs.push(values.get(value_id).ok_or(ExecutionError::MissingValue {
                    node: node.id,
                    value: *value_id,
                })?);
            }

            let outputs = self.provider.execute(op, &node_inputs).map_err(|source| {
                ExecutionError::Provider {
                    node: node.id,
                    source,
                }
            })?;

            if outputs.len() != node.outputs.len() {
                return Err(ExecutionError::OutputArity {
                    node: node.id,
                    expected: node.outputs.len(),
                    actual: outputs.len(),
                });
            }

            for (decl, tensor) in node.outputs.iter().zip(outputs.into_iter()) {
                validate_decl(node.id, *decl, &tensor)?;
                values.insert(decl.id, tensor);
            }
        }

        graph
            .outputs
            .iter()
            .map(|value_id| {
                values
                    .get(value_id)
                    .cloned()
                    .ok_or(ExecutionError::MissingValue {
                        node: NodeId(u32::MAX),
                        value: *value_id,
                    })
            })
            .collect()
    }
}

fn validate_decl(node: NodeId, decl: ValueDecl, tensor: &Tensor) -> Result<(), ExecutionError> {
    let expected_rank = match decl.ty {
        ValueType::Scalar(_) => 0,
        ValueType::Tensor { rank, .. } => usize::from(rank),
        ValueType::Bytes | ValueType::Handle => {
            return Err(ExecutionError::OutputType {
                node,
                value: decl.id,
            })
        }
    };

    if tensor.rank() != expected_rank {
        return Err(ExecutionError::OutputRank {
            node,
            value: decl.id,
            expected: expected_rank,
            actual: tensor.rank(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::CpuReferenceProvider;
    use ntd_ir::{DType, IrVersion, Node, TensorOp};

    fn tensor_decl(id: u32, rank: u8) -> ValueDecl {
        ValueDecl {
            id: ValueId(id),
            ty: ValueType::Tensor {
                dtype: DType::F32,
                rank,
            },
        }
    }

    #[test]
    fn executes_forward_tensor_graph() {
        let graph = Graph {
            version: IrVersion::CURRENT,
            inputs: vec![tensor_decl(0, 2), tensor_decl(1, 2)],
            outputs: vec![ValueId(2)],
            nodes: vec![Node {
                id: NodeId(10),
                op: OpKind::Tensor(TensorOp::MatMul),
                inputs: vec![ValueId(0), ValueId(1)],
                outputs: vec![tensor_decl(2, 2)],
            }],
        };

        let mut inputs = BTreeMap::new();
        inputs.insert(
            ValueId(0),
            Tensor::new(vec![1, 2], vec![1.0, 2.0]).expect("lhs"),
        );
        inputs.insert(
            ValueId(1),
            Tensor::new(vec![2, 2], vec![3.0, 4.0, 5.0, 6.0]).expect("rhs"),
        );

        let result = GraphExecutor::new(CpuReferenceProvider)
            .execute(&graph, inputs)
            .expect("execute");
        assert_eq!(result[0].data(), &[13.0, 16.0]);
    }
}
