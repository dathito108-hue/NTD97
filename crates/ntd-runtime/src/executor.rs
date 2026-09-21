#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

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
    Resolve {
        value: ValueId,
        source: TensorResolveError,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorResolveError {
    Unavailable,
    Invalid,
}

pub trait TensorResolver {
    fn resolve(&self, value: ValueId) -> Result<Option<Tensor>, TensorResolveError>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct EmptyTensorResolver;

impl TensorResolver for EmptyTensorResolver {
    fn resolve(&self, _value: ValueId) -> Result<Option<Tensor>, TensorResolveError> {
        Ok(None)
    }
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
        self.execute_with_resolver(graph, inputs, &EmptyTensorResolver)
    }

    pub fn execute_with_resolver<R: TensorResolver>(
        &self,
        graph: &Graph,
        inputs: BTreeMap<ValueId, Tensor>,
        resolver: &R,
    ) -> Result<Vec<Tensor>, ExecutionError> {
        graph.validate().map_err(ExecutionError::InvalidGraph)?;

        let input_decls = graph
            .inputs
            .iter()
            .map(|decl| (decl.id, *decl))
            .collect::<BTreeMap<_, _>>();
        let output_ids = graph.outputs.iter().copied().collect::<BTreeSet<_>>();
        let mut remaining_uses = BTreeMap::<ValueId, usize>::new();
        for node in &graph.nodes {
            for value in &node.inputs {
                *remaining_uses.entry(*value).or_default() += 1;
            }
        }

        let mut values = BTreeMap::new();
        for (value_id, tensor) in inputs {
            if let Some(decl) = input_decls.get(&value_id).copied() {
                validate_decl(NodeId(u32::MAX), decl, &tensor)?;
                values.insert(value_id, tensor);
            }
        }

        for node in &graph.nodes {
            let op = match &node.op {
                OpKind::Tensor(op) => *op,
                _ => return Err(ExecutionError::UnsupportedNode(node.id)),
            };

            for value_id in &node.inputs {
                if values.contains_key(value_id) {
                    continue;
                }
                let Some(decl) = input_decls.get(value_id).copied() else {
                    return Err(ExecutionError::MissingValue {
                        node: node.id,
                        value: *value_id,
                    });
                };
                let tensor = resolver
                    .resolve(*value_id)
                    .map_err(|source| ExecutionError::Resolve {
                        value: *value_id,
                        source,
                    })?
                    .ok_or(ExecutionError::MissingGraphInput(*value_id))?;
                validate_decl(NodeId(u32::MAX), decl, &tensor)?;
                values.insert(*value_id, tensor);
            }

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
            drop(node_inputs);

            if outputs.len() != node.outputs.len() {
                return Err(ExecutionError::OutputArity {
                    node: node.id,
                    expected: node.outputs.len(),
                    actual: outputs.len(),
                });
            }

            for value_id in &node.inputs {
                if let Some(remaining) = remaining_uses.get_mut(value_id) {
                    *remaining = (*remaining).saturating_sub(1);
                    if *remaining == 0 && !output_ids.contains(value_id) {
                        values.remove(value_id);
                    }
                }
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
    fn resolves_graph_inputs_on_demand() {
        use std::cell::Cell;

        struct Resolver {
            calls: Cell<usize>,
        }

        impl TensorResolver for Resolver {
            fn resolve(&self, value: ValueId) -> Result<Option<Tensor>, TensorResolveError> {
                self.calls.set(self.calls.get() + 1);
                if value == ValueId(1) {
                    Tensor::new(vec![2, 2], vec![3.0, 4.0, 5.0, 6.0])
                        .map(Some)
                        .map_err(|_| TensorResolveError::Invalid)
                } else {
                    Ok(None)
                }
            }
        }

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
        let resolver = Resolver {
            calls: Cell::new(0),
        };

        let result = GraphExecutor::new(CpuReferenceProvider)
            .execute_with_resolver(&graph, inputs, &resolver)
            .expect("execute");
        assert_eq!(result[0].data(), &[13.0, 16.0]);
        assert_eq!(resolver.calls.get(), 1);
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
