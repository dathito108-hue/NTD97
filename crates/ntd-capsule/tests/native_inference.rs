#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use ntd_capsule::{
    encode_native_tensor, load_native_program, push_graph_section, push_native_tensor_shard,
    push_tensor_descriptor_section, CapsuleBuilder, CapsuleKind, CapsuleView, MemoryContentStore,
    NativeTensor, QuantizationMetadata, SectionKind, TensorDescriptor,
};
use ntd_ir::{
    DType, Graph, IrVersion, Node, NodeId, OpKind, TensorOp, ValueDecl, ValueId, ValueType,
};
use ntd_runtime::{
    CpuReferenceProvider, GraphExecutor, QuantizationParams, Tensor, TensorLoader,
};

fn tensor_decl(id: u32, dtype: DType, rank: u8) -> ValueDecl {
    ValueDecl {
        id: ValueId(id),
        ty: ValueType::Tensor { dtype, rank },
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn runtime_quantization(metadata: QuantizationMetadata) -> QuantizationParams {
    match metadata {
        QuantizationMetadata::None => QuantizationParams::None,
        QuantizationMetadata::SymmetricI8 { scale_bits } => QuantizationParams::SymmetricI8 {
            scale: f32::from_bits(scale_bits),
        },
        QuantizationMetadata::AffineI8 {
            scale_bits,
            zero_point,
        } => QuantizationParams::AffineI8 {
            scale: f32::from_bits(scale_bits),
            zero_point,
        },
    }
}

#[test]
fn ncc97_to_cpu_inference_is_deterministic_end_to_end() {
    let graph = Graph {
        version: IrVersion::CURRENT,
        inputs: vec![
            tensor_decl(0, DType::F32, 2),
            tensor_decl(1, DType::I8, 2),
            tensor_decl(2, DType::F32, 2),
        ],
        outputs: vec![ValueId(4)],
        nodes: vec![
            Node {
                id: NodeId(10),
                op: OpKind::Tensor(TensorOp::QuantizedMatMul),
                inputs: vec![ValueId(0), ValueId(1)],
                outputs: vec![tensor_decl(3, DType::F32, 2)],
            },
            Node {
                id: NodeId(11),
                op: OpKind::Tensor(TensorOp::Add),
                inputs: vec![ValueId(3), ValueId(2)],
                outputs: vec![tensor_decl(4, DType::F32, 2)],
            },
        ],
    };

    let weight = NativeTensor {
        descriptor: TensorDescriptor {
            id: 7,
            dtype: DType::I8,
            shape: vec![2, 2],
            byte_len: 4,
        },
        graph_value: Some(ValueId(1)),
        quantization: QuantizationMetadata::symmetric_i8(1.0).expect("weight quantization"),
        payload: vec![3, 4, 5, 6],
    };

    let bias = NativeTensor {
        descriptor: TensorDescriptor {
            id: 8,
            dtype: DType::F32,
            shape: vec![1, 2],
            byte_len: 8,
        },
        graph_value: Some(ValueId(2)),
        quantization: QuantizationMetadata::None,
        payload: f32_bytes(&[0.5, -1.0]),
    };

    let weight_shard = encode_native_tensor(&weight).expect("encode weight");
    let mut store = MemoryContentStore::default();
    let weight_hash = store.insert(weight_shard.clone());

    let mut builder = CapsuleBuilder::new(CapsuleKind::Thin, *b"NTD97-E2E-CPU-01");
    push_graph_section(&mut builder, &graph).expect("graph");
    push_tensor_descriptor_section(
        &mut builder,
        &[weight.descriptor.clone(), bias.descriptor.clone()],
    )
    .expect("descriptors");
    builder.push_external(
        SectionKind::Tensors,
        u64::try_from(weight_shard.len()).expect("weight shard length"),
        weight_hash,
    );
    push_native_tensor_shard(&mut builder, &bias).expect("bias shard");

    let capsule_bytes = builder.write().expect("write capsule");
    let capsule = CapsuleView::read(&capsule_bytes).expect("read capsule");
    let program = load_native_program(&capsule, &store).expect("load native program");

    let mut inputs = BTreeMap::new();
    inputs.insert(
        ValueId(0),
        Tensor::new(vec![1, 2], vec![1.0, 2.0]).expect("runtime input"),
    );

    for native in &program.tensors {
        let value_id = native.graph_value.expect("native graph input binding");
        let tensor = TensorLoader::load(
            native.descriptor.dtype,
            &native.descriptor.shape,
            runtime_quantization(native.quantization),
            &native.payload,
        )
        .expect("materialize native tensor");
        inputs.insert(value_id, tensor);
    }

    let executor = GraphExecutor::new(CpuReferenceProvider);
    let first = executor
        .execute(&program.graph, inputs.clone())
        .expect("first inference");
    let second = executor
        .execute(&program.graph, inputs)
        .expect("second inference");

    assert_eq!(first, second);
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].shape(), &[1, 2]);
    assert_eq!(first[0].data(), &[13.5, 15.0]);
}
