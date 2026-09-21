#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use ntd_capsule::{
    load_native_generative_program, push_graph_section, push_native_tensor_shard,
    push_native_tokenizer_section, push_tensor_descriptor_section, CapsuleBuilder, CapsuleKind,
    CapsuleView, MemoryContentStore, NativeTensor, NativeTokenizerDescriptor, QuantizationMetadata,
    TensorDescriptor,
};
use ntd_ir::{
    DType, Graph, IrVersion, Node, NodeId, OpKind, TensorOp, ValueDecl, ValueId, ValueType,
};
use ntd_runtime::{
    CpuReferenceProvider, DistributionKind, GenerationConfig, GraphGenerator, QuantizationParams,
    SamplingMode, TensorLoader, VocabularyTokenizer,
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
fn full_ncc97_package_generates_text_without_external_model_runtime() {
    let graph = Graph {
        version: IrVersion::CURRENT,
        inputs: vec![tensor_decl(0, DType::I32, 1), tensor_decl(1, DType::F32, 2)],
        outputs: vec![ValueId(3)],
        nodes: vec![
            Node {
                id: NodeId(0),
                op: OpKind::Tensor(TensorOp::Gather),
                inputs: vec![ValueId(1), ValueId(0)],
                outputs: vec![tensor_decl(2, DType::F32, 2)],
            },
            Node {
                id: NodeId(1),
                op: OpKind::Tensor(TensorOp::Softmax),
                inputs: vec![ValueId(2)],
                outputs: vec![tensor_decl(3, DType::F32, 2)],
            },
        ],
    };

    let transitions = [
        0.0, 12.0, 0.0, 0.0, //
        0.0, 0.0, 12.0, 0.0, //
        0.0, 0.0, 0.0, 12.0, //
        0.0, 0.0, 0.0, 12.0,
    ];
    let transition_tensor = NativeTensor {
        descriptor: TensorDescriptor {
            id: 100,
            dtype: DType::F32,
            shape: vec![4, 4],
            byte_len: 64,
        },
        graph_value: Some(ValueId(1)),
        quantization: QuantizationMetadata::None,
        payload: f32_bytes(&transitions),
    };

    let tokenizer_descriptor = NativeTokenizerDescriptor {
        tokens: vec![
            b"a".to_vec(),
            b"b".to_vec(),
            b"c".to_vec(),
            b"<eos>".to_vec(),
        ],
        bos_token: None,
        eos_token: Some(3),
        unknown_token: None,
    };

    let mut builder = CapsuleBuilder::new(CapsuleKind::Full, *b"NTD97-GEN-CPU-01");
    push_graph_section(&mut builder, &graph).expect("graph");
    push_tensor_descriptor_section(
        &mut builder,
        std::slice::from_ref(&transition_tensor.descriptor),
    )
    .expect("tensor descriptors");
    push_native_tensor_shard(&mut builder, &transition_tensor).expect("transition shard");
    push_native_tokenizer_section(&mut builder, &tokenizer_descriptor).expect("tokenizer");

    let bytes = builder.write().expect("write capsule");
    let view = CapsuleView::read(&bytes).expect("read capsule");
    let package =
        load_native_generative_program(&view, &MemoryContentStore::default()).expect("package");

    let tokenizer = VocabularyTokenizer::new(
        package.tokenizer.tokens.clone(),
        package.tokenizer.bos_token,
        package.tokenizer.eos_token,
        package.tokenizer.unknown_token,
    )
    .expect("runtime tokenizer");

    let mut static_inputs = BTreeMap::new();
    for native in &package.program.tensors {
        let graph_value = native.graph_value.expect("weight graph binding");
        let tensor = TensorLoader::load(
            native.descriptor.dtype,
            &native.descriptor.shape,
            runtime_quantization(native.quantization),
            &native.payload,
        )
        .expect("materialize tensor");
        static_inputs.insert(graph_value, tensor);
    }

    let generator = GraphGenerator::new(
        package.program.graph,
        CpuReferenceProvider,
        static_inputs,
        ValueId(0),
        0,
        tokenizer.vocab_size(),
    )
    .expect("generator");

    let first = generator
        .generate_text(
            &tokenizer,
            "a",
            false,
            GenerationConfig {
                max_new_tokens: 8,
                context_limit: 4,
                eos_token: None,
                distribution: DistributionKind::Probabilities,
                sampling: SamplingMode::Greedy,
            },
        )
        .expect("first generation");
    let second = generator
        .generate_text(
            &tokenizer,
            "a",
            false,
            GenerationConfig {
                max_new_tokens: 8,
                context_limit: 4,
                eos_token: None,
                distribution: DistributionKind::Probabilities,
                sampling: SamplingMode::Greedy,
            },
        )
        .expect("second generation");

    assert_eq!(first, second);
    assert_eq!(first.generated_tokens, vec![1, 2, 3]);
    assert_eq!(first.text, "bc");
}
