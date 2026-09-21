use ntd_ir::TensorOp;
use ntd_runtime::{CpuReferenceProvider, ExecutionProvider, Tensor};

#[test]
fn silu_executes_natively() {
    let input = Tensor::new(vec![2], vec![0.0, 1.0]).expect("input");
    let output = CpuReferenceProvider
        .execute(TensorOp::Silu, &[&input])
        .expect("silu")
        .remove(0);
    assert_eq!(output.shape(), &[2]);
    assert_eq!(output.data()[0], 0.0);
    assert!((output.data()[1] - 0.731_058_6).abs() < 1.0e-6);
}

#[test]
fn reshape_and_transpose_preserve_elements() {
    let input = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("input");
    let shape = Tensor::new(vec![2], vec![3.0, 2.0]).expect("shape");
    let reshaped = CpuReferenceProvider
        .execute(TensorOp::Reshape, &[&input, &shape])
        .expect("reshape")
        .remove(0);
    assert_eq!(reshaped.shape(), &[3, 2]);
    assert_eq!(reshaped.data(), input.data());

    let permutation = Tensor::new(vec![2], vec![1.0, 0.0]).expect("permutation");
    let transposed = CpuReferenceProvider
        .execute(TensorOp::Transpose, &[&input, &permutation])
        .expect("transpose")
        .remove(0);
    assert_eq!(transposed.shape(), &[3, 2]);
    assert_eq!(transposed.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn weighted_rms_norm_applies_learned_scale() {
    let input = Tensor::new(vec![1, 2], vec![3.0, 4.0]).expect("input");
    let weight = Tensor::new(vec![2], vec![2.0, 0.5]).expect("weight");
    let output = CpuReferenceProvider
        .execute(TensorOp::RmsNorm, &[&input, &weight])
        .expect("rms")
        .remove(0);

    let inv_rms = 1.0 / (((9.0f32 + 16.0) / 2.0) + 1.0e-5).sqrt();
    assert!((output.data()[0] - 3.0 * inv_rms * 2.0).abs() < 1.0e-6);
    assert!((output.data()[1] - 4.0 * inv_rms * 0.5).abs() < 1.0e-6);
}

#[test]
fn causal_attention_supports_grouped_query_heads() {
    let query = Tensor::new(vec![1, 2, 2], vec![1.0, 0.0, 0.0, 1.0]).expect("q");
    let key = Tensor::new(vec![1, 1, 2], vec![1.0, 1.0]).expect("k");
    let value = Tensor::new(vec![1, 1, 2], vec![3.0, 4.0]).expect("v");

    let output = CpuReferenceProvider
        .execute(TensorOp::CausalAttention, &[&query, &key, &value])
        .expect("attention")
        .remove(0);

    assert_eq!(output.shape(), &[1, 2, 2]);
    assert_eq!(output.data(), &[3.0, 4.0, 3.0, 4.0]);
}
