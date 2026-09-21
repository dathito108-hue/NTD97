#![forbid(unsafe_code)]

use std::sync::OnceLock;

use ntd_ir::TensorOp;
use ntd_runtime::{CpuTiledProvider, ExecutionProvider, Tensor};

use crate::ValidationError;

const MATRIX_WIDTH: usize = 96;

static INPUTS: OnceLock<(Tensor, Tensor)> = OnceLock::new();

pub fn run_native_validation_workload() -> Result<u64, ValidationError> {
    let (left, right) = INPUTS.get_or_init(build_inputs);
    let outputs = CpuTiledProvider::new(24)
        .execute(TensorOp::MatMul, &[left, right])
        .map_err(|error| ValidationError::RuntimeWorkload(format!("{error:?}")))?;
    let output = outputs
        .first()
        .ok_or_else(|| ValidationError::RuntimeWorkload("missing output".into()))?;

    let mut checksum = 0xcbf2_9ce4_8422_2325u64;
    for value in output.data().iter().step_by(17) {
        if !value.is_finite() {
            return Err(ValidationError::RuntimeWorkload(
                "non-finite validation output".into(),
            ));
        }
        checksum ^= u64::from(value.to_bits());
        checksum = checksum.wrapping_mul(0x1000_0000_01b3);
    }
    Ok(checksum & i64::MAX as u64)
}

fn build_inputs() -> (Tensor, Tensor) {
    let element_count = MATRIX_WIDTH * MATRIX_WIDTH;
    let left = (0..element_count)
        .map(|index| {
            let value = i32::try_from(index % 31).unwrap_or(0) - 15;
            value as f32 / 31.0
        })
        .collect::<Vec<_>>();
    let right = (0..element_count)
        .map(|index| {
            let value = i32::try_from(index % 29).unwrap_or(0) - 14;
            value as f32 / 29.0
        })
        .collect::<Vec<_>>();

    (
        Tensor::new(vec![MATRIX_WIDTH, MATRIX_WIDTH], left)
            .expect("fixed validation tensor must be valid"),
        Tensor::new(vec![MATRIX_WIDTH, MATRIX_WIDTH], right)
            .expect("fixed validation tensor must be valid"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_workload_is_deterministic_and_nonzero() {
        let first = run_native_validation_workload().expect("first");
        let second = run_native_validation_workload().expect("second");
        assert_ne!(first, 0);
        assert_eq!(first, second);
    }
}
