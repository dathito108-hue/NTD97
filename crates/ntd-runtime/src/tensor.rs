#![forbid(unsafe_code)]

use ntd_ir::{DType, TensorOp};

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self, TensorError> {
        let expected = element_count_usize(&shape)?;
        if data.len() != expected {
            return Err(TensorError::ElementCountMismatch {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { shape, data })
    }

    pub fn scalar(value: f32) -> Self {
        Self {
            shape: Vec::new(),
            data: vec![value],
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantizationParams {
    None,
    SymmetricI8 { scale: f32 },
    AffineI8 { scale: f32, zero_point: i32 },
}

#[derive(Debug, Clone, PartialEq)]
pub enum TensorLoadError {
    InvalidShape,
    ByteLengthMismatch { expected: usize, actual: usize },
    InvalidQuantization,
    UnsupportedQuantizationDType(DType),
    InvalidBool(u8),
}

#[derive(Debug, Clone, PartialEq)]
pub enum TensorError {
    InvalidShape,
    ElementCountMismatch { expected: usize, actual: usize },
    Arity { expected: usize, actual: usize },
    ShapeMismatch,
    UnsupportedOp(TensorOp),
    InvalidIndex,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct TensorLoader;

impl TensorLoader {
    pub fn load(
        dtype: DType,
        shape: &[u64],
        quantization: QuantizationParams,
        bytes: &[u8],
    ) -> Result<Tensor, TensorLoadError> {
        let shape = shape
            .iter()
            .map(|dimension| usize::try_from(*dimension).map_err(|_| TensorLoadError::InvalidShape))
            .collect::<Result<Vec<_>, _>>()?;
        let elements = element_count_usize(&shape).map_err(|_| TensorLoadError::InvalidShape)?;

        if !matches!(quantization, QuantizationParams::None) && dtype != DType::I8 {
            return Err(TensorLoadError::UnsupportedQuantizationDType(dtype));
        }

        let data = match quantization {
            QuantizationParams::None => decode_unquantized(dtype, elements, bytes)?,
            QuantizationParams::SymmetricI8 { scale } => {
                validate_scale(scale)?;
                decode_i8_quantized(elements, bytes, scale, 0)?
            }
            QuantizationParams::AffineI8 { scale, zero_point } => {
                validate_scale(scale)?;
                if !(-128..=127).contains(&zero_point) {
                    return Err(TensorLoadError::InvalidQuantization);
                }
                decode_i8_quantized(elements, bytes, scale, zero_point)?
            }
        };

        Tensor::new(shape, data).map_err(|_| TensorLoadError::InvalidShape)
    }
}

pub trait ExecutionProvider {
    fn execute(&self, op: TensorOp, inputs: &[&Tensor]) -> Result<Vec<Tensor>, TensorError>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CpuReferenceProvider;

impl ExecutionProvider for CpuReferenceProvider {
    fn execute(&self, op: TensorOp, inputs: &[&Tensor]) -> Result<Vec<Tensor>, TensorError> {
        let output = match op {
            TensorOp::Add => elementwise_binary(inputs, |a, b| a + b)?,
            TensorOp::Mul => elementwise_binary(inputs, |a, b| a * b)?,
            TensorOp::MatMul | TensorOp::QuantizedMatMul => matmul(inputs)?,
            TensorOp::RmsNorm => rms_norm(inputs)?,
            TensorOp::Softmax => softmax(inputs)?,
            TensorOp::Gather | TensorOp::RotaryPosition => {
                return Err(TensorError::UnsupportedOp(op));
            }
        };
        Ok(vec![output])
    }
}

fn decode_unquantized(
    dtype: DType,
    elements: usize,
    bytes: &[u8],
) -> Result<Vec<f32>, TensorLoadError> {
    let width = dtype_width(dtype);
    let expected = elements
        .checked_mul(width)
        .ok_or(TensorLoadError::InvalidShape)?;
    if bytes.len() != expected {
        return Err(TensorLoadError::ByteLengthMismatch {
            expected,
            actual: bytes.len(),
        });
    }

    let mut out = Vec::with_capacity(elements);
    match dtype {
        DType::F32 => {
            for chunk in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        }
        DType::F16 => {
            for chunk in bytes.chunks_exact(2) {
                out.push(f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])));
            }
        }
        DType::Bf16 => {
            for chunk in bytes.chunks_exact(2) {
                let bits = u32::from(u16::from_le_bytes([chunk[0], chunk[1]])) << 16;
                out.push(f32::from_bits(bits));
            }
        }
        DType::I8 => out.extend(bytes.iter().map(|value| (*value as i8) as f32)),
        DType::U8 => out.extend(bytes.iter().map(|value| f32::from(*value))),
        DType::I32 => {
            for chunk in bytes.chunks_exact(4) {
                out.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32);
            }
        }
        DType::I64 => {
            for chunk in bytes.chunks_exact(8) {
                out.push(i64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]) as f32);
            }
        }
        DType::Bool => {
            for value in bytes {
                match *value {
                    0 => out.push(0.0),
                    1 => out.push(1.0),
                    other => return Err(TensorLoadError::InvalidBool(other)),
                }
            }
        }
    }
    Ok(out)
}

fn decode_i8_quantized(
    elements: usize,
    bytes: &[u8],
    scale: f32,
    zero_point: i32,
) -> Result<Vec<f32>, TensorLoadError> {
    if bytes.len() != elements {
        return Err(TensorLoadError::ByteLengthMismatch {
            expected: elements,
            actual: bytes.len(),
        });
    }
    Ok(bytes
        .iter()
        .map(|value| (i32::from(*value as i8) - zero_point) as f32 * scale)
        .collect())
}

fn validate_scale(scale: f32) -> Result<(), TensorLoadError> {
    if scale.is_finite() && scale > 0.0 {
        Ok(())
    } else {
        Err(TensorLoadError::InvalidQuantization)
    }
}

fn dtype_width(dtype: DType) -> usize {
    match dtype {
        DType::F32 | DType::I32 => 4,
        DType::F16 | DType::Bf16 => 2,
        DType::I64 => 8,
        DType::I8 | DType::U8 | DType::Bool => 1,
    }
}

fn elementwise_binary(
    inputs: &[&Tensor],
    op: impl Fn(f32, f32) -> f32,
) -> Result<Tensor, TensorError> {
    require_arity(inputs, 2)?;
    if inputs[0].shape != inputs[1].shape {
        return Err(TensorError::ShapeMismatch);
    }
    let data = inputs[0]
        .data
        .iter()
        .zip(inputs[1].data.iter())
        .map(|(a, b)| op(*a, *b))
        .collect();
    Tensor::new(inputs[0].shape.clone(), data)
}

fn matmul(inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
    require_arity(inputs, 2)?;
    let lhs = inputs[0];
    let rhs = inputs[1];
    if lhs.shape.len() != 2 || rhs.shape.len() != 2 || lhs.shape[1] != rhs.shape[0] {
        return Err(TensorError::ShapeMismatch);
    }

    let m = lhs.shape[0];
    let k = lhs.shape[1];
    let n = rhs.shape[1];
    let mut data = vec![0.0; m.checked_mul(n).ok_or(TensorError::InvalidShape)?];

    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for inner in 0..k {
                sum += lhs.data[row * k + inner] * rhs.data[inner * n + col];
            }
            data[row * n + col] = sum;
        }
    }
    Tensor::new(vec![m, n], data)
}

fn rms_norm(inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
    require_arity(inputs, 1)?;
    let input = inputs[0];
    let width = *input.shape.last().ok_or(TensorError::InvalidShape)?;
    if width == 0 {
        return Err(TensorError::InvalidShape);
    }

    let mut data = Vec::with_capacity(input.data.len());
    for row in input.data.chunks_exact(width) {
        let mean_square = row.iter().map(|value| value * value).sum::<f32>() / width as f32;
        let inv_rms = 1.0 / (mean_square + 1.0e-5).sqrt();
        data.extend(row.iter().map(|value| value * inv_rms));
    }
    Tensor::new(input.shape.clone(), data)
}

fn softmax(inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
    require_arity(inputs, 1)?;
    let input = inputs[0];
    let width = *input.shape.last().ok_or(TensorError::InvalidShape)?;
    if width == 0 {
        return Err(TensorError::InvalidShape);
    }

    let mut data = Vec::with_capacity(input.data.len());
    for row in input.data.chunks_exact(width) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp = row.iter().map(|value| (*value - max).exp()).collect::<Vec<_>>();
        let sum = exp.iter().sum::<f32>();
        data.extend(exp.into_iter().map(|value| value / sum));
    }
    Tensor::new(input.shape.clone(), data)
}

fn require_arity(inputs: &[&Tensor], expected: usize) -> Result<(), TensorError> {
    if inputs.len() == expected {
        Ok(())
    } else {
        Err(TensorError::Arity {
            expected,
            actual: inputs.len(),
        })
    }
}

fn element_count_usize(shape: &[usize]) -> Result<usize, TensorError> {
    shape.iter().try_fold(1usize, |count, dimension| {
        count.checked_mul(*dimension).ok_or(TensorError::InvalidShape)
    })
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = u32::from(bits & 0x03ff);

    let value_bits = match exponent {
        0 => {
            if fraction == 0 {
                sign
            } else {
                let mut frac = fraction;
                let mut exp = -14i32;
                while frac & 0x0400 == 0 {
                    frac <<= 1;
                    exp -= 1;
                }
                frac &= 0x03ff;
                let f32_exp = u32::try_from(exp + 127).unwrap_or(0);
                sign | (f32_exp << 23) | (frac << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | (u32::from(exponent + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(value_bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_affine_i8_deterministically() {
        let tensor = TensorLoader::load(
            DType::I8,
            &[2, 2],
            QuantizationParams::AffineI8 {
                scale: 0.5,
                zero_point: -1,
            },
            &[255, 0, 1, 2],
        )
        .expect("load");
        assert_eq!(tensor.data(), &[0.0, 0.5, 1.0, 1.5]);
    }

    #[test]
    fn cpu_matmul_is_deterministic() {
        let provider = CpuReferenceProvider;
        let lhs = Tensor::new(vec![1, 2], vec![1.0, 2.0]).expect("lhs");
        let rhs = Tensor::new(vec![2, 2], vec![3.0, 4.0, 5.0, 6.0]).expect("rhs");
        let first = provider
            .execute(TensorOp::MatMul, &[&lhs, &rhs])
            .expect("first");
        let second = provider
            .execute(TensorOp::MatMul, &[&lhs, &rhs])
            .expect("second");
        assert_eq!(first, second);
        assert_eq!(first[0].data(), &[13.0, 16.0]);
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let provider = CpuReferenceProvider;
        let input = Tensor::new(vec![1, 3], vec![1.0, 2.0, 3.0]).expect("input");
        let output = provider
            .execute(TensorOp::Softmax, &[&input])
            .expect("softmax");
        let sum: f32 = output[0].data().iter().sum();
        assert!((sum - 1.0).abs() < 1.0e-6);
    }
}
