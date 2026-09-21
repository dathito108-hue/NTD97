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
    NoProvider,
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
            TensorOp::Gather => gather(inputs)?,
            TensorOp::RotaryPosition => rotary_position(inputs)?,
            TensorOp::CausalAttention => causal_attention(inputs)?,
            TensorOp::Silu => silu(inputs)?,
            TensorOp::Reshape => reshape(inputs)?,
            TensorOp::Transpose => transpose(inputs)?,
            TensorOp::Linear => linear(inputs)?,
            TensorOp::PositionIds => position_ids(inputs)?,
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
    if inputs.is_empty() || inputs.len() > 3 {
        return Err(TensorError::Arity {
            expected: 1,
            actual: inputs.len(),
        });
    }
    let input = inputs[0];
    let width = *input.shape.last().ok_or(TensorError::InvalidShape)?;
    if width == 0 {
        return Err(TensorError::InvalidShape);
    }

    let weight = if inputs.len() >= 2 {
        let weight = inputs[1];
        if weight.shape != vec![width] {
            return Err(TensorError::ShapeMismatch);
        }
        Some(weight)
    } else {
        None
    };
    let epsilon = if inputs.len() == 3 {
        scalar_value(inputs[2])?
    } else {
        1.0e-5
    };
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(TensorError::InvalidShape);
    }

    let mut data = Vec::with_capacity(input.data.len());
    for row in input.data.chunks_exact(width) {
        let mean_square = row.iter().map(|value| value * value).sum::<f32>() / width as f32;
        let inv_rms = 1.0 / (mean_square + epsilon).sqrt();
        for (index, value) in row.iter().enumerate() {
            let scale = weight.map(|tensor| tensor.data[index]).unwrap_or(1.0);
            data.push(value * inv_rms * scale);
        }
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
        let exp = row
            .iter()
            .map(|value| (*value - max).exp())
            .collect::<Vec<_>>();
        let sum = exp.iter().sum::<f32>();
        data.extend(exp.into_iter().map(|value| value / sum));
    }
    Tensor::new(input.shape.clone(), data)
}

fn gather(inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
    require_arity(inputs, 2)?;
    let table = inputs[0];
    let indices = inputs[1];

    if table.shape.len() != 2 || indices.shape.len() != 1 {
        return Err(TensorError::ShapeMismatch);
    }

    let rows = table.shape[0];
    let width = table.shape[1];
    let mut data = Vec::with_capacity(
        indices
            .data
            .len()
            .checked_mul(width)
            .ok_or(TensorError::InvalidShape)?,
    );

    for raw_index in &indices.data {
        if !raw_index.is_finite() || *raw_index < 0.0 || raw_index.fract() != 0.0 {
            return Err(TensorError::InvalidIndex);
        }
        let row = *raw_index as usize;
        if row >= rows {
            return Err(TensorError::InvalidIndex);
        }
        let start = row.checked_mul(width).ok_or(TensorError::InvalidShape)?;
        let end = start.checked_add(width).ok_or(TensorError::InvalidShape)?;
        data.extend_from_slice(&table.data[start..end]);
    }

    Tensor::new(vec![indices.data.len(), width], data)
}

fn rotary_position(inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
    if inputs.len() != 2 && inputs.len() != 3 {
        return Err(TensorError::Arity {
            expected: 2,
            actual: inputs.len(),
        });
    }
    let input = inputs[0];
    let positions = inputs[1];
    let frequency_base = if inputs.len() == 3 {
        scalar_value(inputs[2])?
    } else {
        10_000.0
    };

    if input.shape.len() != 3
        || positions.shape.len() != 1
        || !frequency_base.is_finite()
        || frequency_base <= 1.0
    {
        return Err(TensorError::ShapeMismatch);
    }

    let sequence = input.shape[0];
    let heads = input.shape[1];
    let width = input.shape[2];
    if positions.data.len() != sequence || width == 0 || width % 2 != 0 {
        return Err(TensorError::ShapeMismatch);
    }

    let mut data = input.data.clone();
    for position_index in 0..sequence {
        let position = positions.data[position_index];
        if !position.is_finite() || position < 0.0 {
            return Err(TensorError::InvalidIndex);
        }

        for head in 0..heads {
            let base = (position_index * heads + head) * width;
            for pair in 0..(width / 2) {
                let even = pair * 2;
                let exponent = -(even as f32) / width as f32;
                let theta = position * frequency_base.powf(exponent);
                let cos = theta.cos();
                let sin = theta.sin();
                let left = input.data[base + even];
                let right = input.data[base + even + 1];
                data[base + even] = left * cos - right * sin;
                data[base + even + 1] = left * sin + right * cos;
            }
        }
    }

    Tensor::new(input.shape.clone(), data)
}

fn causal_attention(inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
    require_arity(inputs, 3)?;
    let query = inputs[0];
    let key = inputs[1];
    let value = inputs[2];

    if query.shape.len() != 3
        || key.shape.len() != 3
        || value.shape.len() != 3
        || key.shape != value.shape
        || query.shape[0] != key.shape[0]
        || query.shape[2] != key.shape[2]
    {
        return Err(TensorError::ShapeMismatch);
    }

    let sequence = query.shape[0];
    let query_heads = query.shape[1];
    let kv_heads = key.shape[1];
    let width = query.shape[2];
    if width == 0 || kv_heads == 0 || query_heads == 0 || query_heads % kv_heads != 0 {
        return Err(TensorError::ShapeMismatch);
    }

    let heads_per_kv = query_heads / kv_heads;
    let scale = (width as f32).sqrt();
    let mut output = vec![0.0f32; query.data.len()];

    for query_position in 0..sequence {
        for query_head in 0..query_heads {
            let kv_head = query_head / heads_per_kv;
            let query_base = (query_position * query_heads + query_head) * width;
            let mut scores = Vec::with_capacity(query_position + 1);

            for key_position in 0..=query_position {
                let key_base = (key_position * kv_heads + kv_head) * width;
                let mut dot = 0.0f32;
                for offset in 0..width {
                    dot += query.data[query_base + offset] * key.data[key_base + offset];
                }
                scores.push(dot / scale);
            }

            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut weights = scores
                .iter()
                .map(|score| (*score - max).exp())
                .collect::<Vec<_>>();
            let weight_sum = weights.iter().sum::<f32>();
            for weight in &mut weights {
                *weight /= weight_sum;
            }

            for (key_position, weight) in weights.iter().enumerate() {
                let value_base = (key_position * kv_heads + kv_head) * width;
                for offset in 0..width {
                    output[query_base + offset] += value.data[value_base + offset] * *weight;
                }
            }
        }
    }

    Tensor::new(query.shape.clone(), output)
}

fn silu(inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
    require_arity(inputs, 1)?;
    let input = inputs[0];
    let data = input
        .data
        .iter()
        .map(|value| *value / (1.0 + (-*value).exp()))
        .collect();
    Tensor::new(input.shape.clone(), data)
}

fn reshape(inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
    require_arity(inputs, 2)?;
    let input = inputs[0];
    let shape_spec = inputs[1];
    if shape_spec.rank() != 1 || shape_spec.data.is_empty() {
        return Err(TensorError::ShapeMismatch);
    }

    let mut shape = Vec::with_capacity(shape_spec.data.len());
    let mut inferred = None;
    let mut known = 1usize;
    for (index, raw) in shape_spec.data.iter().enumerate() {
        if !raw.is_finite() || raw.fract() != 0.0 {
            return Err(TensorError::InvalidShape);
        }
        if *raw == -1.0 {
            if inferred.replace(index).is_some() {
                return Err(TensorError::InvalidShape);
            }
            shape.push(1);
            continue;
        }
        if *raw <= 0.0 {
            return Err(TensorError::InvalidShape);
        }
        let dimension = *raw as usize;
        known = known
            .checked_mul(dimension)
            .ok_or(TensorError::InvalidShape)?;
        shape.push(dimension);
    }

    if let Some(index) = inferred {
        if known == 0 || input.data.len() % known != 0 {
            return Err(TensorError::ShapeMismatch);
        }
        shape[index] = input.data.len() / known;
    }
    if element_count_usize(&shape)? != input.data.len() {
        return Err(TensorError::ShapeMismatch);
    }
    Tensor::new(shape, input.data.clone())
}

fn transpose(inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
    require_arity(inputs, 2)?;
    let input = inputs[0];
    let permutation = inputs[1];
    let rank = input.rank();
    if rank == 0 || permutation.rank() != 1 || permutation.data.len() != rank {
        return Err(TensorError::ShapeMismatch);
    }

    let mut axes = Vec::with_capacity(rank);
    let mut seen = vec![false; rank];
    for raw in &permutation.data {
        if !raw.is_finite() || *raw < 0.0 || raw.fract() != 0.0 {
            return Err(TensorError::InvalidIndex);
        }
        let axis = *raw as usize;
        if axis >= rank || seen[axis] {
            return Err(TensorError::InvalidIndex);
        }
        seen[axis] = true;
        axes.push(axis);
    }

    let output_shape = axes
        .iter()
        .map(|axis| input.shape[*axis])
        .collect::<Vec<_>>();
    let input_strides = row_major_strides(&input.shape)?;
    let output_strides = row_major_strides(&output_shape)?;
    let mut data = vec![0.0f32; input.data.len()];

    for (output_index, slot) in data.iter_mut().enumerate() {
        let mut remainder = output_index;
        let mut input_index = 0usize;
        for output_axis in 0..rank {
            let stride = output_strides[output_axis];
            let coordinate = remainder / stride;
            remainder %= stride;
            let input_axis = axes[output_axis];
            input_index = input_index
                .checked_add(
                    coordinate
                        .checked_mul(input_strides[input_axis])
                        .ok_or(TensorError::InvalidShape)?,
                )
                .ok_or(TensorError::InvalidShape)?;
        }
        *slot = input.data[input_index];
    }

    Tensor::new(output_shape, data)
}

fn linear(inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
    require_arity(inputs, 2)?;
    let input = inputs[0];
    let weight = inputs[1];
    if input.rank() != 2 || weight.rank() != 2 || input.shape[1] != weight.shape[1] {
        return Err(TensorError::ShapeMismatch);
    }

    let rows = input.shape[0];
    let width = input.shape[1];
    let outputs = weight.shape[0];
    let mut data = vec![0.0f32; rows.checked_mul(outputs).ok_or(TensorError::InvalidShape)?];

    for row in 0..rows {
        for output in 0..outputs {
            let mut sum = 0.0f32;
            for inner in 0..width {
                sum += input.data[row * width + inner] * weight.data[output * width + inner];
            }
            data[row * outputs + output] = sum;
        }
    }
    Tensor::new(vec![rows, outputs], data)
}

fn position_ids(inputs: &[&Tensor]) -> Result<Tensor, TensorError> {
    require_arity(inputs, 1)?;
    let tokens = inputs[0];
    if tokens.rank() != 1 {
        return Err(TensorError::ShapeMismatch);
    }
    let data = (0..tokens.data.len()).map(|index| index as f32).collect();
    Tensor::new(vec![tokens.data.len()], data)
}

fn scalar_value(tensor: &Tensor) -> Result<f32, TensorError> {
    if tensor.rank() != 0 || tensor.data.len() != 1 {
        return Err(TensorError::ShapeMismatch);
    }
    Ok(tensor.data[0])
}

fn row_major_strides(shape: &[usize]) -> Result<Vec<usize>, TensorError> {
    let mut strides = vec![1usize; shape.len()];
    let mut running = 1usize;
    for index in (0..shape.len()).rev() {
        strides[index] = running;
        running = running
            .checked_mul(shape[index])
            .ok_or(TensorError::InvalidShape)?;
    }
    Ok(strides)
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
        count
            .checked_mul(*dimension)
            .ok_or(TensorError::InvalidShape)
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
    fn gather_selects_rows_in_token_order() {
        let provider = CpuReferenceProvider;
        let table = Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("table");
        let indices = Tensor::new(vec![2], vec![2.0, 0.0]).expect("indices");
        let output = provider
            .execute(TensorOp::Gather, &[&table, &indices])
            .expect("gather");
        assert_eq!(output[0].shape(), &[2, 2]);
        assert_eq!(output[0].data(), &[5.0, 6.0, 1.0, 2.0]);
    }

    #[test]
    fn rotary_position_at_zero_is_identity() {
        let provider = CpuReferenceProvider;
        let input = Tensor::new(vec![1, 1, 4], vec![1.0, 2.0, 3.0, 4.0]).expect("input");
        let positions = Tensor::new(vec![1], vec![0.0]).expect("positions");
        let output = provider
            .execute(TensorOp::RotaryPosition, &[&input, &positions])
            .expect("rotary");
        assert_eq!(output[0], input);
    }

    #[test]
    fn single_token_causal_attention_returns_value() {
        let provider = CpuReferenceProvider;
        let query = Tensor::new(vec![1, 1, 2], vec![1.0, 0.0]).expect("query");
        let key = Tensor::new(vec![1, 1, 2], vec![1.0, 0.0]).expect("key");
        let value = Tensor::new(vec![1, 1, 2], vec![7.0, 9.0]).expect("value");
        let output = provider
            .execute(TensorOp::CausalAttention, &[&query, &key, &value])
            .expect("attention");
        assert_eq!(output[0].data(), &[7.0, 9.0]);
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
