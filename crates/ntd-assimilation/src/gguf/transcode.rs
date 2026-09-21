#![forbid(unsafe_code)]

use ntd_ir::DType;

use super::{GgufByteSource, GgufError, GgufModel, GgufTensorInfo, SliceGgufSource};

const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_F16: u32 = 1;
const GGML_TYPE_Q4_0: u32 = 2;
const GGML_TYPE_Q8_0: u32 = 8;
const GGML_TYPE_Q4_K: u32 = 12;
const GGML_TYPE_Q5_K: u32 = 13;
const GGML_TYPE_Q6_K: u32 = 14;
const GGML_TYPE_BF16: u32 = 30;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscodedTensor {
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GgmlLayout {
    block_elements: u64,
    block_bytes: u64,
}

pub fn ggml_tensor_byte_len(tensor: &GgufTensorInfo) -> Result<u64, GgufError> {
    let layout = ggml_layout(tensor.ggml_type)?;
    let first = *tensor.dimensions.first().ok_or(GgufError::InvalidTensor)?;
    if first == 0 || first % layout.block_elements != 0 {
        return Err(GgufError::InvalidTensor);
    }

    let rows = tensor
        .dimensions
        .iter()
        .skip(1)
        .try_fold(1u64, |product, dimension| {
            product.checked_mul(*dimension).ok_or(GgufError::Overflow)
        })?;
    let blocks_per_row = first / layout.block_elements;
    rows.checked_mul(blocks_per_row)
        .and_then(|blocks| blocks.checked_mul(layout.block_bytes))
        .ok_or(GgufError::Overflow)
}

pub fn gguf_tensor_bytes<'a>(
    file: &'a [u8],
    model: &GgufModel,
    tensor: &GgufTensorInfo,
) -> Result<&'a [u8], GgufError> {
    let file_len = u64::try_from(file.len()).map_err(|_| GgufError::LimitExceeded)?;
    if file_len != model.file_len {
        return Err(GgufError::InvalidTensor);
    }

    let len = ggml_tensor_byte_len(tensor)?;
    let start = model
        .data_offset
        .checked_add(tensor.data_offset)
        .ok_or(GgufError::Overflow)?;
    let end = start.checked_add(len).ok_or(GgufError::Overflow)?;
    if end > file_len {
        return Err(GgufError::Truncated);
    }

    let start = usize::try_from(start).map_err(|_| GgufError::LimitExceeded)?;
    let end = usize::try_from(end).map_err(|_| GgufError::LimitExceeded)?;
    file.get(start..end).ok_or(GgufError::Truncated)
}

pub fn gguf_tensor_bytes_from_source(
    source: &dyn GgufByteSource,
    model: &GgufModel,
    tensor: &GgufTensorInfo,
) -> Result<Vec<u8>, GgufError> {
    let file_len = source.byte_len()?;
    if file_len != model.file_len {
        return Err(GgufError::InvalidTensor);
    }

    let len = ggml_tensor_byte_len(tensor)?;
    let start = model
        .data_offset
        .checked_add(tensor.data_offset)
        .ok_or(GgufError::Overflow)?;
    let end = start.checked_add(len).ok_or(GgufError::Overflow)?;
    if end > file_len {
        return Err(GgufError::Truncated);
    }

    source.read_exact_at(
        start,
        usize::try_from(len).map_err(|_| GgufError::LimitExceeded)?,
    )
}

pub fn transcode_tensor(
    file: &[u8],
    model: &GgufModel,
    tensor: &GgufTensorInfo,
    transpose_2d: bool,
) -> Result<TranscodedTensor, GgufError> {
    transcode_tensor_from_source(&SliceGgufSource::new(file), model, tensor, transpose_2d)
}

pub fn transcode_tensor_from_source(
    source: &dyn GgufByteSource,
    model: &GgufModel,
    tensor: &GgufTensorInfo,
    transpose_2d: bool,
) -> Result<TranscodedTensor, GgufError> {
    if tensor.dimensions.is_empty() || tensor.dimensions.len() > 2 {
        return Err(GgufError::InvalidTensor);
    }
    if transpose_2d && tensor.dimensions.len() != 2 {
        return Err(GgufError::InvalidTensor);
    }

    let bytes = gguf_tensor_bytes_from_source(source, model, tensor)?;
    transcode_tensor_payload(tensor, &bytes, transpose_2d)
}

fn transcode_tensor_payload(
    tensor: &GgufTensorInfo,
    bytes: &[u8],
    transpose_2d: bool,
) -> Result<TranscodedTensor, GgufError> {
    match tensor.ggml_type {
        GGML_TYPE_F32 => transcode_unquantized(tensor, bytes, DType::F32, 4, transpose_2d),
        GGML_TYPE_F16 => transcode_unquantized(tensor, bytes, DType::F16, 2, transpose_2d),
        GGML_TYPE_BF16 => transcode_unquantized(tensor, bytes, DType::Bf16, 2, transpose_2d),
        GGML_TYPE_Q4_0 => {
            let values = decode_q4_0(tensor, bytes)?;
            transcode_dequantized(tensor, values, transpose_2d)
        }
        GGML_TYPE_Q8_0 => {
            let values = decode_q8_0(tensor, bytes)?;
            transcode_dequantized(tensor, values, transpose_2d)
        }
        GGML_TYPE_Q4_K => {
            let values = decode_q4_k(tensor, bytes)?;
            transcode_dequantized(tensor, values, transpose_2d)
        }
        GGML_TYPE_Q5_K => {
            let values = decode_q5_k(tensor, bytes)?;
            transcode_dequantized(tensor, values, transpose_2d)
        }
        GGML_TYPE_Q6_K => {
            let values = decode_q6_k(tensor, bytes)?;
            transcode_dequantized(tensor, values, transpose_2d)
        }
        other => Err(GgufError::UnsupportedTensorType(other)),
    }
}

pub fn ggml_type_supported(ggml_type: u32) -> bool {
    matches!(
        ggml_type,
        GGML_TYPE_F32
            | GGML_TYPE_F16
            | GGML_TYPE_Q4_0
            | GGML_TYPE_Q8_0
            | GGML_TYPE_Q4_K
            | GGML_TYPE_Q5_K
            | GGML_TYPE_Q6_K
            | GGML_TYPE_BF16
    )
}

fn ggml_layout(ggml_type: u32) -> Result<GgmlLayout, GgufError> {
    match ggml_type {
        GGML_TYPE_F32 => Ok(GgmlLayout {
            block_elements: 1,
            block_bytes: 4,
        }),
        GGML_TYPE_F16 | GGML_TYPE_BF16 => Ok(GgmlLayout {
            block_elements: 1,
            block_bytes: 2,
        }),
        GGML_TYPE_Q4_0 => Ok(GgmlLayout {
            block_elements: 32,
            block_bytes: 18,
        }),
        GGML_TYPE_Q8_0 => Ok(GgmlLayout {
            block_elements: 32,
            block_bytes: 34,
        }),
        GGML_TYPE_Q4_K => Ok(GgmlLayout {
            block_elements: 256,
            block_bytes: 144,
        }),
        GGML_TYPE_Q5_K => Ok(GgmlLayout {
            block_elements: 256,
            block_bytes: 176,
        }),
        GGML_TYPE_Q6_K => Ok(GgmlLayout {
            block_elements: 256,
            block_bytes: 210,
        }),
        other => Err(GgufError::UnsupportedTensorType(other)),
    }
}

fn transcode_unquantized(
    tensor: &GgufTensorInfo,
    bytes: &[u8],
    dtype: DType,
    element_width: usize,
    transpose_2d: bool,
) -> Result<TranscodedTensor, GgufError> {
    let shape = native_shape(tensor, transpose_2d)?;
    let payload = if transpose_2d {
        transpose_elements(
            bytes,
            usize::try_from(tensor.dimensions[1]).map_err(|_| GgufError::LimitExceeded)?,
            usize::try_from(tensor.dimensions[0]).map_err(|_| GgufError::LimitExceeded)?,
            element_width,
        )?
    } else {
        bytes.to_vec()
    };
    Ok(TranscodedTensor {
        dtype,
        shape,
        payload,
    })
}

fn transcode_dequantized(
    tensor: &GgufTensorInfo,
    values: Vec<f32>,
    transpose_2d: bool,
) -> Result<TranscodedTensor, GgufError> {
    let shape = native_shape(tensor, transpose_2d)?;
    let values = if transpose_2d {
        transpose_values(
            &values,
            usize::try_from(tensor.dimensions[1]).map_err(|_| GgufError::LimitExceeded)?,
            usize::try_from(tensor.dimensions[0]).map_err(|_| GgufError::LimitExceeded)?,
        )?
    } else {
        values
    };

    let mut payload = Vec::with_capacity(values.len().checked_mul(4).ok_or(GgufError::Overflow)?);
    for value in values {
        if !value.is_finite() {
            return Err(GgufError::InvalidTensor);
        }
        payload.extend_from_slice(&value.to_le_bytes());
    }

    Ok(TranscodedTensor {
        dtype: DType::F32,
        shape,
        payload,
    })
}

fn native_shape(tensor: &GgufTensorInfo, transpose_2d: bool) -> Result<Vec<u64>, GgufError> {
    match tensor.dimensions.as_slice() {
        [width] if *width > 0 && !transpose_2d => Ok(vec![*width]),
        [ne0, ne1] if *ne0 > 0 && *ne1 > 0 => {
            if transpose_2d {
                Ok(vec![*ne0, *ne1])
            } else {
                Ok(vec![*ne1, *ne0])
            }
        }
        _ => Err(GgufError::InvalidTensor),
    }
}

fn transpose_elements(
    bytes: &[u8],
    rows: usize,
    cols: usize,
    element_width: usize,
) -> Result<Vec<u8>, GgufError> {
    let elements = rows.checked_mul(cols).ok_or(GgufError::Overflow)?;
    let expected = elements
        .checked_mul(element_width)
        .ok_or(GgufError::Overflow)?;
    if bytes.len() != expected {
        return Err(GgufError::InvalidTensor);
    }

    let mut output = vec![0u8; expected];
    for row in 0..rows {
        for col in 0..cols {
            let source_element = row
                .checked_mul(cols)
                .and_then(|value| value.checked_add(col))
                .ok_or(GgufError::Overflow)?;
            let target_element = col
                .checked_mul(rows)
                .and_then(|value| value.checked_add(row))
                .ok_or(GgufError::Overflow)?;
            let source = source_element
                .checked_mul(element_width)
                .ok_or(GgufError::Overflow)?;
            let target = target_element
                .checked_mul(element_width)
                .ok_or(GgufError::Overflow)?;
            output[target..target + element_width]
                .copy_from_slice(&bytes[source..source + element_width]);
        }
    }
    Ok(output)
}

fn transpose_values(values: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>, GgufError> {
    let elements = rows.checked_mul(cols).ok_or(GgufError::Overflow)?;
    if values.len() != elements {
        return Err(GgufError::InvalidTensor);
    }

    let mut output = vec![0.0f32; elements];
    for row in 0..rows {
        for col in 0..cols {
            output[col * rows + row] = values[row * cols + col];
        }
    }
    Ok(output)
}

fn decode_q8_0(tensor: &GgufTensorInfo, bytes: &[u8]) -> Result<Vec<f32>, GgufError> {
    let expected = ggml_tensor_byte_len(tensor)?;
    if u64::try_from(bytes.len()).map_err(|_| GgufError::LimitExceeded)? != expected {
        return Err(GgufError::InvalidTensor);
    }

    let element_count = element_count(tensor)?;
    let capacity = usize::try_from(element_count).map_err(|_| GgufError::LimitExceeded)?;
    let mut output = Vec::with_capacity(capacity);
    for block in bytes.chunks_exact(34) {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        if !scale.is_finite() {
            return Err(GgufError::InvalidTensor);
        }
        output.extend(
            block[2..34]
                .iter()
                .map(|value| scale * f32::from(*value as i8)),
        );
    }
    if output.len() != capacity {
        return Err(GgufError::InvalidTensor);
    }
    Ok(output)
}

fn decode_q4_0(tensor: &GgufTensorInfo, bytes: &[u8]) -> Result<Vec<f32>, GgufError> {
    let expected = ggml_tensor_byte_len(tensor)?;
    if u64::try_from(bytes.len()).map_err(|_| GgufError::LimitExceeded)? != expected {
        return Err(GgufError::InvalidTensor);
    }

    let element_count = element_count(tensor)?;
    let capacity = usize::try_from(element_count).map_err(|_| GgufError::LimitExceeded)?;
    let mut output = Vec::with_capacity(capacity);
    for block in bytes.chunks_exact(18) {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        if !scale.is_finite() {
            return Err(GgufError::InvalidTensor);
        }
        for packed in &block[2..18] {
            let quant = i32::from(*packed & 0x0f) - 8;
            output.push(scale * quant as f32);
        }
        for packed in &block[2..18] {
            let quant = i32::from(*packed >> 4) - 8;
            output.push(scale * quant as f32);
        }
    }
    if output.len() != capacity {
        return Err(GgufError::InvalidTensor);
    }
    Ok(output)
}

fn decode_q4_k(tensor: &GgufTensorInfo, bytes: &[u8]) -> Result<Vec<f32>, GgufError> {
    validate_quantized_payload(tensor, bytes)?;
    let capacity = usize::try_from(element_count(tensor)?).map_err(|_| GgufError::LimitExceeded)?;
    let mut output = Vec::with_capacity(capacity);

    for block in bytes.chunks_exact(144) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        if !d.is_finite() || !dmin.is_finite() {
            return Err(GgufError::InvalidTensor);
        }

        let scales = &block[4..16];
        let qs = &block[16..144];
        let mut scale_index = 0usize;
        for group in 0..4 {
            let (scale1, min1) = scale_min_k4(scale_index, scales)?;
            let (scale2, min2) = scale_min_k4(scale_index + 1, scales)?;
            let d1 = d * f32::from(scale1);
            let m1 = dmin * f32::from(min1);
            let d2 = d * f32::from(scale2);
            let m2 = dmin * f32::from(min2);
            let q = &qs[group * 32..group * 32 + 32];

            output.extend(q.iter().map(|value| d1 * f32::from(value & 0x0f) - m1));
            output.extend(q.iter().map(|value| d2 * f32::from(value >> 4) - m2));
            scale_index += 2;
        }
    }

    if output.len() != capacity || output.iter().any(|value| !value.is_finite()) {
        return Err(GgufError::InvalidTensor);
    }
    Ok(output)
}

fn decode_q5_k(tensor: &GgufTensorInfo, bytes: &[u8]) -> Result<Vec<f32>, GgufError> {
    validate_quantized_payload(tensor, bytes)?;
    let capacity = usize::try_from(element_count(tensor)?).map_err(|_| GgufError::LimitExceeded)?;
    let mut output = Vec::with_capacity(capacity);

    for block in bytes.chunks_exact(176) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        if !d.is_finite() || !dmin.is_finite() {
            return Err(GgufError::InvalidTensor);
        }

        let scales = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..176];
        let mut scale_index = 0usize;
        let mut high_low_mask = 1u8;
        let mut high_high_mask = 2u8;

        for group in 0..4 {
            let (scale1, min1) = scale_min_k4(scale_index, scales)?;
            let (scale2, min2) = scale_min_k4(scale_index + 1, scales)?;
            let d1 = d * f32::from(scale1);
            let m1 = dmin * f32::from(min1);
            let d2 = d * f32::from(scale2);
            let m2 = dmin * f32::from(min2);
            let q = &qs[group * 32..group * 32 + 32];

            for (&low, &high_bits) in q.iter().zip(qh.iter()) {
                let high = if high_bits & high_low_mask != 0 {
                    16
                } else {
                    0
                };
                output.push(d1 * f32::from((low & 0x0f) + high) - m1);
            }
            for (&low, &high_bits) in q.iter().zip(qh.iter()) {
                let high = if high_bits & high_high_mask != 0 {
                    16
                } else {
                    0
                };
                output.push(d2 * f32::from((low >> 4) + high) - m2);
            }

            scale_index += 2;
            high_low_mask <<= 2;
            high_high_mask <<= 2;
        }
    }

    if output.len() != capacity || output.iter().any(|value| !value.is_finite()) {
        return Err(GgufError::InvalidTensor);
    }
    Ok(output)
}

fn decode_q6_k(tensor: &GgufTensorInfo, bytes: &[u8]) -> Result<Vec<f32>, GgufError> {
    validate_quantized_payload(tensor, bytes)?;
    let capacity = usize::try_from(element_count(tensor)?).map_err(|_| GgufError::LimitExceeded)?;
    let mut output = Vec::with_capacity(capacity);

    for block in bytes.chunks_exact(210) {
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        if !d.is_finite() {
            return Err(GgufError::InvalidTensor);
        }

        let mut values = [0.0f32; 256];
        for half in 0..2 {
            let ql_base = half * 64;
            let qh_base = half * 32;
            let scale_base = half * 8;
            let out_base = half * 128;

            for lane in 0..32 {
                let scale_pair = lane / 16;
                let high = qh[qh_base + lane];
                let q1 = i32::from((ql[ql_base + lane] & 0x0f) | ((high & 0x03) << 4)) - 32;
                let q2 =
                    i32::from((ql[ql_base + lane + 32] & 0x0f) | (((high >> 2) & 0x03) << 4)) - 32;
                let q3 = i32::from((ql[ql_base + lane] >> 4) | (((high >> 4) & 0x03) << 4)) - 32;
                let q4 =
                    i32::from((ql[ql_base + lane + 32] >> 4) | (((high >> 6) & 0x03) << 4)) - 32;

                let s1 = f32::from(scales[scale_base + scale_pair] as i8);
                let s2 = f32::from(scales[scale_base + scale_pair + 2] as i8);
                let s3 = f32::from(scales[scale_base + scale_pair + 4] as i8);
                let s4 = f32::from(scales[scale_base + scale_pair + 6] as i8);

                values[out_base + lane] = d * s1 * q1 as f32;
                values[out_base + lane + 32] = d * s2 * q2 as f32;
                values[out_base + lane + 64] = d * s3 * q3 as f32;
                values[out_base + lane + 96] = d * s4 * q4 as f32;
            }
        }
        output.extend_from_slice(&values);
    }

    if output.len() != capacity || output.iter().any(|value| !value.is_finite()) {
        return Err(GgufError::InvalidTensor);
    }
    Ok(output)
}

fn validate_quantized_payload(tensor: &GgufTensorInfo, bytes: &[u8]) -> Result<(), GgufError> {
    let expected = ggml_tensor_byte_len(tensor)?;
    if u64::try_from(bytes.len()).map_err(|_| GgufError::LimitExceeded)? != expected {
        return Err(GgufError::InvalidTensor);
    }
    Ok(())
}

fn scale_min_k4(index: usize, packed: &[u8]) -> Result<(u8, u8), GgufError> {
    if packed.len() != 12 || index >= 8 {
        return Err(GgufError::InvalidTensor);
    }

    if index < 4 {
        Ok((packed[index] & 0x3f, packed[index + 4] & 0x3f))
    } else {
        Ok((
            (packed[index + 4] & 0x0f) | ((packed[index - 4] >> 6) << 4),
            (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4),
        ))
    }
}

fn element_count(tensor: &GgufTensorInfo) -> Result<u64, GgufError> {
    tensor
        .dimensions
        .iter()
        .try_fold(1u64, |product, dimension| {
            product.checked_mul(*dimension).ok_or(GgufError::Overflow)
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
    use std::collections::BTreeMap;

    use super::*;

    fn model_for(bytes: &[u8], tensor: GgufTensorInfo) -> GgufModel {
        GgufModel {
            version: 3,
            alignment: 32,
            metadata: BTreeMap::new(),
            tensors: vec![tensor],
            data_offset: 0,
            file_len: bytes.len() as u64,
        }
    }

    fn f32_payload(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn decode_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    #[test]
    fn linear_weight_transpose_matches_runtime_matmul_layout() {
        let bytes = f32_payload(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let tensor = GgufTensorInfo {
            name: "linear.weight".into(),
            dimensions: vec![2, 3],
            ggml_type: GGML_TYPE_F32,
            data_offset: 0,
        };
        let model = model_for(&bytes, tensor.clone());

        let native = transcode_tensor(&bytes, &model, &tensor, true).expect("transcode");
        assert_eq!(native.dtype, DType::F32);
        assert_eq!(native.shape, vec![2, 3]);
        assert_eq!(
            decode_f32(&native.payload),
            vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
        );
    }

    #[test]
    fn q4_0_is_dequantized_into_native_f32() {
        let mut bytes = vec![0x00, 0x3c];
        bytes.extend(std::iter::repeat(0x98u8).take(16));
        let tensor = GgufTensorInfo {
            name: "q4.weight".into(),
            dimensions: vec![32],
            ggml_type: GGML_TYPE_Q4_0,
            data_offset: 0,
        };
        let model = model_for(&bytes, tensor.clone());

        let native = transcode_tensor(&bytes, &model, &tensor, false).expect("transcode");
        assert_eq!(native.dtype, DType::F32);
        assert_eq!(native.shape, vec![32]);
        let values = decode_f32(&native.payload);
        assert_eq!(&values[..16], &[0.0; 16]);
        assert_eq!(&values[16..], &[1.0; 16]);
    }

    #[test]
    fn q4_k_dequantization_matches_scale_min_packing() {
        let mut bytes = vec![0u8; 144];
        bytes[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        bytes[2..4].copy_from_slice(&0u16.to_le_bytes());
        bytes[4..16].copy_from_slice(&[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]);
        bytes[16..144].fill(0x21);

        let tensor = GgufTensorInfo {
            name: "q4_k.weight".into(),
            dimensions: vec![256],
            ggml_type: GGML_TYPE_Q4_K,
            data_offset: 0,
        };
        let model = model_for(&bytes, tensor.clone());

        assert_eq!(ggml_tensor_byte_len(&tensor), Ok(144));
        let native = transcode_tensor(&bytes, &model, &tensor, false).expect("transcode");
        let values = decode_f32(&native.payload);
        assert_eq!(values.len(), 256);
        for group in 0..4 {
            assert_eq!(&values[group * 64..group * 64 + 32], &[1.0; 32]);
            assert_eq!(&values[group * 64 + 32..group * 64 + 64], &[2.0; 32]);
        }
    }

    #[test]
    fn q5_k_dequantization_reconstructs_low_and_high_planes() {
        let mut bytes = vec![0u8; 176];
        bytes[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        bytes[2..4].copy_from_slice(&0u16.to_le_bytes());
        bytes[4..16].copy_from_slice(&[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]);
        bytes[48..176].fill(0x21);

        let tensor = GgufTensorInfo {
            name: "q5_k.weight".into(),
            dimensions: vec![256],
            ggml_type: GGML_TYPE_Q5_K,
            data_offset: 0,
        };
        let model = model_for(&bytes, tensor.clone());

        assert_eq!(ggml_tensor_byte_len(&tensor), Ok(176));
        let native = transcode_tensor(&bytes, &model, &tensor, false).expect("transcode");
        let values = decode_f32(&native.payload);
        assert_eq!(values.len(), 256);
        for group in 0..4 {
            assert_eq!(&values[group * 64..group * 64 + 32], &[1.0; 32]);
            assert_eq!(&values[group * 64 + 32..group * 64 + 64], &[2.0; 32]);
        }

        bytes[16..48].fill(0xff);
        let model = model_for(&bytes, tensor.clone());
        let native = transcode_tensor(&bytes, &model, &tensor, false).expect("high plane");
        let values = decode_f32(&native.payload);
        for group in 0..4 {
            assert_eq!(&values[group * 64..group * 64 + 32], &[17.0; 32]);
            assert_eq!(&values[group * 64 + 32..group * 64 + 64], &[18.0; 32]);
        }
    }

    #[test]
    fn q6_k_dequantization_reconstructs_signed_six_bit_values() {
        let mut bytes = vec![0u8; 210];
        bytes[0..128].fill(0x11);
        bytes[128..192].fill(0xaa);
        bytes[192..208].fill(1);
        bytes[208..210].copy_from_slice(&0x3c00u16.to_le_bytes());

        let tensor = GgufTensorInfo {
            name: "q6_k.weight".into(),
            dimensions: vec![256],
            ggml_type: GGML_TYPE_Q6_K,
            data_offset: 0,
        };
        let model = model_for(&bytes, tensor.clone());

        assert_eq!(ggml_tensor_byte_len(&tensor), Ok(210));
        let native = transcode_tensor(&bytes, &model, &tensor, false).expect("transcode");
        assert_eq!(decode_f32(&native.payload), vec![1.0; 256]);
    }

    #[test]
    fn k_quant_rows_require_complete_256_element_blocks() {
        for (ggml_type, block_bytes) in [
            (GGML_TYPE_Q4_K, 144usize),
            (GGML_TYPE_Q5_K, 176usize),
            (GGML_TYPE_Q6_K, 210usize),
        ] {
            let bytes = vec![0u8; block_bytes];
            let tensor = GgufTensorInfo {
                name: "invalid-k.weight".into(),
                dimensions: vec![128],
                ggml_type,
                data_offset: 0,
            };
            assert_eq!(ggml_tensor_byte_len(&tensor), Err(GgufError::InvalidTensor));
            let model = model_for(&bytes, tensor.clone());
            assert_eq!(
                transcode_tensor(&bytes, &model, &tensor, false),
                Err(GgufError::InvalidTensor)
            );
        }
    }

    #[test]
    fn tensor_slice_fails_closed_when_payload_is_truncated() {
        let bytes = f32_payload(&[1.0, 2.0, 3.0]);
        let tensor = GgufTensorInfo {
            name: "broken.weight".into(),
            dimensions: vec![2, 2],
            ggml_type: GGML_TYPE_F32,
            data_offset: 0,
        };
        let model = model_for(&bytes, tensor.clone());
        assert_eq!(
            gguf_tensor_bytes(&bytes, &model, &tensor),
            Err(GgufError::Truncated)
        );
    }
}
