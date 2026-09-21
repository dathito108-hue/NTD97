#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use super::{
    GgufError, GgufModel, GgufTensorInfo, GgufValue, GgufValueType, GGUF_DEFAULT_ALIGNMENT,
    GGUF_MAGIC, GGUF_VERSION,
};

const MAX_METADATA_ENTRIES: u64 = 100_000;
const MAX_TENSORS: u64 = 1_000_000;
const MAX_ARRAY_ELEMENTS: u64 = 16_000_000;
const MAX_STRING_BYTES: u64 = 64 * 1024 * 1024;
const MAX_DIMS: u32 = 4;

pub fn parse_gguf(bytes: &[u8]) -> Result<GgufModel, GgufError> {
    let mut cursor = Cursor::new(bytes);
    if cursor.take(4)? != GGUF_MAGIC.as_slice() {
        return Err(GgufError::InvalidMagic);
    }

    let version = cursor.u32()?;
    if version != GGUF_VERSION {
        return Err(GgufError::UnsupportedVersion(version));
    }

    let tensor_count = cursor.u64()?;
    let metadata_count = cursor.u64()?;
    if tensor_count > MAX_TENSORS || metadata_count > MAX_METADATA_ENTRIES {
        return Err(GgufError::LimitExceeded);
    }

    let mut metadata = BTreeMap::new();
    for _ in 0..metadata_count {
        let key = cursor.string()?;
        if key.is_empty() || metadata.contains_key(&key) {
            return Err(GgufError::DuplicateMetadata(key));
        }
        let value_type = GgufValueType::try_from(cursor.u32()?)?;
        let value = parse_value(&mut cursor, value_type)?;
        metadata.insert(key, value);
    }

    let alignment = match metadata.get("general.alignment") {
        Some(value) => value.as_u32().ok_or(GgufError::InvalidAlignment)?,
        None => GGUF_DEFAULT_ALIGNMENT,
    };
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(GgufError::InvalidAlignment);
    }

    let mut names = BTreeSet::new();
    let mut tensors =
        Vec::with_capacity(usize::try_from(tensor_count).map_err(|_| GgufError::LimitExceeded)?);
    for _ in 0..tensor_count {
        let name = cursor.string()?;
        if name.is_empty() || !names.insert(name.clone()) {
            return Err(GgufError::DuplicateTensor(name));
        }
        let dims = cursor.u32()?;
        if dims == 0 || dims > MAX_DIMS {
            return Err(GgufError::InvalidTensor);
        }
        let mut dimensions = Vec::with_capacity(dims as usize);
        for _ in 0..dims {
            let dimension = cursor.u64()?;
            if dimension == 0 {
                return Err(GgufError::InvalidTensor);
            }
            dimensions.push(dimension);
        }
        let ggml_type = cursor.u32()?;
        let data_offset = cursor.u64()?;
        if data_offset % u64::from(alignment) != 0 {
            return Err(GgufError::InvalidTensorAlignment);
        }
        tensors.push(GgufTensorInfo {
            name,
            dimensions,
            ggml_type,
            data_offset,
        });
    }

    let data_offset = align_up(cursor.offset_u64()?, u64::from(alignment))?;
    let file_len = u64::try_from(bytes.len()).map_err(|_| GgufError::LimitExceeded)?;
    if data_offset > file_len {
        return Err(GgufError::Truncated);
    }
    for tensor in &tensors {
        let absolute = data_offset
            .checked_add(tensor.data_offset)
            .ok_or(GgufError::Overflow)?;
        if absolute > file_len {
            return Err(GgufError::Truncated);
        }
    }

    let model = GgufModel {
        version,
        alignment,
        metadata,
        tensors,
        data_offset,
        file_len,
    };
    model.architecture()?;
    model.tokenizer()?;
    Ok(model)
}

fn parse_value(cursor: &mut Cursor<'_>, value_type: GgufValueType) -> Result<GgufValue, GgufError> {
    match value_type {
        GgufValueType::Uint8 => Ok(GgufValue::Uint8(cursor.u8()?)),
        GgufValueType::Int8 => Ok(GgufValue::Int8(cursor.i8()?)),
        GgufValueType::Uint16 => Ok(GgufValue::Uint16(cursor.u16()?)),
        GgufValueType::Int16 => Ok(GgufValue::Int16(cursor.i16()?)),
        GgufValueType::Uint32 => Ok(GgufValue::Uint32(cursor.u32()?)),
        GgufValueType::Int32 => Ok(GgufValue::Int32(cursor.i32()?)),
        GgufValueType::Float32 => Ok(GgufValue::Float32(cursor.f32()?)),
        GgufValueType::Bool => match cursor.u8()? {
            0 => Ok(GgufValue::Bool(false)),
            1 => Ok(GgufValue::Bool(true)),
            other => Err(GgufError::InvalidBool(other)),
        },
        GgufValueType::String => Ok(GgufValue::String(cursor.string()?)),
        GgufValueType::Array => {
            let element_type = GgufValueType::try_from(cursor.u32()?)?;
            if element_type == GgufValueType::Array {
                return Err(GgufError::InvalidArrayType);
            }
            let count = cursor.u64()?;
            if count > MAX_ARRAY_ELEMENTS {
                return Err(GgufError::LimitExceeded);
            }
            let mut values =
                Vec::with_capacity(usize::try_from(count).map_err(|_| GgufError::LimitExceeded)?);
            for _ in 0..count {
                values.push(parse_value(cursor, element_type)?);
            }
            Ok(GgufValue::Array {
                element_type,
                values,
            })
        }
        GgufValueType::Uint64 => Ok(GgufValue::Uint64(cursor.u64()?)),
        GgufValueType::Int64 => Ok(GgufValue::Int64(cursor.i64()?)),
        GgufValueType::Float64 => Ok(GgufValue::Float64(cursor.f64()?)),
    }
}

fn align_up(value: u64, alignment: u64) -> Result<u64, GgufError> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(GgufError::InvalidAlignment);
    }
    value
        .checked_add(alignment - 1)
        .map(|aligned| aligned & !(alignment - 1))
        .ok_or(GgufError::Overflow)
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], GgufError> {
        let end = self.offset.checked_add(len).ok_or(GgufError::Overflow)?;
        let out = self
            .bytes
            .get(self.offset..end)
            .ok_or(GgufError::Truncated)?;
        self.offset = end;
        Ok(out)
    }

    fn offset_u64(&self) -> Result<u64, GgufError> {
        u64::try_from(self.offset).map_err(|_| GgufError::LimitExceeded)
    }

    fn u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.take(1)?[0])
    }

    fn i8(&mut self) -> Result<i8, GgufError> {
        Ok(self.u8()? as i8)
    }

    fn u16(&mut self) -> Result<u16, GgufError> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn i16(&mut self) -> Result<i16, GgufError> {
        let bytes = self.take(2)?;
        Ok(i16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> Result<u32, GgufError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn i32(&mut self) -> Result<i32, GgufError> {
        let bytes = self.take(4)?;
        Ok(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn f32(&mut self) -> Result<f32, GgufError> {
        Ok(f32::from_bits(self.u32()?))
    }

    fn u64(&mut self) -> Result<u64, GgufError> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn i64(&mut self) -> Result<i64, GgufError> {
        let bytes = self.take(8)?;
        Ok(i64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn f64(&mut self) -> Result<f64, GgufError> {
        Ok(f64::from_bits(self.u64()?))
    }

    fn string(&mut self) -> Result<String, GgufError> {
        let len = self.u64()?;
        if len > MAX_STRING_BYTES {
            return Err(GgufError::LimitExceeded);
        }
        let bytes = self.take(usize::try_from(len).map_err(|_| GgufError::LimitExceeded)?)?;
        std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| GgufError::InvalidUtf8)
    }
}
