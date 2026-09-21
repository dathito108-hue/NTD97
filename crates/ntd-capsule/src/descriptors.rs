#![forbid(unsafe_code)]

use ntd_ir::{DType, IrVersion};

pub const TENSOR_DESCRIPTOR_MAGIC: [u8; 6] = *b"NTS97\0";
pub const TENSOR_DESCRIPTOR_HEADER_LEN: usize = 16;
const TENSOR_RECORD_HEADER_LEN: usize = 16;

pub const DESCRIPTOR_FRAME_MAGIC: [u8; 6] = *b"NDF97\0";
pub const DESCRIPTOR_FRAME_HEADER_LEN: usize = 32;
pub const DESCRIPTOR_MAJOR: u16 = 0;
pub const DESCRIPTOR_MINOR: u16 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDescriptor {
    pub id: u32,
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub byte_len: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DescriptorFrameKind {
    Tokenizer = 1,
    Codec = 2,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DescriptorFrame {
    pub kind: DescriptorFrameKind,
    pub format: String,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DescriptorError {
    Truncated,
    InvalidMagic,
    InvalidHeader,
    InvalidKind(u8),
    InvalidDType(u8),
    InvalidUtf8,
    UnsupportedVersion { major: u16, minor: u16 },
    NonCanonicalEncoding,
    Overflow,
}

pub fn encode_tensor_descriptors(
    descriptors: &[TensorDescriptor],
) -> Result<Vec<u8>, DescriptorError> {
    let count = u32::try_from(descriptors.len()).map_err(|_| DescriptorError::Overflow)?;
    let mut out = Vec::new();

    out.extend_from_slice(&TENSOR_DESCRIPTOR_MAGIC);
    push_u16(&mut out, TENSOR_DESCRIPTOR_HEADER_LEN as u16);
    push_u16(&mut out, IrVersion::CURRENT.major);
    push_u16(&mut out, IrVersion::CURRENT.minor);
    push_u32(&mut out, count);

    for descriptor in descriptors {
        let rank = u8::try_from(descriptor.shape.len()).map_err(|_| DescriptorError::Overflow)?;
        push_u32(&mut out, descriptor.id);
        out.push(dtype_tag(descriptor.dtype));
        out.push(rank);
        push_u16(&mut out, 0);
        push_u64(&mut out, descriptor.byte_len);

        for dimension in &descriptor.shape {
            push_u64(&mut out, *dimension);
        }
    }

    Ok(out)
}

pub fn decode_tensor_descriptors(
    bytes: &[u8],
) -> Result<Vec<TensorDescriptor>, DescriptorError> {
    let mut cursor = Cursor::new(bytes);

    if cursor.take(6)? != TENSOR_DESCRIPTOR_MAGIC.as_slice() {
        return Err(DescriptorError::InvalidMagic);
    }

    if usize::from(cursor.u16()?) != TENSOR_DESCRIPTOR_HEADER_LEN {
        return Err(DescriptorError::InvalidHeader);
    }

    let major = cursor.u16()?;
    let minor = cursor.u16()?;
    let version = IrVersion { major, minor };

    if !IrVersion::CURRENT.can_read(version) {
        return Err(DescriptorError::UnsupportedVersion { major, minor });
    }

    let count = cursor.u32()?;
    let mut descriptors =
        Vec::with_capacity(usize::try_from(count).map_err(|_| DescriptorError::Overflow)?);

    for _ in 0..count {
        let id = cursor.u32()?;
        let dtype = decode_dtype(cursor.u8()?)?;
        let rank = cursor.u8()?;

        if cursor.u16()? != 0 {
            return Err(DescriptorError::NonCanonicalEncoding);
        }

        let byte_len = cursor.u64()?;
        let mut shape = Vec::with_capacity(usize::from(rank));

        for _ in 0..rank {
            shape.push(cursor.u64()?);
        }

        descriptors.push(TensorDescriptor {
            id,
            dtype,
            shape,
            byte_len,
        });
    }

    if !cursor.is_finished() {
        return Err(DescriptorError::NonCanonicalEncoding);
    }

    Ok(descriptors)
}

pub fn encode_descriptor_frame(frame: &DescriptorFrame) -> Result<Vec<u8>, DescriptorError> {
    let format_len = u32::try_from(frame.format.len()).map_err(|_| DescriptorError::Overflow)?;
    let payload_len =
        u64::try_from(frame.payload.len()).map_err(|_| DescriptorError::Overflow)?;

    let mut out = Vec::new();
    out.extend_from_slice(&DESCRIPTOR_FRAME_MAGIC);
    push_u16(&mut out, DESCRIPTOR_FRAME_HEADER_LEN as u16);
    out.push(frame.kind as u8);
    out.push(0);
    push_u16(&mut out, DESCRIPTOR_MAJOR);
    push_u16(&mut out, DESCRIPTOR_MINOR);
    push_u16(&mut out, 0);
    push_u32(&mut out, format_len);
    push_u64(&mut out, payload_len);
    push_u32(&mut out, 0);
    out.extend_from_slice(frame.format.as_bytes());
    out.extend_from_slice(&frame.payload);
    Ok(out)
}

pub fn decode_descriptor_frame(bytes: &[u8]) -> Result<DescriptorFrame, DescriptorError> {
    let mut cursor = Cursor::new(bytes);

    if cursor.take(6)? != DESCRIPTOR_FRAME_MAGIC.as_slice() {
        return Err(DescriptorError::InvalidMagic);
    }

    if usize::from(cursor.u16()?) != DESCRIPTOR_FRAME_HEADER_LEN {
        return Err(DescriptorError::InvalidHeader);
    }

    let kind = match cursor.u8()? {
        1 => DescriptorFrameKind::Tokenizer,
        2 => DescriptorFrameKind::Codec,
        other => return Err(DescriptorError::InvalidKind(other)),
    };

    if cursor.u8()? != 0 {
        return Err(DescriptorError::NonCanonicalEncoding);
    }

    let major = cursor.u16()?;
    let minor = cursor.u16()?;

    if major != DESCRIPTOR_MAJOR || minor > DESCRIPTOR_MINOR {
        return Err(DescriptorError::UnsupportedVersion { major, minor });
    }

    if cursor.u16()? != 0 {
        return Err(DescriptorError::NonCanonicalEncoding);
    }

    let format_len = cursor.u32()?;
    let payload_len = cursor.u64()?;

    if cursor.u32()? != 0 {
        return Err(DescriptorError::NonCanonicalEncoding);
    }

    let format_bytes =
        cursor.take(usize::try_from(format_len).map_err(|_| DescriptorError::Overflow)?)?;
    let format = std::str::from_utf8(format_bytes)
        .map_err(|_| DescriptorError::InvalidUtf8)?
        .to_owned();
    let payload = cursor
        .take(usize::try_from(payload_len).map_err(|_| DescriptorError::Overflow)?)?
        .to_vec();

    if !cursor.is_finished() {
        return Err(DescriptorError::NonCanonicalEncoding);
    }

    Ok(DescriptorFrame {
        kind,
        format,
        payload,
    })
}

fn dtype_tag(dtype: DType) -> u8 {
    match dtype {
        DType::F32 => 1,
        DType::F16 => 2,
        DType::Bf16 => 3,
        DType::I8 => 4,
        DType::U8 => 5,
        DType::I32 => 6,
        DType::I64 => 7,
        DType::Bool => 8,
    }
}

fn decode_dtype(tag: u8) -> Result<DType, DescriptorError> {
    match tag {
        1 => Ok(DType::F32),
        2 => Ok(DType::F16),
        3 => Ok(DType::Bf16),
        4 => Ok(DType::I8),
        5 => Ok(DType::U8),
        6 => Ok(DType::I32),
        7 => Ok(DType::I64),
        8 => Ok(DType::Bool),
        other => Err(DescriptorError::InvalidDType(other)),
    }
}

fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], DescriptorError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(DescriptorError::Overflow)?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or(DescriptorError::Truncated)?;
        self.offset = end;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8, DescriptorError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, DescriptorError> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> Result<u32, DescriptorError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn u64(&mut self) -> Result<u64, DescriptorError> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn is_finished(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_descriptors_round_trip() {
        let descriptors = vec![
            TensorDescriptor {
                id: 7,
                dtype: DType::F16,
                shape: vec![32, 4096],
                byte_len: 262_144,
            },
            TensorDescriptor {
                id: 8,
                dtype: DType::I8,
                shape: vec![4096, 4096],
                byte_len: 16_777_216,
            },
        ];

        let encoded = encode_tensor_descriptors(&descriptors).expect("encode");
        assert_eq!(
            decode_tensor_descriptors(&encoded).expect("decode"),
            descriptors
        );
    }

    #[test]
    fn tensor_descriptor_encoding_is_deterministic() {
        let descriptors = vec![TensorDescriptor {
            id: 1,
            dtype: DType::F32,
            shape: vec![2, 3],
            byte_len: 24,
        }];

        assert_eq!(
            encode_tensor_descriptors(&descriptors).expect("first"),
            encode_tensor_descriptors(&descriptors).expect("second")
        );
    }

    #[test]
    fn tokenizer_frame_round_trips() {
        let frame = DescriptorFrame {
            kind: DescriptorFrameKind::Tokenizer,
            format: "ntd97.tokenizer.bpe".to_owned(),
            payload: b"canonical-tokenizer-descriptor".to_vec(),
        };

        let encoded = encode_descriptor_frame(&frame).expect("encode");
        assert_eq!(decode_descriptor_frame(&encoded).expect("decode"), frame);
    }

    #[test]
    fn codec_frame_round_trips() {
        let frame = DescriptorFrame {
            kind: DescriptorFrameKind::Codec,
            format: "ntd97.codec.audio".to_owned(),
            payload: b"codec-descriptor".to_vec(),
        };

        let encoded = encode_descriptor_frame(&frame).expect("encode");
        assert_eq!(decode_descriptor_frame(&encoded).expect("decode"), frame);
    }

    #[test]
    fn frames_reject_trailing_bytes() {
        let frame = DescriptorFrame {
            kind: DescriptorFrameKind::Codec,
            format: "codec".to_owned(),
            payload: vec![1, 2, 3],
        };

        let mut encoded = encode_descriptor_frame(&frame).expect("encode");
        encoded.push(0);

        assert_eq!(
            decode_descriptor_frame(&encoded),
            Err(DescriptorError::NonCanonicalEncoding)
        );
    }

    #[test]
    fn fixed_record_lengths_are_stable() {
        assert_eq!(TENSOR_RECORD_HEADER_LEN, 16);
    }
}
