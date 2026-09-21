#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use ntd_ir::{DType, Graph, ValueId, ValueType};

use crate::{
    decode_graph, decode_tensor_descriptors, encode_tensor_descriptors, sha256, CapsuleBuilder,
    CapsuleView, ChunkStorageView, ChunkView, DescriptorError, Digest, IrCodecError, SectionKind,
    TensorDescriptor, TENSOR_DESCRIPTOR_MAGIC,
};

pub const NATIVE_TENSOR_MAGIC: [u8; 6] = *b"NTP97\0";
pub const NATIVE_TENSOR_HEADER_LEN: usize = 80;
pub const NATIVE_TENSOR_MAJOR: u16 = 0;
pub const NATIVE_TENSOR_MINOR: u16 = 1;
pub const NO_GRAPH_BINDING: u32 = u32::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationMetadata {
    None,
    SymmetricI8 { scale_bits: u32 },
    AffineI8 { scale_bits: u32, zero_point: i32 },
}

impl QuantizationMetadata {
    pub fn symmetric_i8(scale: f32) -> Result<Self, NativeTensorError> {
        validate_scale(scale)?;
        Ok(Self::SymmetricI8 {
            scale_bits: scale.to_bits(),
        })
    }

    pub fn affine_i8(scale: f32, zero_point: i32) -> Result<Self, NativeTensorError> {
        validate_scale(scale)?;
        if !(-128..=127).contains(&zero_point) {
            return Err(NativeTensorError::InvalidQuantization);
        }
        Ok(Self::AffineI8 {
            scale_bits: scale.to_bits(),
            zero_point,
        })
    }

    pub fn scale(self) -> Option<f32> {
        match self {
            Self::None => None,
            Self::SymmetricI8 { scale_bits } | Self::AffineI8 { scale_bits, .. } => {
                Some(f32::from_bits(scale_bits))
            }
        }
    }

    pub fn zero_point(self) -> Option<i32> {
        match self {
            Self::None => None,
            Self::SymmetricI8 { .. } => Some(0),
            Self::AffineI8 { zero_point, .. } => Some(zero_point),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeTensor {
    pub descriptor: TensorDescriptor,
    pub graph_value: Option<ValueId>,
    pub quantization: QuantizationMetadata,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeProgram {
    pub graph: Graph,
    pub tensors: Vec<NativeTensor>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeTensorError {
    Descriptor(DescriptorError),
    Graph(IrCodecError),
    Truncated,
    InvalidMagic,
    InvalidHeader,
    UnsupportedVersion { major: u16, minor: u16 },
    InvalidDType(u8),
    InvalidQuantization,
    InvalidShape,
    PayloadLengthMismatch { expected: u64, actual: u64 },
    DescriptorIntegrityMismatch,
    NonCanonicalEncoding,
    MissingGraph,
    MultipleGraphs,
    MissingDescriptorTable,
    MultipleDescriptorTables,
    DuplicateTensorId(u32),
    DuplicateGraphBinding(ValueId),
    MissingTensorShard(u32),
    UnknownTensorShard(u32),
    TensorDescriptorMismatch(u32),
    GraphBindingMissing(ValueId),
    GraphBindingTypeMismatch(ValueId),
    MissingExternalContent(Digest),
    ExternalLengthMismatch { expected: u64, actual: u64 },
    ExternalIntegrityMismatch(Digest),
    UnknownTensorChunk,
    Overflow,
}

pub trait ContentStore {
    fn get(&self, hash: &Digest) -> Option<&[u8]>;
}

#[derive(Debug, Default, Clone)]
pub struct MemoryContentStore {
    entries: BTreeMap<Digest, Vec<u8>>,
}

impl MemoryContentStore {
    pub fn insert(&mut self, bytes: Vec<u8>) -> Digest {
        let hash = sha256(&bytes);
        self.entries.entry(hash).or_insert(bytes);
        hash
    }

    pub fn insert_verified(
        &mut self,
        hash: Digest,
        bytes: Vec<u8>,
    ) -> Result<(), NativeTensorError> {
        if sha256(&bytes) != hash {
            return Err(NativeTensorError::ExternalIntegrityMismatch(hash));
        }
        self.entries.entry(hash).or_insert(bytes);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl ContentStore for MemoryContentStore {
    fn get(&self, hash: &Digest) -> Option<&[u8]> {
        self.entries.get(hash).map(Vec::as_slice)
    }
}

pub fn encode_native_tensor(tensor: &NativeTensor) -> Result<Vec<u8>, NativeTensorError> {
    validate_tensor(tensor)?;

    let descriptor_bytes = encode_tensor_descriptors(std::slice::from_ref(&tensor.descriptor))
        .map_err(NativeTensorError::Descriptor)?;
    let descriptor_hash = sha256(&descriptor_bytes);
    let rank =
        u8::try_from(tensor.descriptor.shape.len()).map_err(|_| NativeTensorError::Overflow)?;
    let element_count = element_count(&tensor.descriptor.shape)?;
    let payload_len =
        u64::try_from(tensor.payload.len()).map_err(|_| NativeTensorError::Overflow)?;
    let (quant_tag, scale_bits, zero_point) = encode_quantization(tensor.quantization)?;

    let shape_bytes = tensor
        .descriptor
        .shape
        .len()
        .checked_mul(8)
        .ok_or(NativeTensorError::Overflow)?;
    let capacity = NATIVE_TENSOR_HEADER_LEN
        .checked_add(shape_bytes)
        .and_then(|value| value.checked_add(tensor.payload.len()))
        .ok_or(NativeTensorError::Overflow)?;
    let mut out = Vec::with_capacity(capacity);

    out.extend_from_slice(&NATIVE_TENSOR_MAGIC);
    push_u16(&mut out, NATIVE_TENSOR_HEADER_LEN as u16);
    push_u16(&mut out, NATIVE_TENSOR_MAJOR);
    push_u16(&mut out, NATIVE_TENSOR_MINOR);
    push_u32(&mut out, tensor.descriptor.id);
    push_u32(
        &mut out,
        tensor.graph_value.map_or(NO_GRAPH_BINDING, |value| value.0),
    );
    out.push(dtype_tag(tensor.descriptor.dtype));
    out.push(quant_tag);
    out.push(rank);
    out.push(0);
    push_u64(&mut out, element_count);
    push_u64(&mut out, payload_len);
    push_u32(&mut out, scale_bits);
    push_i32(&mut out, zero_point);
    out.extend_from_slice(&descriptor_hash);

    for dimension in &tensor.descriptor.shape {
        push_u64(&mut out, *dimension);
    }
    out.extend_from_slice(&tensor.payload);
    Ok(out)
}

pub fn decode_native_tensor(bytes: &[u8]) -> Result<NativeTensor, NativeTensorError> {
    let mut cursor = Cursor::new(bytes);
    if cursor.take(6)? != NATIVE_TENSOR_MAGIC.as_slice() {
        return Err(NativeTensorError::InvalidMagic);
    }
    if usize::from(cursor.u16()?) != NATIVE_TENSOR_HEADER_LEN {
        return Err(NativeTensorError::InvalidHeader);
    }

    let major = cursor.u16()?;
    let minor = cursor.u16()?;
    if major != NATIVE_TENSOR_MAJOR || minor > NATIVE_TENSOR_MINOR {
        return Err(NativeTensorError::UnsupportedVersion { major, minor });
    }

    let tensor_id = cursor.u32()?;
    let graph_value_raw = cursor.u32()?;
    let dtype = decode_dtype(cursor.u8()?)?;
    let quant_tag = cursor.u8()?;
    let rank = cursor.u8()?;
    if cursor.u8()? != 0 {
        return Err(NativeTensorError::NonCanonicalEncoding);
    }
    let encoded_element_count = cursor.u64()?;
    let payload_len = cursor.u64()?;
    let scale_bits = cursor.u32()?;
    let zero_point = cursor.i32()?;
    let mut descriptor_hash = [0u8; 32];
    descriptor_hash.copy_from_slice(cursor.take(32)?);

    let mut shape = Vec::with_capacity(usize::from(rank));
    for _ in 0..rank {
        shape.push(cursor.u64()?);
    }
    if element_count(&shape)? != encoded_element_count {
        return Err(NativeTensorError::InvalidShape);
    }

    let payload_size = usize::try_from(payload_len).map_err(|_| NativeTensorError::Overflow)?;
    let payload = cursor.take(payload_size)?.to_vec();
    if !cursor.is_finished() {
        return Err(NativeTensorError::NonCanonicalEncoding);
    }

    let quantization = decode_quantization(quant_tag, scale_bits, zero_point)?;
    let descriptor = TensorDescriptor {
        id: tensor_id,
        dtype,
        shape,
        byte_len: payload_len,
    };
    let tensor = NativeTensor {
        descriptor,
        graph_value: (graph_value_raw != NO_GRAPH_BINDING).then_some(ValueId(graph_value_raw)),
        quantization,
        payload,
    };
    validate_tensor(&tensor)?;

    let canonical_descriptor = encode_tensor_descriptors(std::slice::from_ref(&tensor.descriptor))
        .map_err(NativeTensorError::Descriptor)?;
    if sha256(&canonical_descriptor) != descriptor_hash {
        return Err(NativeTensorError::DescriptorIntegrityMismatch);
    }
    Ok(tensor)
}

pub fn push_tensor_descriptor_section(
    builder: &mut CapsuleBuilder,
    descriptors: &[TensorDescriptor],
) -> Result<(), NativeTensorError> {
    let bytes = encode_tensor_descriptors(descriptors).map_err(NativeTensorError::Descriptor)?;
    builder.push_embedded(SectionKind::Tensors, bytes);
    Ok(())
}

pub fn push_native_tensor_shard(
    builder: &mut CapsuleBuilder,
    tensor: &NativeTensor,
) -> Result<(), NativeTensorError> {
    builder.push_embedded(SectionKind::Tensors, encode_native_tensor(tensor)?);
    Ok(())
}

pub fn load_native_program<S: ContentStore>(
    capsule: &CapsuleView<'_>,
    store: &S,
) -> Result<NativeProgram, NativeTensorError> {
    let graph_chunks = capsule
        .chunks
        .iter()
        .filter(|chunk| chunk.kind == SectionKind::Graph)
        .collect::<Vec<_>>();
    let graph_chunk = match graph_chunks.as_slice() {
        [] => return Err(NativeTensorError::MissingGraph),
        [chunk] => *chunk,
        _ => return Err(NativeTensorError::MultipleGraphs),
    };
    let graph_content = resolve_chunk(graph_chunk, store)?;
    let graph = decode_graph(graph_content.as_slice()).map_err(NativeTensorError::Graph)?;

    let mut descriptor_table = None;
    let mut shards = Vec::new();
    for chunk in capsule
        .chunks
        .iter()
        .filter(|chunk| chunk.kind == SectionKind::Tensors)
    {
        let content = resolve_chunk(chunk, store)?;
        let bytes = content.as_slice();
        if bytes.starts_with(&TENSOR_DESCRIPTOR_MAGIC) {
            if descriptor_table.is_some() {
                return Err(NativeTensorError::MultipleDescriptorTables);
            }
            descriptor_table =
                Some(decode_tensor_descriptors(bytes).map_err(NativeTensorError::Descriptor)?);
        } else if bytes.starts_with(&NATIVE_TENSOR_MAGIC) {
            shards.push(decode_native_tensor(bytes)?);
        } else {
            return Err(NativeTensorError::UnknownTensorChunk);
        }
    }

    let descriptors = descriptor_table.ok_or(NativeTensorError::MissingDescriptorTable)?;
    let mut descriptor_by_id = BTreeMap::new();
    for descriptor in descriptors {
        let id = descriptor.id;
        if descriptor_by_id.insert(id, descriptor).is_some() {
            return Err(NativeTensorError::DuplicateTensorId(id));
        }
    }

    let mut seen_tensor_ids = BTreeSet::new();
    let mut seen_bindings = BTreeSet::new();
    for shard in &shards {
        let id = shard.descriptor.id;
        if !seen_tensor_ids.insert(id) {
            return Err(NativeTensorError::DuplicateTensorId(id));
        }
        let descriptor = descriptor_by_id
            .get(&id)
            .ok_or(NativeTensorError::UnknownTensorShard(id))?;
        if descriptor != &shard.descriptor {
            return Err(NativeTensorError::TensorDescriptorMismatch(id));
        }
        if let Some(value_id) = shard.graph_value {
            if !seen_bindings.insert(value_id) {
                return Err(NativeTensorError::DuplicateGraphBinding(value_id));
            }
            validate_graph_binding(&graph, value_id, descriptor)?;
        }
    }

    for tensor_id in descriptor_by_id.keys() {
        if !seen_tensor_ids.contains(tensor_id) {
            return Err(NativeTensorError::MissingTensorShard(*tensor_id));
        }
    }

    shards.sort_by_key(|tensor| tensor.descriptor.id);
    Ok(NativeProgram {
        graph,
        tensors: shards,
    })
}

fn validate_graph_binding(
    graph: &Graph,
    value_id: ValueId,
    descriptor: &TensorDescriptor,
) -> Result<(), NativeTensorError> {
    let decl = graph
        .inputs
        .iter()
        .find(|decl| decl.id == value_id)
        .ok_or(NativeTensorError::GraphBindingMissing(value_id))?;
    match decl.ty {
        ValueType::Scalar(dtype) if dtype == descriptor.dtype && descriptor.shape.is_empty() => {
            Ok(())
        }
        ValueType::Tensor { dtype, rank }
            if dtype == descriptor.dtype && usize::from(rank) == descriptor.shape.len() =>
        {
            Ok(())
        }
        _ => Err(NativeTensorError::GraphBindingTypeMismatch(value_id)),
    }
}

fn validate_tensor(tensor: &NativeTensor) -> Result<(), NativeTensorError> {
    let elements = element_count(&tensor.descriptor.shape)?;
    let actual = u64::try_from(tensor.payload.len()).map_err(|_| NativeTensorError::Overflow)?;
    if actual != tensor.descriptor.byte_len {
        return Err(NativeTensorError::PayloadLengthMismatch {
            expected: tensor.descriptor.byte_len,
            actual,
        });
    }

    let expected = match tensor.quantization {
        QuantizationMetadata::None => elements
            .checked_mul(
                u64::try_from(dtype_width(tensor.descriptor.dtype))
                    .map_err(|_| NativeTensorError::Overflow)?,
            )
            .ok_or(NativeTensorError::Overflow)?,
        QuantizationMetadata::SymmetricI8 { scale_bits } => {
            validate_quantized_i8(tensor.descriptor.dtype, scale_bits, 0)?;
            elements
        }
        QuantizationMetadata::AffineI8 {
            scale_bits,
            zero_point,
        } => {
            validate_quantized_i8(tensor.descriptor.dtype, scale_bits, zero_point)?;
            elements
        }
    };
    if actual != expected {
        return Err(NativeTensorError::PayloadLengthMismatch { expected, actual });
    }
    Ok(())
}

fn validate_quantized_i8(
    dtype: DType,
    scale_bits: u32,
    zero_point: i32,
) -> Result<(), NativeTensorError> {
    if dtype != DType::I8 || !(-128..=127).contains(&zero_point) {
        return Err(NativeTensorError::InvalidQuantization);
    }
    validate_scale(f32::from_bits(scale_bits))
}

fn validate_scale(scale: f32) -> Result<(), NativeTensorError> {
    if scale.is_finite() && scale > 0.0 {
        Ok(())
    } else {
        Err(NativeTensorError::InvalidQuantization)
    }
}

fn encode_quantization(
    quantization: QuantizationMetadata,
) -> Result<(u8, u32, i32), NativeTensorError> {
    match quantization {
        QuantizationMetadata::None => Ok((0, 0, 0)),
        QuantizationMetadata::SymmetricI8 { scale_bits } => {
            validate_scale(f32::from_bits(scale_bits))?;
            Ok((1, scale_bits, 0))
        }
        QuantizationMetadata::AffineI8 {
            scale_bits,
            zero_point,
        } => {
            validate_scale(f32::from_bits(scale_bits))?;
            if !(-128..=127).contains(&zero_point) {
                return Err(NativeTensorError::InvalidQuantization);
            }
            Ok((2, scale_bits, zero_point))
        }
    }
}

fn decode_quantization(
    tag: u8,
    scale_bits: u32,
    zero_point: i32,
) -> Result<QuantizationMetadata, NativeTensorError> {
    match tag {
        0 if scale_bits == 0 && zero_point == 0 => Ok(QuantizationMetadata::None),
        1 if zero_point == 0 => {
            validate_scale(f32::from_bits(scale_bits))?;
            Ok(QuantizationMetadata::SymmetricI8 { scale_bits })
        }
        2 if (-128..=127).contains(&zero_point) => {
            validate_scale(f32::from_bits(scale_bits))?;
            Ok(QuantizationMetadata::AffineI8 {
                scale_bits,
                zero_point,
            })
        }
        _ => Err(NativeTensorError::InvalidQuantization),
    }
}

fn element_count(shape: &[u64]) -> Result<u64, NativeTensorError> {
    shape.iter().try_fold(1u64, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or(NativeTensorError::InvalidShape)
    })
}

fn dtype_width(dtype: DType) -> usize {
    match dtype {
        DType::F32 | DType::I32 => 4,
        DType::F16 | DType::Bf16 => 2,
        DType::I64 => 8,
        DType::I8 | DType::U8 | DType::Bool => 1,
    }
}

enum ResolvedContent<'embedded, 'store> {
    Embedded(&'embedded [u8]),
    External(&'store [u8]),
}

impl ResolvedContent<'_, '_> {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Embedded(bytes) => bytes,
            Self::External(bytes) => bytes,
        }
    }
}

fn resolve_chunk<'embedded, 'store, S: ContentStore>(
    chunk: &ChunkView<'embedded>,
    store: &'store S,
) -> Result<ResolvedContent<'embedded, 'store>, NativeTensorError> {
    match chunk.storage {
        ChunkStorageView::Embedded(bytes) => Ok(ResolvedContent::Embedded(bytes)),
        ChunkStorageView::External => {
            let bytes = store
                .get(&chunk.hash)
                .ok_or(NativeTensorError::MissingExternalContent(chunk.hash))?;
            let actual = u64::try_from(bytes.len()).map_err(|_| NativeTensorError::Overflow)?;
            if actual != chunk.logical_len {
                return Err(NativeTensorError::ExternalLengthMismatch {
                    expected: chunk.logical_len,
                    actual,
                });
            }
            if sha256(bytes) != chunk.hash {
                return Err(NativeTensorError::ExternalIntegrityMismatch(chunk.hash));
            }
            Ok(ResolvedContent::External(bytes))
        }
    }
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

fn decode_dtype(tag: u8) -> Result<DType, NativeTensorError> {
    match tag {
        1 => Ok(DType::F32),
        2 => Ok(DType::F16),
        3 => Ok(DType::Bf16),
        4 => Ok(DType::I8),
        5 => Ok(DType::U8),
        6 => Ok(DType::I32),
        7 => Ok(DType::I64),
        8 => Ok(DType::Bool),
        other => Err(NativeTensorError::InvalidDType(other)),
    }
}

fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_i32(out: &mut Vec<u8>, value: i32) {
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

    fn take(&mut self, len: usize) -> Result<&'a [u8], NativeTensorError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(NativeTensorError::Overflow)?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or(NativeTensorError::Truncated)?;
        self.offset = end;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8, NativeTensorError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, NativeTensorError> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> Result<u32, NativeTensorError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn i32(&mut self) -> Result<i32, NativeTensorError> {
        let bytes = self.take(4)?;
        Ok(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn u64(&mut self) -> Result<u64, NativeTensorError> {
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
    use crate::{push_graph_section, CapsuleKind};
    use ntd_ir::{IrVersion, ValueDecl};

    fn tensor_decl(id: u32, dtype: DType, rank: u8) -> ValueDecl {
        ValueDecl {
            id: ValueId(id),
            ty: ValueType::Tensor { dtype, rank },
        }
    }

    #[test]
    fn scalar_descriptor_binds_scalar_graph_input() {
        let graph = Graph {
            version: IrVersion::CURRENT,
            inputs: vec![ValueDecl {
                id: ValueId(0),
                ty: ValueType::Scalar(DType::F32),
            }],
            outputs: vec![ValueId(0)],
            nodes: Vec::new(),
        };
        let descriptor = TensorDescriptor {
            id: 1,
            dtype: DType::F32,
            shape: Vec::new(),
            byte_len: 4,
        };

        assert_eq!(validate_graph_binding(&graph, ValueId(0), &descriptor), Ok(()));
    }

    #[test]
    fn native_tensor_round_trips_deterministically() {
        let tensor = NativeTensor {
            descriptor: TensorDescriptor {
                id: 7,
                dtype: DType::I8,
                shape: vec![2, 2],
                byte_len: 4,
            },
            graph_value: Some(ValueId(1)),
            quantization: QuantizationMetadata::symmetric_i8(0.25).expect("quantization"),
            payload: vec![1, 2, 3, 4],
        };
        let first = encode_native_tensor(&tensor).expect("first");
        let second = encode_native_tensor(&tensor).expect("second");
        assert_eq!(first, second);
        assert_eq!(decode_native_tensor(&first).expect("decode"), tensor);
    }

    #[test]
    fn thin_capsule_resolves_external_tensor_by_content_hash() {
        let graph = Graph {
            version: IrVersion::CURRENT,
            inputs: vec![tensor_decl(0, DType::F32, 2)],
            outputs: vec![ValueId(0)],
            nodes: Vec::new(),
        };
        let tensor = NativeTensor {
            descriptor: TensorDescriptor {
                id: 3,
                dtype: DType::F32,
                shape: vec![1, 1],
                byte_len: 4,
            },
            graph_value: Some(ValueId(0)),
            quantization: QuantizationMetadata::None,
            payload: 2.0f32.to_le_bytes().to_vec(),
        };
        let shard = encode_native_tensor(&tensor).expect("shard");
        let mut store = MemoryContentStore::default();
        let hash = store.insert(shard.clone());

        let mut builder = CapsuleBuilder::new(CapsuleKind::Thin, *b"NTD97-NATIVE-001");
        push_graph_section(&mut builder, &graph).expect("graph");
        push_tensor_descriptor_section(&mut builder, std::slice::from_ref(&tensor.descriptor))
            .expect("descriptors");
        builder.push_external(
            SectionKind::Tensors,
            u64::try_from(shard.len()).expect("len"),
            hash,
        );

        let bytes = builder.write().expect("capsule");
        let view = CapsuleView::read(&bytes).expect("view");
        let program = load_native_program(&view, &store).expect("program");
        assert_eq!(program.graph, graph);
        assert_eq!(program.tensors, vec![tensor]);
        assert_eq!(store.len(), 1);
    }
}
