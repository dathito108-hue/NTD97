#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use ntd_capsule::{
    decode_graph, decode_native_generative_manifest, decode_native_tensor, decode_native_tokenizer,
    decode_tensor_descriptors, CapsuleKind, CapsuleView, ChunkStorageView, NativeGenerativeManifest,
    NativeTokenizerDescriptor, QuantizationMetadata, SectionKind, TensorDescriptor,
    TENSOR_DESCRIPTOR_MAGIC,
};
use ntd_ir::{DType, Graph, ValueId, ValueType};
use ntd_runtime::{
    QuantizationParams, Tensor, TensorLoader, TensorResolveError, TensorResolver,
};

use crate::{FileTensorShardStore, GgufError, TensorShardRef};

#[derive(Debug, Clone, PartialEq)]
pub struct ThinGenerativeActivation {
    pub graph: Graph,
    pub tokenizer: NativeTokenizerDescriptor,
    pub manifest: NativeGenerativeManifest,
    pub tensor_shards: BTreeMap<ValueId, TensorShardRef>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThinActivationError {
    Capsule(String),
    WrongCapsuleKind(CapsuleKind),
    MissingGraph,
    MultipleGraphs,
    MissingDescriptorTable,
    MultipleDescriptorTables,
    MissingTokenizer,
    MultipleTokenizers,
    MissingGenerativeManifest,
    MultipleGenerativeManifests,
    UnexpectedEmbeddedTensorChunk,
    UnexpectedExternalChunk(SectionKind),
    DuplicateTensorId(u32),
    DuplicateGraphBinding(ValueId),
    MissingTensorShard(u32),
    UnknownTensorShard(u32),
    TensorDescriptorMismatch(u32),
    MissingGraphBinding(u32),
    GraphBindingMissing(ValueId),
    GraphBindingTypeMismatch(ValueId),
    InvalidTokenInput(ValueId),
    InvalidDistributionOutput(u32),
    VocabularyMismatch { manifest: u32, tokenizer: usize },
    Shard(String),
}

#[derive(Debug, Clone)]
pub struct FileBackedTensorResolver {
    store: FileTensorShardStore,
    tensor_shards: BTreeMap<ValueId, TensorShardRef>,
}

impl FileBackedTensorResolver {
    pub fn new(
        store: FileTensorShardStore,
        tensor_shards: BTreeMap<ValueId, TensorShardRef>,
    ) -> Self {
        Self {
            store,
            tensor_shards,
        }
    }

    pub fn from_activation(
        store: FileTensorShardStore,
        activation: &ThinGenerativeActivation,
    ) -> Self {
        Self::new(store, activation.tensor_shards.clone())
    }

    pub fn binding_count(&self) -> usize {
        self.tensor_shards.len()
    }
}

impl TensorResolver for FileBackedTensorResolver {
    fn resolve(&self, value: ValueId) -> Result<Option<Tensor>, TensorResolveError> {
        let Some(reference) = self.tensor_shards.get(&value) else {
            return Ok(None);
        };

        let bytes = self
            .store
            .read_verified(reference)
            .map_err(|_| TensorResolveError::Invalid)?;
        let native = decode_native_tensor(&bytes).map_err(|_| TensorResolveError::Invalid)?;
        if native.descriptor != reference.descriptor || native.graph_value != Some(value) {
            return Err(TensorResolveError::Invalid);
        }

        let quantization = match native.quantization {
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
        };
        TensorLoader::load(
            native.descriptor.dtype,
            &native.descriptor.shape,
            quantization,
            &native.payload,
        )
        .map(Some)
        .map_err(|_| TensorResolveError::Invalid)
    }
}

pub fn activate_thin_generative_capsule(
    native_capsule: &[u8],
    store: &FileTensorShardStore,
) -> Result<ThinGenerativeActivation, ThinActivationError> {
    let view = CapsuleView::read(native_capsule)
        .map_err(|error| ThinActivationError::Capsule(format!("{error:?}")))?;
    if view.kind != CapsuleKind::Thin {
        return Err(ThinActivationError::WrongCapsuleKind(view.kind));
    }

    let graph = decode_single_embedded(
        &view,
        SectionKind::Graph,
        ThinActivationError::MissingGraph,
        ThinActivationError::MultipleGraphs,
        |bytes| {
            decode_graph(bytes)
                .map_err(|error| ThinActivationError::Capsule(format!("graph: {error:?}")))
        },
    )?;

    let tokenizer = decode_single_embedded(
        &view,
        SectionKind::Tokenizer,
        ThinActivationError::MissingTokenizer,
        ThinActivationError::MultipleTokenizers,
        |bytes| {
            decode_native_tokenizer(bytes)
                .map_err(|error| ThinActivationError::Capsule(format!("tokenizer: {error:?}")))
        },
    )?;

    let manifest = decode_single_embedded(
        &view,
        SectionKind::GenerativeManifest,
        ThinActivationError::MissingGenerativeManifest,
        ThinActivationError::MultipleGenerativeManifests,
        |bytes| {
            decode_native_generative_manifest(bytes).map_err(|error| {
                ThinActivationError::Capsule(format!("generative manifest: {error:?}"))
            })
        },
    )?;

    validate_manifest(&graph, &tokenizer, manifest)?;

    let tensor_chunks = view
        .chunks
        .iter()
        .filter(|chunk| chunk.kind == SectionKind::Tensors)
        .collect::<Vec<_>>();

    let mut descriptor_table = None;
    let mut external_chunks = Vec::new();
    for chunk in tensor_chunks {
        match chunk.storage {
            ChunkStorageView::Embedded(bytes) if bytes.starts_with(&TENSOR_DESCRIPTOR_MAGIC) => {
                if descriptor_table.is_some() {
                    return Err(ThinActivationError::MultipleDescriptorTables);
                }
                descriptor_table = Some(
                    decode_tensor_descriptors(bytes).map_err(|error| {
                        ThinActivationError::Capsule(format!("tensor descriptors: {error:?}"))
                    })?,
                );
            }
            ChunkStorageView::Embedded(_) => {
                return Err(ThinActivationError::UnexpectedEmbeddedTensorChunk);
            }
            ChunkStorageView::External => external_chunks.push(chunk),
        }
    }

    let descriptors = descriptor_table.ok_or(ThinActivationError::MissingDescriptorTable)?;
    let descriptor_by_id = descriptor_map(descriptors)?;

    let mut seen_ids = BTreeSet::new();
    let mut tensor_shards = BTreeMap::new();
    for chunk in external_chunks {
        let bytes = store
            .read_hash_verified(&chunk.hash, chunk.logical_len)
            .map_err(shard_error)?;
        let native = decode_native_tensor(&bytes)
            .map_err(|error| ThinActivationError::Shard(format!("{error:?}")))?;
        let id = native.descriptor.id;
        if !seen_ids.insert(id) {
            return Err(ThinActivationError::DuplicateTensorId(id));
        }
        let descriptor = descriptor_by_id
            .get(&id)
            .ok_or(ThinActivationError::UnknownTensorShard(id))?;
        if descriptor != &native.descriptor {
            return Err(ThinActivationError::TensorDescriptorMismatch(id));
        }
        let value = native
            .graph_value
            .ok_or(ThinActivationError::MissingGraphBinding(id))?;
        validate_graph_binding(&graph, value, descriptor)?;
        if tensor_shards
            .insert(
                value,
                TensorShardRef {
                    descriptor: native.descriptor,
                    graph_value: Some(value),
                    logical_len: chunk.logical_len,
                    hash: chunk.hash,
                },
            )
            .is_some()
        {
            return Err(ThinActivationError::DuplicateGraphBinding(value));
        }
    }

    for id in descriptor_by_id.keys() {
        if !seen_ids.contains(id) {
            return Err(ThinActivationError::MissingTensorShard(*id));
        }
    }
    if tensor_shards.contains_key(&manifest.token_input) {
        return Err(ThinActivationError::InvalidTokenInput(manifest.token_input));
    }

    Ok(ThinGenerativeActivation {
        graph,
        tokenizer,
        manifest,
        tensor_shards,
    })
}

fn descriptor_map(
    descriptors: Vec<TensorDescriptor>,
) -> Result<BTreeMap<u32, TensorDescriptor>, ThinActivationError> {
    let mut output = BTreeMap::new();
    for descriptor in descriptors {
        let id = descriptor.id;
        if output.insert(id, descriptor).is_some() {
            return Err(ThinActivationError::DuplicateTensorId(id));
        }
    }
    Ok(output)
}

fn validate_manifest(
    graph: &Graph,
    tokenizer: &NativeTokenizerDescriptor,
    manifest: NativeGenerativeManifest,
) -> Result<(), ThinActivationError> {
    let token_decl = graph
        .inputs
        .iter()
        .find(|decl| decl.id == manifest.token_input)
        .ok_or(ThinActivationError::InvalidTokenInput(manifest.token_input))?;
    if token_decl.ty
        != (ValueType::Tensor {
            dtype: DType::I32,
            rank: 1,
        })
    {
        return Err(ThinActivationError::InvalidTokenInput(manifest.token_input));
    }

    let distribution_output = usize::try_from(manifest.distribution_output)
        .map_err(|_| ThinActivationError::InvalidDistributionOutput(manifest.distribution_output))?;
    if distribution_output >= graph.outputs.len() {
        return Err(ThinActivationError::InvalidDistributionOutput(
            manifest.distribution_output,
        ));
    }

    let tokenizer_vocab = tokenizer.tokens.len();
    if usize::try_from(manifest.vocabulary_size).ok() != Some(tokenizer_vocab) {
        return Err(ThinActivationError::VocabularyMismatch {
            manifest: manifest.vocabulary_size,
            tokenizer: tokenizer_vocab,
        });
    }
    Ok(())
}

fn validate_graph_binding(
    graph: &Graph,
    value: ValueId,
    descriptor: &TensorDescriptor,
) -> Result<(), ThinActivationError> {
    let decl = graph
        .inputs
        .iter()
        .find(|decl| decl.id == value)
        .ok_or(ThinActivationError::GraphBindingMissing(value))?;
    match decl.ty {
        ValueType::Scalar(dtype) if dtype == descriptor.dtype && descriptor.shape.is_empty() => {
            Ok(())
        }
        ValueType::Tensor { dtype, rank }
            if dtype == descriptor.dtype && usize::from(rank) == descriptor.shape.len() =>
        {
            Ok(())
        }
        _ => Err(ThinActivationError::GraphBindingTypeMismatch(value)),
    }
}

fn decode_single_embedded<T>(
    view: &CapsuleView<'_>,
    kind: SectionKind,
    missing: ThinActivationError,
    multiple: ThinActivationError,
    decode: impl FnOnce(&[u8]) -> Result<T, ThinActivationError>,
) -> Result<T, ThinActivationError> {
    let chunks = view
        .chunks
        .iter()
        .filter(|chunk| chunk.kind == kind)
        .collect::<Vec<_>>();
    let chunk = match chunks.as_slice() {
        [] => return Err(missing),
        [chunk] => *chunk,
        _ => return Err(multiple),
    };
    let ChunkStorageView::Embedded(bytes) = chunk.storage else {
        return Err(ThinActivationError::UnexpectedExternalChunk(kind));
    };
    decode(bytes)
}

fn shard_error(error: GgufError) -> ThinActivationError {
    ThinActivationError::Shard(format!("{error:?}"))
}

#[cfg(test)]
mod tests {
    use ntd_capsule::{
        encode_native_generative_manifest, encode_native_tokenizer, encode_tensor_descriptors,
        CapsuleBuilder, NativeTokenizerModel,
    };
    use ntd_ir::{IrVersion, ValueDecl};

    use super::*;

    #[test]
    fn rejects_manifest_vocab_drift_before_tensor_resolution() {
        let graph = Graph {
            version: IrVersion::CURRENT,
            inputs: vec![ValueDecl {
                id: ValueId(0),
                ty: ValueType::Tensor {
                    dtype: DType::I32,
                    rank: 1,
                },
            }],
            outputs: vec![ValueId(0)],
            nodes: vec![],
        };
        let tokenizer = NativeTokenizerDescriptor {
            tokens: vec![b"a".to_vec()],
            bos_token: None,
            eos_token: None,
            unknown_token: None,
            model: NativeTokenizerModel::Vocabulary,
        };
        let manifest = NativeGenerativeManifest {
            token_input: ValueId(0),
            distribution_output: 0,
            vocabulary_size: 2,
        };

        let mut builder = CapsuleBuilder::new(CapsuleKind::Thin, *b"NTD97-ACTIV-0001");
        builder.push_embedded(
            SectionKind::Graph,
            ntd_capsule::encode_graph(&graph).expect("graph"),
        );
        builder.push_embedded(
            SectionKind::Tokenizer,
            encode_native_tokenizer(&tokenizer).expect("tokenizer"),
        );
        builder.push_embedded(
            SectionKind::GenerativeManifest,
            encode_native_generative_manifest(manifest).expect("manifest"),
        );
        builder.push_embedded(
            SectionKind::Tensors,
            encode_tensor_descriptors(&[]).expect("descriptors"),
        );
        let capsule = builder.write().expect("capsule");

        let root = std::env::temp_dir().join(format!(
            "ntd97-activation-test-{}",
            std::process::id()
        ));
        let store = FileTensorShardStore::open(&root).expect("store");
        assert_eq!(
            activate_thin_generative_capsule(&capsule, &store),
            Err(ThinActivationError::VocabularyMismatch {
                manifest: 2,
                tokenizer: 1,
            })
        );
        let _ = std::fs::remove_dir_all(root);
    }
}
