#![forbid(unsafe_code)]

use std::collections::BTreeSet;

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use ntd_capsule::{
    decode_graph_section, encode_native_generative_manifest, encode_native_tokenizer,
    encode_tensor_descriptors, load_native_generative_program, sha256, CapsuleBuilder, CapsuleKind,
    CapsuleView, ChunkStorageView, Digest, MemoryContentStore, NativeGenerativeManifest,
    SectionKind,
};
use ntd_core::{CapabilityId, SideEffectClass};
use ntd_runtime::{AuthorityScope, CapabilityDescriptor, CapabilityDomain};

use crate::activation::activate_thin_generative_capsule;
use crate::gguf::{FileTensorShardStore, StreamedLoweredLlamaModel};
use crate::{AssetKind, AssimilationError, NativeCandidate, ProvenanceRecord, SandboxReport};

const SIGNATURE_MAGIC: [u8; 6] = *b"NAS97\0";
const SIGNATURE_MAJOR: u16 = 0;
const SIGNATURE_MINOR: u16 = 1;
const SIGNATURE_LEN: usize = 106;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssimilationIdentity {
    signing_seed: [u8; 32],
    verify_key: [u8; 32],
}

impl AssimilationIdentity {
    pub fn from_seed(signing_seed: [u8; 32]) -> Self {
        let signing = SigningKey::from_bytes(&signing_seed);
        Self {
            signing_seed,
            verify_key: signing.verifying_key().to_bytes(),
        }
    }

    pub fn verify_key(&self) -> [u8; 32] {
        self.verify_key
    }

    fn sign(&self, bytes: &[u8]) -> [u8; 64] {
        SigningKey::from_bytes(&self.signing_seed)
            .sign(bytes)
            .to_bytes()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativePackage {
    pub asset_id: String,
    pub version: u32,
    pub kind: AssetKind,
    pub native_capsule: Vec<u8>,
    pub capsule_hash: Digest,
    pub capability: Option<CapabilityDescriptor>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamedPackageSpec {
    pub asset_id: String,
    pub version: u32,
    pub importer_id: String,
}

impl StreamedPackageSpec {
    pub fn new(
        asset_id: impl Into<String>,
        version: u32,
        importer_id: impl Into<String>,
    ) -> Result<Self, AssimilationError> {
        let asset_id = asset_id.into().trim().to_owned();
        let importer_id = importer_id.into().trim().to_owned();
        if asset_id.is_empty() || version == 0 || importer_id.is_empty() {
            return Err(AssimilationError::InvalidPackage);
        }
        Ok(Self {
            asset_id,
            version,
            importer_id,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ThinPackageChunk {
    Embedded(SectionKind, Vec<u8>),
    External {
        kind: SectionKind,
        logical_len: u64,
        hash: Digest,
    },
}

impl ThinPackageChunk {
    fn kind(&self) -> SectionKind {
        match self {
            Self::Embedded(kind, _) | Self::External { kind, .. } => *kind,
        }
    }

    fn logical_len(&self) -> Result<u64, AssimilationError> {
        match self {
            Self::Embedded(_, bytes) => {
                u64::try_from(bytes.len()).map_err(|_| AssimilationError::Overflow)
            }
            Self::External { logical_len, .. } => Ok(*logical_len),
        }
    }

    fn hash(&self) -> Digest {
        match self {
            Self::Embedded(_, bytes) => sha256(bytes),
            Self::External { hash, .. } => *hash,
        }
    }

    fn storage_tag(&self) -> u8 {
        match self {
            Self::Embedded(_, _) => 0,
            Self::External { .. } => 1,
        }
    }
}

pub fn build_native_package(
    candidate: &NativeCandidate,
    provenance: &ProvenanceRecord,
    importer_id: &str,
    report: &SandboxReport,
    identity: &AssimilationIdentity,
) -> Result<NativePackage, AssimilationError> {
    candidate.validate()?;
    provenance.validate()?;
    validate_report(candidate, report)?;
    if importer_id.trim().is_empty() {
        return Err(AssimilationError::InvalidPackage);
    }

    let provenance_bytes = encode_provenance(provenance)?;
    let log_bytes = encode_assimilation_log(
        candidate.asset_id(),
        candidate.version(),
        candidate.kind(),
        importer_id,
        &report.passed_regressions,
    )?;

    let mut sections = Vec::new();
    let capability = match candidate {
        NativeCandidate::Capability {
            descriptor,
            adapter,
            ..
        } => {
            sections.push((
                SectionKind::Capabilities,
                encode_capability_descriptor(descriptor)?,
            ));
            sections.push((
                SectionKind::Adapters,
                encode_adapter(&adapter.format, &adapter.bytes)?,
            ));
            Some(descriptor.clone())
        }
        NativeCandidate::Intelligence {
            graph,
            sections: extra,
            ..
        } => {
            sections.push((
                SectionKind::Graph,
                ntd_capsule::encode_graph(graph)
                    .map_err(|error| AssimilationError::Capsule(format!("{error:?}")))?,
            ));
            for section in extra {
                sections.push((section.kind, section.bytes.clone()));
            }
            None
        }
    };
    sections.push((SectionKind::Provenance, provenance_bytes));
    sections.push((SectionKind::AssimilationLog, log_bytes));

    let signable = section_signing_payload(&sections)?;
    let signature = identity.sign(&signable);
    sections.push((
        SectionKind::Signatures,
        encode_signature(identity.verify_key(), signature),
    ));

    let mut id_material = Vec::new();
    push_string(&mut id_material, candidate.asset_id())?;
    push_u32(&mut id_material, candidate.version());
    id_material.extend_from_slice(&provenance.source_digest);
    let id_digest = sha256(&id_material);
    let mut capsule_id = [0u8; 16];
    capsule_id.copy_from_slice(&id_digest[..16]);

    let mut builder = CapsuleBuilder::new(CapsuleKind::Full, capsule_id);
    for (kind, bytes) in sections {
        builder.push_embedded(kind, bytes);
    }
    let native_capsule = builder
        .write()
        .map_err(|error| AssimilationError::Capsule(format!("{error:?}")))?;
    let package = NativePackage {
        asset_id: candidate.asset_id().trim().to_owned(),
        version: candidate.version(),
        kind: candidate.kind(),
        capsule_hash: sha256(&native_capsule),
        native_capsule,
        capability,
    };
    verify_native_package(&package, &identity.verify_key())?;
    Ok(package)
}

pub fn build_streamed_native_package(
    spec: StreamedPackageSpec,
    model: &StreamedLoweredLlamaModel,
    provenance: &ProvenanceRecord,
    report: &SandboxReport,
    shard_store: &FileTensorShardStore,
    identity: &AssimilationIdentity,
) -> Result<NativePackage, AssimilationError> {
    provenance.validate()?;
    model
        .graph
        .validate()
        .map_err(|error| AssimilationError::InvalidCandidate(format!("{error:?}")))?;
    if model.tensor_shards.is_empty() || model.vocabulary_size == 0 {
        return Err(AssimilationError::InvalidPackage);
    }
    validate_streamed_report(report)?;

    let descriptors = model
        .tensor_shards
        .iter()
        .map(|shard| shard.descriptor.clone())
        .collect::<Vec<_>>();
    let descriptor_bytes = encode_tensor_descriptors(&descriptors)
        .map_err(|error| AssimilationError::Capsule(format!("{error:?}")))?;
    let tokenizer_bytes = encode_native_tokenizer(&model.tokenizer)
        .map_err(|error| AssimilationError::Capsule(format!("{error:?}")))?;
    let manifest_bytes = encode_native_generative_manifest(NativeGenerativeManifest {
        token_input: model.token_input,
        distribution_output: u32::try_from(model.distribution_output)
            .map_err(|_| AssimilationError::Overflow)?,
        vocabulary_size: u32::try_from(model.vocabulary_size)
            .map_err(|_| AssimilationError::Overflow)?,
    })
    .map_err(|error| AssimilationError::Capsule(format!("{error:?}")))?;
    let graph_bytes = ntd_capsule::encode_graph(&model.graph)
        .map_err(|error| AssimilationError::Capsule(format!("{error:?}")))?;
    let provenance_bytes = encode_provenance(provenance)?;
    let log_bytes = encode_assimilation_log(
        &spec.asset_id,
        spec.version,
        AssetKind::Intelligence,
        &spec.importer_id,
        &report.passed_regressions,
    )?;

    let mut chunks = Vec::with_capacity(model.tensor_shards.len().saturating_add(7));
    chunks.push(ThinPackageChunk::Embedded(SectionKind::Graph, graph_bytes));
    chunks.push(ThinPackageChunk::Embedded(
        SectionKind::Tensors,
        descriptor_bytes,
    ));
    for shard in &model.tensor_shards {
        chunks.push(ThinPackageChunk::External {
            kind: SectionKind::Tensors,
            logical_len: shard.logical_len,
            hash: shard.hash,
        });
    }
    chunks.push(ThinPackageChunk::Embedded(
        SectionKind::Tokenizer,
        tokenizer_bytes,
    ));
    chunks.push(ThinPackageChunk::Embedded(
        SectionKind::GenerativeManifest,
        manifest_bytes,
    ));
    chunks.push(ThinPackageChunk::Embedded(
        SectionKind::Provenance,
        provenance_bytes,
    ));
    chunks.push(ThinPackageChunk::Embedded(
        SectionKind::AssimilationLog,
        log_bytes,
    ));

    let signable = thin_section_signing_payload(&chunks)?;
    let signature = identity.sign(&signable);
    chunks.push(ThinPackageChunk::Embedded(
        SectionKind::Signatures,
        encode_signature(identity.verify_key(), signature),
    ));

    let capsule_id = native_capsule_id(&spec.asset_id, spec.version, &provenance.source_digest)?;
    let mut builder = CapsuleBuilder::new(CapsuleKind::Thin, capsule_id);
    for chunk in chunks {
        match chunk {
            ThinPackageChunk::Embedded(kind, bytes) => builder.push_embedded(kind, bytes),
            ThinPackageChunk::External {
                kind,
                logical_len,
                hash,
            } => builder.push_external(kind, logical_len, hash),
        }
    }
    let native_capsule = builder
        .write()
        .map_err(|error| AssimilationError::Capsule(format!("{error:?}")))?;
    let package = NativePackage {
        asset_id: spec.asset_id,
        version: spec.version,
        kind: AssetKind::Intelligence,
        capsule_hash: sha256(&native_capsule),
        native_capsule,
        capability: None,
    };
    verify_native_package_with_shards(&package, &identity.verify_key(), shard_store)?;
    Ok(package)
}

pub fn verify_native_package(
    package: &NativePackage,
    trusted_verify_key: &[u8; 32],
) -> Result<(), AssimilationError> {
    if package.asset_id.trim().is_empty()
        || package.version == 0
        || sha256(&package.native_capsule) != package.capsule_hash
    {
        return Err(AssimilationError::InvalidPackage);
    }

    let view = CapsuleView::read(&package.native_capsule)
        .map_err(|error| AssimilationError::Capsule(format!("{error:?}")))?;
    if view.kind != CapsuleKind::Full {
        return Err(AssimilationError::InvalidPackage);
    }

    let mut signable_sections = Vec::new();
    let mut signature = None;
    let mut log = None;
    for chunk in &view.chunks {
        let ChunkStorageView::Embedded(bytes) = chunk.storage else {
            return Err(AssimilationError::InvalidPackage);
        };
        if chunk.kind == SectionKind::Signatures {
            if signature.replace(decode_signature(bytes)?).is_some() {
                return Err(AssimilationError::InvalidPackage);
            }
        } else {
            signable_sections.push((chunk.kind, bytes.to_vec()));
        }
        if chunk.kind == SectionKind::AssimilationLog {
            log = Some(decode_assimilation_log(bytes)?);
        }
    }

    let (verify_key, signature) = signature.ok_or(AssimilationError::InvalidSignature)?;
    if &verify_key != trusted_verify_key {
        return Err(AssimilationError::UntrustedSigner);
    }
    let message = section_signing_payload(&signable_sections)?;
    let key =
        VerifyingKey::from_bytes(&verify_key).map_err(|_| AssimilationError::InvalidSignature)?;
    key.verify(&message, &Signature::from_bytes(&signature))
        .map_err(|_| AssimilationError::InvalidSignature)?;

    let (asset_id, version, kind) = log.ok_or(AssimilationError::InvalidPackage)?;
    if asset_id != package.asset_id || version != package.version || kind != package.kind {
        return Err(AssimilationError::InvalidPackage);
    }

    match package.kind {
        AssetKind::Capability => {
            let descriptor = load_native_capability(&package.native_capsule)?;
            if package.capability.as_ref() != Some(&descriptor) {
                return Err(AssimilationError::InvalidPackage);
            }
        }
        AssetKind::Intelligence => {
            if package.capability.is_some() {
                return Err(AssimilationError::InvalidPackage);
            }
            let graph_chunks = view
                .chunks
                .iter()
                .filter(|chunk| chunk.kind == SectionKind::Graph)
                .collect::<Vec<_>>();
            if graph_chunks.len() != 1 {
                return Err(AssimilationError::InvalidPackage);
            }
            decode_graph_section(graph_chunks[0])
                .map_err(|error| AssimilationError::Capsule(format!("{error:?}")))?;

            let has_tensors = view
                .chunks
                .iter()
                .any(|chunk| chunk.kind == SectionKind::Tensors);
            let has_tokenizer = view
                .chunks
                .iter()
                .any(|chunk| chunk.kind == SectionKind::Tokenizer);
            if has_tensors || has_tokenizer {
                if !has_tensors || !has_tokenizer {
                    return Err(AssimilationError::InvalidPackage);
                }
                load_native_generative_program(&view, &MemoryContentStore::default())
                    .map_err(|error| AssimilationError::Capsule(format!("{error:?}")))?;
            }
        }
    }
    Ok(())
}

pub fn verify_native_package_with_shards(
    package: &NativePackage,
    trusted_verify_key: &[u8; 32],
    shard_store: &FileTensorShardStore,
) -> Result<(), AssimilationError> {
    if package.asset_id.trim().is_empty()
        || package.version == 0
        || sha256(&package.native_capsule) != package.capsule_hash
        || package.kind != AssetKind::Intelligence
        || package.capability.is_some()
    {
        return Err(AssimilationError::InvalidPackage);
    }

    let view = CapsuleView::read(&package.native_capsule)
        .map_err(|error| AssimilationError::Capsule(format!("{error:?}")))?;
    if view.kind == CapsuleKind::Full {
        return verify_native_package(package, trusted_verify_key);
    }
    if view.kind != CapsuleKind::Thin {
        return Err(AssimilationError::InvalidPackage);
    }

    let mut signable = Vec::new();
    let mut signature = None;
    let mut log = None;
    for chunk in &view.chunks {
        if chunk.kind == SectionKind::Signatures {
            let ChunkStorageView::Embedded(bytes) = chunk.storage else {
                return Err(AssimilationError::InvalidPackage);
            };
            if signature.replace(decode_signature(bytes)?).is_some() {
                return Err(AssimilationError::InvalidPackage);
            }
            continue;
        }

        match chunk.storage {
            ChunkStorageView::Embedded(bytes) => {
                signable.push(ThinPackageChunk::Embedded(chunk.kind, bytes.to_vec()));
                if chunk.kind == SectionKind::AssimilationLog
                    && log.replace(decode_assimilation_log(bytes)?).is_some()
                {
                    return Err(AssimilationError::InvalidPackage);
                }
            }
            ChunkStorageView::External => {
                if chunk.kind != SectionKind::Tensors {
                    return Err(AssimilationError::InvalidPackage);
                }
                signable.push(ThinPackageChunk::External {
                    kind: chunk.kind,
                    logical_len: chunk.logical_len,
                    hash: chunk.hash,
                });
            }
        }
    }

    let (verify_key, signature) = signature.ok_or(AssimilationError::InvalidSignature)?;
    if &verify_key != trusted_verify_key {
        return Err(AssimilationError::UntrustedSigner);
    }
    let message = thin_section_signing_payload(&signable)?;
    let key =
        VerifyingKey::from_bytes(&verify_key).map_err(|_| AssimilationError::InvalidSignature)?;
    key.verify(&message, &Signature::from_bytes(&signature))
        .map_err(|_| AssimilationError::InvalidSignature)?;

    let (asset_id, version, kind) = log.ok_or(AssimilationError::InvalidPackage)?;
    if asset_id != package.asset_id || version != package.version || kind != AssetKind::Intelligence
    {
        return Err(AssimilationError::InvalidPackage);
    }

    activate_thin_generative_capsule(&package.native_capsule, shard_store)
        .map_err(|error| AssimilationError::Capsule(format!("{error:?}")))?;
    Ok(())
}

pub fn load_native_capability(
    native_capsule: &[u8],
) -> Result<CapabilityDescriptor, AssimilationError> {
    let view = CapsuleView::read(native_capsule)
        .map_err(|error| AssimilationError::Capsule(format!("{error:?}")))?;
    let chunks = view
        .chunks
        .iter()
        .filter(|chunk| chunk.kind == SectionKind::Capabilities)
        .collect::<Vec<_>>();
    if chunks.len() != 1 {
        return Err(AssimilationError::InvalidPackage);
    }
    let ChunkStorageView::Embedded(bytes) = chunks[0].storage else {
        return Err(AssimilationError::InvalidPackage);
    };
    decode_capability_descriptor(bytes)
}

fn validate_report(
    candidate: &NativeCandidate,
    report: &SandboxReport,
) -> Result<(), AssimilationError> {
    if !report.isolated || report.network_used || report.external_write_used {
        return Err(AssimilationError::SandboxNotIsolated);
    }
    for regression in candidate.regressions() {
        if !report
            .passed_regressions
            .iter()
            .any(|name| name == regression.name.trim())
        {
            return Err(AssimilationError::RegressionMissing(
                regression.name.clone(),
            ));
        }
    }
    Ok(())
}

fn validate_streamed_report(report: &SandboxReport) -> Result<(), AssimilationError> {
    if !report.isolated || report.network_used || report.external_write_used {
        return Err(AssimilationError::SandboxNotIsolated);
    }
    let unique = report
        .passed_regressions
        .iter()
        .map(|name| name.trim().to_owned())
        .collect::<BTreeSet<_>>();
    if unique.len() != report.passed_regressions.len() {
        return Err(AssimilationError::InvalidPackage);
    }
    if !unique.contains("nir97-roundtrip") {
        return Err(AssimilationError::RegressionMissing(
            "nir97-roundtrip".into(),
        ));
    }
    Ok(())
}

fn native_capsule_id(
    asset_id: &str,
    version: u32,
    source_digest: &Digest,
) -> Result<[u8; 16], AssimilationError> {
    let mut material = Vec::new();
    push_string(&mut material, asset_id)?;
    push_u32(&mut material, version);
    material.extend_from_slice(source_digest);
    let digest = sha256(&material);
    let mut capsule_id = [0u8; 16];
    capsule_id.copy_from_slice(&digest[..16]);
    Ok(capsule_id)
}

fn thin_section_signing_payload(chunks: &[ThinPackageChunk]) -> Result<Vec<u8>, AssimilationError> {
    let count = u32::try_from(chunks.len()).map_err(|_| AssimilationError::Overflow)?;
    let mut out = b"NTD97-NATIVE-ASSET-SIGN-v2".to_vec();
    push_u32(&mut out, count);
    for chunk in chunks {
        push_u16(&mut out, chunk.kind() as u16);
        out.push(chunk.storage_tag());
        out.push(0);
        push_u64(&mut out, chunk.logical_len()?);
        out.extend_from_slice(&chunk.hash());
    }
    Ok(out)
}

fn section_signing_payload(
    sections: &[(SectionKind, Vec<u8>)],
) -> Result<Vec<u8>, AssimilationError> {
    let count = u32::try_from(sections.len()).map_err(|_| AssimilationError::Overflow)?;
    let mut out = b"NTD97-NATIVE-ASSET-SIGN-v1".to_vec();
    push_u32(&mut out, count);
    for (kind, bytes) in sections {
        push_u16(&mut out, *kind as u16);
        push_u64(
            &mut out,
            u64::try_from(bytes.len()).map_err(|_| AssimilationError::Overflow)?,
        );
        out.extend_from_slice(&sha256(bytes));
    }
    Ok(out)
}

fn encode_signature(verify_key: [u8; 32], signature: [u8; 64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(SIGNATURE_LEN);
    out.extend_from_slice(&SIGNATURE_MAGIC);
    push_u16(&mut out, SIGNATURE_MAJOR);
    push_u16(&mut out, SIGNATURE_MINOR);
    out.extend_from_slice(&verify_key);
    out.extend_from_slice(&signature);
    out
}

fn decode_signature(bytes: &[u8]) -> Result<([u8; 32], [u8; 64]), AssimilationError> {
    if bytes.len() != SIGNATURE_LEN
        || bytes[..6] != SIGNATURE_MAGIC
        || u16::from_le_bytes([bytes[6], bytes[7]]) != SIGNATURE_MAJOR
        || u16::from_le_bytes([bytes[8], bytes[9]]) > SIGNATURE_MINOR
    {
        return Err(AssimilationError::InvalidSignature);
    }
    let mut key = [0u8; 32];
    key.copy_from_slice(&bytes[10..42]);
    let mut signature = [0u8; 64];
    signature.copy_from_slice(&bytes[42..106]);
    Ok((key, signature))
}

fn encode_provenance(provenance: &ProvenanceRecord) -> Result<Vec<u8>, AssimilationError> {
    let mut out = b"NPR97\0".to_vec();
    push_u16(&mut out, 0);
    push_u16(&mut out, 1);
    push_string(&mut out, &provenance.source_uri)?;
    out.extend_from_slice(&provenance.source_digest);
    push_string(&mut out, &provenance.license.spdx_id)?;
    push_string(&mut out, &provenance.license.notice)?;
    push_string(&mut out, &provenance.attribution)?;
    Ok(out)
}

fn encode_assimilation_log(
    asset_id: &str,
    version: u32,
    kind: AssetKind,
    importer_id: &str,
    regressions: &[String],
) -> Result<Vec<u8>, AssimilationError> {
    let mut out = b"NAL97\0".to_vec();
    push_u16(&mut out, 0);
    push_u16(&mut out, 1);
    out.push(match kind {
        AssetKind::Capability => 1,
        AssetKind::Intelligence => 2,
    });
    push_string(&mut out, asset_id)?;
    push_u32(&mut out, version);
    push_string(&mut out, importer_id)?;
    push_u32(
        &mut out,
        u32::try_from(regressions.len()).map_err(|_| AssimilationError::Overflow)?,
    );
    for regression in regressions {
        push_string(&mut out, regression)?;
    }
    Ok(out)
}

fn decode_assimilation_log(bytes: &[u8]) -> Result<(String, u32, AssetKind), AssimilationError> {
    let mut cursor = Cursor::new(bytes);
    if cursor.take(6)? != b"NAL97\0" || cursor.u16()? != 0 || cursor.u16()? > 1 {
        return Err(AssimilationError::InvalidPackage);
    }
    let kind = match cursor.u8()? {
        1 => AssetKind::Capability,
        2 => AssetKind::Intelligence,
        _ => return Err(AssimilationError::InvalidPackage),
    };
    let asset_id = cursor.string()?;
    let version = cursor.u32()?;
    let _importer_id = cursor.string()?;
    let regression_count = cursor.u32()?;
    for _ in 0..regression_count {
        let _ = cursor.string()?;
    }
    if !cursor.is_finished() {
        return Err(AssimilationError::InvalidPackage);
    }
    Ok((asset_id, version, kind))
}

fn encode_adapter(format: &str, bytes: &[u8]) -> Result<Vec<u8>, AssimilationError> {
    if format.trim().is_empty() || bytes.is_empty() {
        return Err(AssimilationError::InvalidPackage);
    }
    let mut out = b"NAD97\0".to_vec();
    push_u16(&mut out, 0);
    push_u16(&mut out, 1);
    push_string(&mut out, format)?;
    push_bytes(&mut out, bytes)?;
    Ok(out)
}

fn encode_capability_descriptor(
    descriptor: &CapabilityDescriptor,
) -> Result<Vec<u8>, AssimilationError> {
    let mut descriptor = descriptor.clone();
    descriptor
        .normalize()
        .map_err(|error| AssimilationError::InvalidCandidate(format!("{error:?}")))?;
    let mut out = b"NCP97\0".to_vec();
    push_u16(&mut out, 0);
    push_u16(&mut out, 1);
    push_string(&mut out, &descriptor.id.0)?;
    push_u32(&mut out, descriptor.version);
    out.push(descriptor.domain as u8);
    out.push(match descriptor.side_effect {
        SideEffectClass::ReadOnly => 1,
        SideEffectClass::Reversible => 2,
        SideEffectClass::ExternalWrite => 3,
        SideEffectClass::Irreversible => 4,
    });
    out.push(u8::from(descriptor.verification_required));
    out.push(u8::from(descriptor.rollback_supported));
    out.push(u8::from(descriptor.resumable));
    push_u32(
        &mut out,
        u32::try_from(descriptor.required_scopes.len()).map_err(|_| AssimilationError::Overflow)?,
    );
    for scope in &descriptor.required_scopes {
        push_string(&mut out, scope.as_str())?;
    }
    Ok(out)
}

fn decode_capability_descriptor(bytes: &[u8]) -> Result<CapabilityDescriptor, AssimilationError> {
    let mut cursor = Cursor::new(bytes);
    if cursor.take(6)? != b"NCP97\0" || cursor.u16()? != 0 || cursor.u16()? > 1 {
        return Err(AssimilationError::InvalidPackage);
    }
    let id = CapabilityId(cursor.string()?);
    let version = cursor.u32()?;
    let domain = CapabilityDomain::try_from(cursor.u8()?)
        .map_err(|error| AssimilationError::InvalidCandidate(format!("{error:?}")))?;
    let side_effect = match cursor.u8()? {
        1 => SideEffectClass::ReadOnly,
        2 => SideEffectClass::Reversible,
        3 => SideEffectClass::ExternalWrite,
        4 => SideEffectClass::Irreversible,
        _ => return Err(AssimilationError::InvalidPackage),
    };
    let verification_required = cursor.boolean()?;
    let rollback_supported = cursor.boolean()?;
    let resumable = cursor.boolean()?;
    let scope_count = cursor.u32()?;
    let mut required_scopes = Vec::new();
    for _ in 0..scope_count {
        required_scopes.push(
            AuthorityScope::new(cursor.string()?)
                .map_err(|error| AssimilationError::InvalidCandidate(format!("{error:?}")))?,
        );
    }
    if !cursor.is_finished() {
        return Err(AssimilationError::InvalidPackage);
    }
    let mut descriptor = CapabilityDescriptor::new(id, version, domain, side_effect)
        .map_err(|error| AssimilationError::InvalidCandidate(format!("{error:?}")))?;
    descriptor.verification_required = verification_required;
    descriptor.rollback_supported = rollback_supported;
    descriptor.resumable = resumable;
    descriptor.required_scopes = required_scopes;
    descriptor
        .normalize()
        .map_err(|error| AssimilationError::InvalidCandidate(format!("{error:?}")))?;
    Ok(descriptor)
}

fn push_bytes(out: &mut Vec<u8>, bytes: &[u8]) -> Result<(), AssimilationError> {
    push_u32(
        out,
        u32::try_from(bytes.len()).map_err(|_| AssimilationError::Overflow)?,
    );
    out.extend_from_slice(bytes);
    Ok(())
}

fn push_string(out: &mut Vec<u8>, value: &str) -> Result<(), AssimilationError> {
    push_bytes(out, value.as_bytes())
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

    fn take(&mut self, len: usize) -> Result<&'a [u8], AssimilationError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(AssimilationError::Overflow)?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or(AssimilationError::InvalidPackage)?;
        self.offset = end;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8, AssimilationError> {
        Ok(self.take(1)?[0])
    }

    fn boolean(&mut self) -> Result<bool, AssimilationError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(AssimilationError::InvalidPackage),
        }
    }

    fn u16(&mut self) -> Result<u16, AssimilationError> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> Result<u32, AssimilationError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn bytes(&mut self) -> Result<&'a [u8], AssimilationError> {
        let len = usize::try_from(self.u32()?).map_err(|_| AssimilationError::Overflow)?;
        self.take(len)
    }

    fn string(&mut self) -> Result<String, AssimilationError> {
        String::from_utf8(self.bytes()?.to_vec()).map_err(|_| AssimilationError::InvalidPackage)
    }

    fn is_finished(&self) -> bool {
        self.offset == self.bytes.len()
    }
}
