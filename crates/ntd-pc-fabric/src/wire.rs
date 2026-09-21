#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use ntd_core::SideEffectClass;
use ntd_runtime::{ActionOutput, ActionValue};
use sha2::{Digest, Sha256};

use crate::{ArtifactChunk, ArtifactDescriptor, PcFabricError};

pub const PCF97_MAGIC: [u8; 6] = *b"PCF97\0";
pub const PCF97_MAJOR: u16 = 0;
pub const PCF97_MINOR: u16 = 1;
const PCF97_HEADER_LEN: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteCapability {
    pub id: String,
    pub version: u32,
    pub side_effect: SideEffectClass,
    pub verification_required: bool,
    pub rollback_supported: bool,
    pub resumable: bool,
    pub required_scopes: Vec<String>,
}

impl RemoteCapability {
    pub fn validate(&self) -> Result<(), PcFabricError> {
        if self.id.trim().is_empty() || self.version == 0 {
            return Err(PcFabricError::InvalidMessage);
        }
        if self
            .required_scopes
            .iter()
            .any(|scope| scope.trim().is_empty())
        {
            return Err(PcFabricError::InvalidMessage);
        }
        if !self
            .required_scopes
            .windows(2)
            .all(|pair| pair[0] < pair[1])
        {
            return Err(PcFabricError::InvalidMessage);
        }
        if self.side_effect == SideEffectClass::Irreversible && self.rollback_supported {
            return Err(PcFabricError::InvalidMessage);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RemoteAction {
    Observe {
        surface: String,
    },
    Execute {
        program: String,
        args: Vec<String>,
        working_dir: Option<String>,
    },
    ArtifactRead {
        path: String,
    },
    ArtifactWrite {
        path: String,
        bytes: Vec<u8>,
    },
}

impl RemoteAction {
    pub fn validate(&self) -> Result<(), PcFabricError> {
        match self {
            Self::Observe { surface } => nonempty(surface),
            Self::Execute {
                program,
                working_dir,
                ..
            } => {
                nonempty(program)?;
                if let Some(working_dir) = working_dir {
                    nonempty(working_dir)?;
                }
                Ok(())
            }
            Self::ArtifactRead { path } | Self::ArtifactWrite { path, .. } => nonempty(path),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteRequest {
    pub request_id: u64,
    pub capability: String,
    pub capability_version: u32,
    pub action: RemoteAction,
    pub request_digest: [u8; 32],
}

impl RemoteRequest {
    pub fn new(
        request_id: u64,
        capability: impl Into<String>,
        capability_version: u32,
        action: RemoteAction,
    ) -> Result<Self, PcFabricError> {
        let capability = capability.into();
        if request_id == 0 || capability_version == 0 {
            return Err(PcFabricError::InvalidMessage);
        }
        nonempty(&capability)?;
        action.validate()?;

        let mut request = Self {
            request_id,
            capability,
            capability_version,
            action,
            request_digest: [0; 32],
        };
        request.request_digest = digest_request(&request)?;
        Ok(request)
    }

    pub fn validate(&self) -> Result<(), PcFabricError> {
        if self.request_id == 0 || self.capability_version == 0 {
            return Err(PcFabricError::InvalidMessage);
        }
        nonempty(&self.capability)?;
        self.action.validate()?;
        if digest_request(self)? != self.request_digest {
            return Err(PcFabricError::InvalidMessage);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RemoteResultStatus {
    Completed,
    Retryable(String),
    Rejected(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteResult {
    pub request_id: u64,
    pub capability: String,
    pub capability_version: u32,
    pub request_digest: [u8; 32],
    pub status: RemoteResultStatus,
    pub output: ActionOutput,
    pub artifacts: Vec<ArtifactDescriptor>,
    pub result_digest: [u8; 32],
}

impl RemoteResult {
    pub fn completed(
        request: &RemoteRequest,
        output: ActionOutput,
        artifacts: Vec<ArtifactDescriptor>,
    ) -> Result<Self, PcFabricError> {
        Self::new(request, RemoteResultStatus::Completed, output, artifacts)
    }

    pub fn retryable(
        request: &RemoteRequest,
        reason: impl Into<String>,
    ) -> Result<Self, PcFabricError> {
        let reason = reason.into();
        nonempty(&reason)?;
        Self::new(
            request,
            RemoteResultStatus::Retryable(reason),
            ActionOutput {
                summary: "remote action retryable".into(),
                value: ActionValue::None,
                evidence: Vec::new(),
            },
            Vec::new(),
        )
    }

    pub fn rejected(
        request: &RemoteRequest,
        reason: impl Into<String>,
    ) -> Result<Self, PcFabricError> {
        let reason = reason.into();
        nonempty(&reason)?;
        Self::new(
            request,
            RemoteResultStatus::Rejected(reason),
            ActionOutput {
                summary: "remote action rejected".into(),
                value: ActionValue::None,
                evidence: Vec::new(),
            },
            Vec::new(),
        )
    }

    fn new(
        request: &RemoteRequest,
        status: RemoteResultStatus,
        output: ActionOutput,
        mut artifacts: Vec<ArtifactDescriptor>,
    ) -> Result<Self, PcFabricError> {
        request.validate()?;
        normalize_artifacts(&mut artifacts)?;
        let mut result = Self {
            request_id: request.request_id,
            capability: request.capability.clone(),
            capability_version: request.capability_version,
            request_digest: request.request_digest,
            status,
            output,
            artifacts,
            result_digest: [0; 32],
        };
        result.validate_fields()?;
        result.result_digest = digest_result(&result)?;
        Ok(result)
    }

    pub fn validate(&self) -> Result<(), PcFabricError> {
        self.validate_fields()?;
        if digest_result(self)? != self.result_digest {
            return Err(PcFabricError::InvalidMessage);
        }
        Ok(())
    }

    pub fn verify_against(&self, request: &RemoteRequest) -> Result<(), PcFabricError> {
        request.validate()?;
        self.validate()?;
        if self.request_id != request.request_id
            || self.capability != request.capability
            || self.capability_version != request.capability_version
            || self.request_digest != request.request_digest
        {
            return Err(PcFabricError::InvalidMessage);
        }
        Ok(())
    }

    fn validate_fields(&self) -> Result<(), PcFabricError> {
        if self.request_id == 0 || self.capability_version == 0 {
            return Err(PcFabricError::InvalidMessage);
        }
        nonempty(&self.capability)?;
        match &self.status {
            RemoteResultStatus::Completed => {}
            RemoteResultStatus::Retryable(reason) | RemoteResultStatus::Rejected(reason) => {
                nonempty(reason)?;
            }
        }
        validate_artifacts(&self.artifacts)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RemoteMessage {
    Capabilities(Vec<RemoteCapability>),
    Request(RemoteRequest),
    Result(RemoteResult),
    ArtifactChunk(ArtifactChunk),
    ArtifactPull { transfer_id: u64, offset: u64 },
    CapabilityQuery,
}

pub fn encode_message(message: &RemoteMessage) -> Result<Vec<u8>, PcFabricError> {
    let (tag, payload) = match message {
        RemoteMessage::Capabilities(capabilities) => {
            validate_capabilities(capabilities)?;
            let mut payload = Vec::new();
            push_u32(&mut payload, len_u32(capabilities.len())?);
            for capability in capabilities {
                encode_capability(&mut payload, capability)?;
            }
            (1u8, payload)
        }
        RemoteMessage::Request(request) => {
            request.validate()?;
            let mut payload = Vec::new();
            encode_request(&mut payload, request)?;
            (2u8, payload)
        }
        RemoteMessage::Result(result) => {
            result.validate()?;
            let mut payload = Vec::new();
            encode_result(&mut payload, result)?;
            (3u8, payload)
        }
        RemoteMessage::ArtifactChunk(chunk) => {
            if chunk.transfer_id == 0 {
                return Err(PcFabricError::InvalidArtifact);
            }
            let mut payload = Vec::new();
            push_u64(&mut payload, chunk.transfer_id);
            push_u64(&mut payload, chunk.offset);
            push_bytes(&mut payload, &chunk.data)?;
            (4u8, payload)
        }
        RemoteMessage::ArtifactPull {
            transfer_id,
            offset,
        } => {
            if *transfer_id == 0 {
                return Err(PcFabricError::InvalidArtifact);
            }
            let mut payload = Vec::new();
            push_u64(&mut payload, *transfer_id);
            push_u64(&mut payload, *offset);
            (5u8, payload)
        }
        RemoteMessage::CapabilityQuery => (6u8, Vec::new()),
    };

    let payload_len = len_u32(payload.len())?;
    let mut out = Vec::with_capacity(
        PCF97_HEADER_LEN
            .checked_add(payload.len())
            .ok_or(PcFabricError::Overflow)?,
    );
    out.extend_from_slice(&PCF97_MAGIC);
    push_u16(&mut out, PCF97_MAJOR);
    push_u16(&mut out, PCF97_MINOR);
    push_u8(&mut out, tag);
    push_u8(&mut out, 0);
    push_u32(&mut out, payload_len);
    out.extend_from_slice(&payload);
    Ok(out)
}

pub fn decode_message(bytes: &[u8]) -> Result<RemoteMessage, PcFabricError> {
    if bytes.len() < PCF97_HEADER_LEN || bytes[..6] != PCF97_MAGIC {
        return Err(PcFabricError::InvalidFrame);
    }
    let major = u16::from_le_bytes([bytes[6], bytes[7]]);
    let minor = u16::from_le_bytes([bytes[8], bytes[9]]);
    if major != PCF97_MAJOR || minor > PCF97_MINOR || bytes[11] != 0 {
        return Err(PcFabricError::InvalidFrame);
    }
    let tag = bytes[10];
    let payload_len = usize::try_from(u32::from_le_bytes(
        bytes[12..16]
            .try_into()
            .map_err(|_| PcFabricError::InvalidFrame)?,
    ))
    .map_err(|_| PcFabricError::Overflow)?;
    let expected = PCF97_HEADER_LEN
        .checked_add(payload_len)
        .ok_or(PcFabricError::Overflow)?;
    if bytes.len() != expected {
        return Err(PcFabricError::InvalidFrame);
    }

    let mut cursor = Cursor::new(&bytes[PCF97_HEADER_LEN..]);
    let message = match tag {
        1 => {
            let count = usize::try_from(cursor.u32()?).map_err(|_| PcFabricError::Overflow)?;
            let mut capabilities = Vec::with_capacity(count);
            for _ in 0..count {
                capabilities.push(decode_capability(&mut cursor)?);
            }
            validate_capabilities(&capabilities)?;
            RemoteMessage::Capabilities(capabilities)
        }
        2 => RemoteMessage::Request(decode_request(&mut cursor)?),
        3 => RemoteMessage::Result(decode_result(&mut cursor)?),
        4 => RemoteMessage::ArtifactChunk(ArtifactChunk {
            transfer_id: cursor.u64()?,
            offset: cursor.u64()?,
            data: cursor.bytes()?.to_vec(),
        }),
        5 => RemoteMessage::ArtifactPull {
            transfer_id: cursor.u64()?,
            offset: cursor.u64()?,
        },
        6 => RemoteMessage::CapabilityQuery,
        _ => return Err(PcFabricError::InvalidMessage),
    };
    if !cursor.is_finished() {
        return Err(PcFabricError::InvalidMessage);
    }
    Ok(message)
}

fn encode_request(out: &mut Vec<u8>, request: &RemoteRequest) -> Result<(), PcFabricError> {
    push_u64(out, request.request_id);
    push_string(out, &request.capability)?;
    push_u32(out, request.capability_version);
    encode_action(out, &request.action)?;
    out.extend_from_slice(&request.request_digest);
    Ok(())
}

fn decode_request(cursor: &mut Cursor<'_>) -> Result<RemoteRequest, PcFabricError> {
    let request_id = cursor.u64()?;
    let capability = cursor.string()?;
    let capability_version = cursor.u32()?;
    let action = decode_action(cursor)?;
    let mut request_digest = [0u8; 32];
    request_digest.copy_from_slice(cursor.take(32)?);
    let request = RemoteRequest {
        request_id,
        capability,
        capability_version,
        action,
        request_digest,
    };
    request.validate()?;
    Ok(request)
}

fn encode_result(out: &mut Vec<u8>, result: &RemoteResult) -> Result<(), PcFabricError> {
    push_u64(out, result.request_id);
    push_string(out, &result.capability)?;
    push_u32(out, result.capability_version);
    out.extend_from_slice(&result.request_digest);
    encode_status(out, &result.status)?;
    encode_output(out, &result.output)?;
    push_u32(out, len_u32(result.artifacts.len())?);
    for artifact in &result.artifacts {
        encode_artifact(out, artifact)?;
    }
    out.extend_from_slice(&result.result_digest);
    Ok(())
}

fn decode_result(cursor: &mut Cursor<'_>) -> Result<RemoteResult, PcFabricError> {
    let request_id = cursor.u64()?;
    let capability = cursor.string()?;
    let capability_version = cursor.u32()?;
    let mut request_digest = [0u8; 32];
    request_digest.copy_from_slice(cursor.take(32)?);
    let status = decode_status(cursor)?;
    let output = decode_output(cursor)?;
    let count = usize::try_from(cursor.u32()?).map_err(|_| PcFabricError::Overflow)?;
    let mut artifacts = Vec::with_capacity(count);
    for _ in 0..count {
        artifacts.push(decode_artifact(cursor)?);
    }
    let mut result_digest = [0u8; 32];
    result_digest.copy_from_slice(cursor.take(32)?);

    let result = RemoteResult {
        request_id,
        capability,
        capability_version,
        request_digest,
        status,
        output,
        artifacts,
        result_digest,
    };
    result.validate()?;
    Ok(result)
}

fn encode_action(out: &mut Vec<u8>, action: &RemoteAction) -> Result<(), PcFabricError> {
    action.validate()?;
    match action {
        RemoteAction::Observe { surface } => {
            push_u8(out, 1);
            push_string(out, surface)?;
        }
        RemoteAction::Execute {
            program,
            args,
            working_dir,
        } => {
            push_u8(out, 2);
            push_string(out, program)?;
            push_strings(out, args)?;
            push_optional_string(out, working_dir.as_deref())?;
        }
        RemoteAction::ArtifactRead { path } => {
            push_u8(out, 3);
            push_string(out, path)?;
        }
        RemoteAction::ArtifactWrite { path, bytes } => {
            push_u8(out, 4);
            push_string(out, path)?;
            push_bytes(out, bytes)?;
        }
    }
    Ok(())
}

fn decode_action(cursor: &mut Cursor<'_>) -> Result<RemoteAction, PcFabricError> {
    let action = match cursor.u8()? {
        1 => RemoteAction::Observe {
            surface: cursor.string()?,
        },
        2 => RemoteAction::Execute {
            program: cursor.string()?,
            args: cursor.strings()?,
            working_dir: cursor.optional_string()?,
        },
        3 => RemoteAction::ArtifactRead {
            path: cursor.string()?,
        },
        4 => RemoteAction::ArtifactWrite {
            path: cursor.string()?,
            bytes: cursor.bytes()?.to_vec(),
        },
        _ => return Err(PcFabricError::InvalidMessage),
    };
    action.validate()?;
    Ok(action)
}

fn encode_status(out: &mut Vec<u8>, status: &RemoteResultStatus) -> Result<(), PcFabricError> {
    match status {
        RemoteResultStatus::Completed => push_u8(out, 1),
        RemoteResultStatus::Retryable(reason) => {
            push_u8(out, 2);
            push_string(out, reason)?;
        }
        RemoteResultStatus::Rejected(reason) => {
            push_u8(out, 3);
            push_string(out, reason)?;
        }
    }
    Ok(())
}

fn decode_status(cursor: &mut Cursor<'_>) -> Result<RemoteResultStatus, PcFabricError> {
    match cursor.u8()? {
        1 => Ok(RemoteResultStatus::Completed),
        2 => Ok(RemoteResultStatus::Retryable(cursor.string()?)),
        3 => Ok(RemoteResultStatus::Rejected(cursor.string()?)),
        _ => Err(PcFabricError::InvalidMessage),
    }
}

fn encode_capability(out: &mut Vec<u8>, value: &RemoteCapability) -> Result<(), PcFabricError> {
    value.validate()?;
    push_string(out, &value.id)?;
    push_u32(out, value.version);
    push_u8(out, encode_side_effect(value.side_effect));
    push_bool(out, value.verification_required);
    push_bool(out, value.rollback_supported);
    push_bool(out, value.resumable);
    push_strings(out, &value.required_scopes)
}

fn decode_capability(cursor: &mut Cursor<'_>) -> Result<RemoteCapability, PcFabricError> {
    let capability = RemoteCapability {
        id: cursor.string()?,
        version: cursor.u32()?,
        side_effect: decode_side_effect(cursor.u8()?)?,
        verification_required: cursor.boolean()?,
        rollback_supported: cursor.boolean()?,
        resumable: cursor.boolean()?,
        required_scopes: cursor.strings()?,
    };
    capability.validate()?;
    Ok(capability)
}

fn encode_output(out: &mut Vec<u8>, output: &ActionOutput) -> Result<(), PcFabricError> {
    push_string(out, &output.summary)?;
    match &output.value {
        ActionValue::None => push_u8(out, 0),
        ActionValue::Text(value) => {
            push_u8(out, 1);
            push_string(out, value)?;
        }
        ActionValue::Bytes(value) => {
            push_u8(out, 2);
            push_bytes(out, value)?;
        }
        ActionValue::TextList(values) => {
            push_u8(out, 3);
            push_strings(out, values)?;
        }
        ActionValue::Fields(fields) => {
            push_u8(out, 4);
            push_u32(out, len_u32(fields.len())?);
            for (key, value) in fields {
                push_string(out, key)?;
                push_string(out, value)?;
            }
        }
    }
    push_strings(out, &output.evidence)
}

fn decode_output(cursor: &mut Cursor<'_>) -> Result<ActionOutput, PcFabricError> {
    let summary = cursor.string()?;
    let value = match cursor.u8()? {
        0 => ActionValue::None,
        1 => ActionValue::Text(cursor.string()?),
        2 => ActionValue::Bytes(cursor.bytes()?.to_vec()),
        3 => ActionValue::TextList(cursor.strings()?),
        4 => {
            let count = usize::try_from(cursor.u32()?).map_err(|_| PcFabricError::Overflow)?;
            let mut fields = BTreeMap::new();
            let mut previous: Option<String> = None;
            for _ in 0..count {
                let key = cursor.string()?;
                if previous.as_ref().is_some_and(|prior| key <= *prior) {
                    return Err(PcFabricError::InvalidMessage);
                }
                previous = Some(key.clone());
                if fields.insert(key, cursor.string()?).is_some() {
                    return Err(PcFabricError::InvalidMessage);
                }
            }
            ActionValue::Fields(fields)
        }
        _ => return Err(PcFabricError::InvalidMessage),
    };
    let evidence = cursor.strings()?;
    Ok(ActionOutput {
        summary,
        value,
        evidence,
    })
}

fn encode_artifact(out: &mut Vec<u8>, artifact: &ArtifactDescriptor) -> Result<(), PcFabricError> {
    artifact.validate()?;
    push_u64(out, artifact.transfer_id);
    push_string(out, &artifact.name)?;
    push_u64(out, artifact.length);
    out.extend_from_slice(&artifact.sha256);
    push_u32(out, artifact.chunk_size);
    Ok(())
}

fn decode_artifact(cursor: &mut Cursor<'_>) -> Result<ArtifactDescriptor, PcFabricError> {
    let transfer_id = cursor.u64()?;
    let name = cursor.string()?;
    let length = cursor.u64()?;
    let mut sha256 = [0u8; 32];
    sha256.copy_from_slice(cursor.take(32)?);
    let artifact = ArtifactDescriptor {
        transfer_id,
        name,
        length,
        sha256,
        chunk_size: cursor.u32()?,
    };
    artifact.validate()?;
    Ok(artifact)
}

fn digest_request(request: &RemoteRequest) -> Result<[u8; 32], PcFabricError> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"NTD97-PCF97-REQUEST");
    push_u64(&mut bytes, request.request_id);
    push_string(&mut bytes, &request.capability)?;
    push_u32(&mut bytes, request.capability_version);
    encode_action(&mut bytes, &request.action)?;
    Ok(Sha256::digest(bytes).into())
}

fn digest_result(result: &RemoteResult) -> Result<[u8; 32], PcFabricError> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"NTD97-PCF97-RESULT");
    push_u64(&mut bytes, result.request_id);
    push_string(&mut bytes, &result.capability)?;
    push_u32(&mut bytes, result.capability_version);
    bytes.extend_from_slice(&result.request_digest);
    encode_status(&mut bytes, &result.status)?;
    encode_output(&mut bytes, &result.output)?;
    push_u32(&mut bytes, len_u32(result.artifacts.len())?);
    for artifact in &result.artifacts {
        encode_artifact(&mut bytes, artifact)?;
    }
    Ok(Sha256::digest(bytes).into())
}

fn normalize_artifacts(artifacts: &mut [ArtifactDescriptor]) -> Result<(), PcFabricError> {
    artifacts.sort_by_key(|artifact| artifact.transfer_id);
    validate_artifacts(artifacts)
}

fn validate_artifacts(artifacts: &[ArtifactDescriptor]) -> Result<(), PcFabricError> {
    let mut previous = 0u64;
    for artifact in artifacts {
        artifact.validate()?;
        if artifact.transfer_id <= previous {
            return Err(PcFabricError::InvalidArtifact);
        }
        previous = artifact.transfer_id;
    }
    Ok(())
}

fn validate_capabilities(capabilities: &[RemoteCapability]) -> Result<(), PcFabricError> {
    let mut previous: Option<&str> = None;
    for capability in capabilities {
        capability.validate()?;
        if previous.is_some_and(|prior| capability.id.as_str() <= prior) {
            return Err(PcFabricError::InvalidMessage);
        }
        previous = Some(&capability.id);
    }
    Ok(())
}

fn encode_side_effect(value: SideEffectClass) -> u8 {
    match value {
        SideEffectClass::ReadOnly => 1,
        SideEffectClass::Reversible => 2,
        SideEffectClass::ExternalWrite => 3,
        SideEffectClass::Irreversible => 4,
    }
}

fn decode_side_effect(value: u8) -> Result<SideEffectClass, PcFabricError> {
    match value {
        1 => Ok(SideEffectClass::ReadOnly),
        2 => Ok(SideEffectClass::Reversible),
        3 => Ok(SideEffectClass::ExternalWrite),
        4 => Ok(SideEffectClass::Irreversible),
        _ => Err(PcFabricError::InvalidMessage),
    }
}

fn nonempty(value: &str) -> Result<(), PcFabricError> {
    if value.trim().is_empty() {
        Err(PcFabricError::InvalidMessage)
    } else {
        Ok(())
    }
}

fn len_u32(value: usize) -> Result<u32, PcFabricError> {
    u32::try_from(value).map_err(|_| PcFabricError::Overflow)
}

fn push_u8(out: &mut Vec<u8>, value: u8) {
    out.push(value);
}

fn push_bool(out: &mut Vec<u8>, value: bool) {
    push_u8(out, u8::from(value));
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

fn push_bytes(out: &mut Vec<u8>, value: &[u8]) -> Result<(), PcFabricError> {
    push_u32(out, len_u32(value.len())?);
    out.extend_from_slice(value);
    Ok(())
}

fn push_string(out: &mut Vec<u8>, value: &str) -> Result<(), PcFabricError> {
    push_bytes(out, value.as_bytes())
}

fn push_strings(out: &mut Vec<u8>, values: &[String]) -> Result<(), PcFabricError> {
    push_u32(out, len_u32(values.len())?);
    for value in values {
        push_string(out, value)?;
    }
    Ok(())
}

fn push_optional_string(out: &mut Vec<u8>, value: Option<&str>) -> Result<(), PcFabricError> {
    match value {
        None => push_u8(out, 0),
        Some(value) => {
            push_u8(out, 1);
            push_string(out, value)?;
        }
    }
    Ok(())
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], PcFabricError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(PcFabricError::Overflow)?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or(PcFabricError::InvalidMessage)?;
        self.offset = end;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8, PcFabricError> {
        Ok(self.take(1)?[0])
    }

    fn boolean(&mut self) -> Result<bool, PcFabricError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(PcFabricError::InvalidMessage),
        }
    }

    fn u32(&mut self) -> Result<u32, PcFabricError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| PcFabricError::InvalidMessage)?,
        ))
    }

    fn u64(&mut self) -> Result<u64, PcFabricError> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| PcFabricError::InvalidMessage)?,
        ))
    }

    fn bytes(&mut self) -> Result<&'a [u8], PcFabricError> {
        let len = usize::try_from(self.u32()?).map_err(|_| PcFabricError::Overflow)?;
        self.take(len)
    }

    fn string(&mut self) -> Result<String, PcFabricError> {
        String::from_utf8(self.bytes()?.to_vec()).map_err(|_| PcFabricError::InvalidMessage)
    }

    fn strings(&mut self) -> Result<Vec<String>, PcFabricError> {
        let count = usize::try_from(self.u32()?).map_err(|_| PcFabricError::Overflow)?;
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(self.string()?);
        }
        Ok(values)
    }

    fn optional_string(&mut self) -> Result<Option<String>, PcFabricError> {
        match self.u8()? {
            0 => Ok(None),
            1 => Ok(Some(self.string()?)),
            _ => Err(PcFabricError::InvalidMessage),
        }
    }

    fn is_finished(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_and_result_round_trip_canonically() {
        let request = RemoteRequest::new(
            7,
            "pc.execute",
            1,
            RemoteAction::Execute {
                program: "cargo".into(),
                args: vec!["test".into(), "--workspace".into()],
                working_dir: Some("/workspace".into()),
            },
        )
        .expect("request");
        let message = RemoteMessage::Request(request.clone());
        let first = encode_message(&message).expect("encode");
        let second = encode_message(&message).expect("encode");
        assert_eq!(first, second);
        assert_eq!(decode_message(&first).expect("decode"), message);

        let result = RemoteResult::completed(
            &request,
            ActionOutput {
                summary: "build passed".into(),
                value: ActionValue::Text("ok".into()),
                evidence: vec!["exit=0".into()],
            },
            Vec::new(),
        )
        .expect("result");
        result.verify_against(&request).expect("verify");
        let encoded = encode_message(&RemoteMessage::Result(result.clone())).expect("encode");
        assert_eq!(
            decode_message(&encoded).expect("decode"),
            RemoteMessage::Result(result)
        );
    }

    #[test]
    fn capability_snapshot_requires_canonical_order() {
        let a = RemoteCapability {
            id: "pc.observe".into(),
            version: 1,
            side_effect: SideEffectClass::ReadOnly,
            verification_required: true,
            rollback_supported: false,
            resumable: false,
            required_scopes: vec!["pc.observe".into()],
        };
        let b = RemoteCapability {
            id: "pc.execute".into(),
            ..a.clone()
        };
        assert!(encode_message(&RemoteMessage::Capabilities(vec![a, b])).is_err());
    }
}
