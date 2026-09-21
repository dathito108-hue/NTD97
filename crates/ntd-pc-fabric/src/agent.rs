#![forbid(unsafe_code)]

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use ntd_core::SideEffectClass;
use ntd_runtime::{ActionOutput, ActionValue};

use crate::{
    decode_message, encode_message, ArtifactSender, PcFabricError, RemoteAction, RemoteCapability,
    RemoteMessage, RemoteRequest, RemoteResult, SecureSession, DEFAULT_ARTIFACT_CHUNK,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DesktopHandlerOutput {
    pub output: ActionOutput,
    pub artifacts: Vec<(String, Vec<u8>)>,
}

impl DesktopHandlerOutput {
    pub fn new(output: ActionOutput) -> Self {
        Self {
            output,
            artifacts: Vec::new(),
        }
    }

    pub fn with_artifact(mut self, name: impl Into<String>, bytes: Vec<u8>) -> Self {
        self.artifacts.push((name.into(), bytes));
        self
    }
}

pub trait DesktopCapabilityHandler {
    fn capability(&self) -> RemoteCapability;

    fn execute(&mut self, action: &RemoteAction) -> Result<DesktopHandlerOutput, PcFabricError>;
}

pub struct DesktopAgent {
    session: SecureSession,
    handlers: BTreeMap<String, Box<dyn DesktopCapabilityHandler>>,
    outgoing_artifacts: BTreeMap<u64, ArtifactSender>,
    completed_requests: BTreeMap<u64, RemoteResult>,
    next_transfer_id: u64,
}

impl DesktopAgent {
    pub fn new(session: SecureSession) -> Self {
        Self {
            session,
            handlers: BTreeMap::new(),
            outgoing_artifacts: BTreeMap::new(),
            completed_requests: BTreeMap::new(),
            next_transfer_id: 1,
        }
    }

    pub fn register_handler<H>(&mut self, handler: H) -> Result<(), PcFabricError>
    where
        H: DesktopCapabilityHandler + 'static,
    {
        let capability = handler.capability();
        capability.validate()?;
        if self.handlers.contains_key(&capability.id) {
            return Err(PcFabricError::InvalidMessage);
        }
        self.handlers.insert(capability.id, Box::new(handler));
        Ok(())
    }

    pub fn capabilities(&self) -> Vec<RemoteCapability> {
        self.handlers
            .values()
            .map(|handler| handler.capability())
            .collect()
    }

    pub fn handle_encrypted_frame(
        &mut self,
        encrypted_frame: &[u8],
    ) -> Result<Vec<u8>, PcFabricError> {
        let plaintext = self.session.open(encrypted_frame)?;
        let message = decode_message(&plaintext)?;

        let response = match message {
            RemoteMessage::CapabilityQuery => RemoteMessage::Capabilities(self.capabilities()),
            RemoteMessage::Request(request) => {
                RemoteMessage::Result(self.execute_request(&request)?)
            }
            RemoteMessage::ArtifactPull {
                transfer_id,
                offset,
            } => {
                let sender = self
                    .outgoing_artifacts
                    .get(&transfer_id)
                    .ok_or(PcFabricError::InvalidArtifact)?;
                RemoteMessage::ArtifactChunk(sender.chunk_at(offset)?)
            }
            RemoteMessage::Capabilities(_)
            | RemoteMessage::Result(_)
            | RemoteMessage::ArtifactChunk(_) => return Err(PcFabricError::InvalidMessage),
        };

        let encoded = encode_message(&response)?;
        self.session.seal(&encoded)
    }

    fn execute_request(&mut self, request: &RemoteRequest) -> Result<RemoteResult, PcFabricError> {
        request.validate()?;

        if let Some(previous) = self.completed_requests.get(&request.request_id) {
            if previous.request_digest != request.request_digest {
                return Err(PcFabricError::InvalidMessage);
            }
            return Ok(previous.clone());
        }

        let execution = {
            let Some(handler) = self.handlers.get_mut(&request.capability) else {
                return RemoteResult::rejected(request, "remote capability is not installed");
            };
            let capability = handler.capability();
            if capability.version != request.capability_version {
                return Err(PcFabricError::CapabilityVersionMismatch {
                    capability: request.capability.clone(),
                    expected: capability.version,
                    actual: request.capability_version,
                });
            }
            handler.execute(&request.action)
        };

        let result = match execution {
            Ok(output) => {
                let descriptors = self.store_artifacts(output.artifacts)?;
                RemoteResult::completed(request, output.output, descriptors)?
            }
            Err(PcFabricError::PolicyDenied(reason)) => RemoteResult::rejected(request, reason)?,
            Err(PcFabricError::Io(reason)) => return RemoteResult::retryable(request, reason),
            Err(error) => RemoteResult::rejected(request, format!("{error:?}"))?,
        };

        self.completed_requests
            .insert(request.request_id, result.clone());
        Ok(result)
    }

    fn store_artifacts(
        &mut self,
        artifacts: Vec<(String, Vec<u8>)>,
    ) -> Result<Vec<crate::ArtifactDescriptor>, PcFabricError> {
        let mut descriptors = Vec::with_capacity(artifacts.len());
        for (name, bytes) in artifacts {
            let transfer_id = self.next_transfer_id;
            self.next_transfer_id = self
                .next_transfer_id
                .checked_add(1)
                .ok_or(PcFabricError::Overflow)?;
            let sender = ArtifactSender::new(transfer_id, name, bytes, DEFAULT_ARTIFACT_CHUNK)?;
            descriptors.push(sender.descriptor().clone());
            self.outgoing_artifacts.insert(transfer_id, sender);
        }
        Ok(descriptors)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DesktopExecutionPolicy {
    allowed_programs: BTreeSet<String>,
    allowed_working_roots: Vec<PathBuf>,
    max_output_bytes: usize,
}

impl DesktopExecutionPolicy {
    pub fn new(
        allowed_programs: impl IntoIterator<Item = String>,
        allowed_working_roots: Vec<PathBuf>,
        max_output_bytes: usize,
    ) -> Result<Self, PcFabricError> {
        let allowed_programs = allowed_programs
            .into_iter()
            .map(|program| program.trim().to_owned())
            .collect::<BTreeSet<_>>();
        if allowed_programs.is_empty()
            || allowed_programs.iter().any(|program| program.is_empty())
            || max_output_bytes == 0
        {
            return Err(PcFabricError::PolicyDenied(
                "invalid execution policy".into(),
            ));
        }
        let allowed_working_roots = canonical_roots(allowed_working_roots)?;
        Ok(Self {
            allowed_programs,
            allowed_working_roots,
            max_output_bytes,
        })
    }

    fn working_directory(&self, requested: Option<&str>) -> Result<PathBuf, PcFabricError> {
        match requested {
            Some(path) => resolve_existing_path(path, &self.allowed_working_roots, true),
            None => self
                .allowed_working_roots
                .first()
                .cloned()
                .ok_or_else(|| PcFabricError::PolicyDenied("no working root".into())),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProcessExecutionHandler {
    capability: RemoteCapability,
    policy: DesktopExecutionPolicy,
}

impl ProcessExecutionHandler {
    pub fn new(policy: DesktopExecutionPolicy) -> Self {
        Self {
            capability: RemoteCapability {
                id: "pc.process.execute".into(),
                version: 1,
                side_effect: SideEffectClass::ExternalWrite,
                verification_required: true,
                rollback_supported: false,
                resumable: false,
                required_scopes: vec!["pc.execute".into()],
            },
            policy,
        }
    }
}

impl DesktopCapabilityHandler for ProcessExecutionHandler {
    fn capability(&self) -> RemoteCapability {
        self.capability.clone()
    }

    fn execute(&mut self, action: &RemoteAction) -> Result<DesktopHandlerOutput, PcFabricError> {
        let RemoteAction::Execute {
            program,
            args,
            working_dir,
        } = action
        else {
            return Err(PcFabricError::InvalidMessage);
        };

        if !self.policy.allowed_programs.contains(program) {
            return Err(PcFabricError::PolicyDenied(format!(
                "program is not allowed: {program}"
            )));
        }
        let working_dir = self.policy.working_directory(working_dir.as_deref())?;

        let output = Command::new(program)
            .args(args)
            .current_dir(&working_dir)
            .output()?;

        let mut fields = BTreeMap::new();
        fields.insert(
            "exit_code".into(),
            output
                .status
                .code()
                .map(|code| code.to_string())
                .unwrap_or_else(|| "signal".into()),
        );
        fields.insert(
            "stdout".into(),
            bounded_text(&output.stdout, self.policy.max_output_bytes),
        );
        fields.insert(
            "stderr".into(),
            bounded_text(&output.stderr, self.policy.max_output_bytes),
        );

        Ok(DesktopHandlerOutput::new(ActionOutput {
            summary: if output.status.success() {
                "remote process completed".into()
            } else {
                "remote process failed".into()
            },
            value: ActionValue::Fields(fields),
            evidence: vec![format!("success={}", output.status.success())],
        }))
    }
}

#[derive(Debug, Clone, Default)]
pub struct SystemObserveHandler;

impl DesktopCapabilityHandler for SystemObserveHandler {
    fn capability(&self) -> RemoteCapability {
        RemoteCapability {
            id: "pc.system.observe".into(),
            version: 1,
            side_effect: SideEffectClass::ReadOnly,
            verification_required: true,
            rollback_supported: false,
            resumable: false,
            required_scopes: vec!["pc.observe".into()],
        }
    }

    fn execute(&mut self, action: &RemoteAction) -> Result<DesktopHandlerOutput, PcFabricError> {
        let RemoteAction::Observe { surface } = action else {
            return Err(PcFabricError::InvalidMessage);
        };

        let mut fields = BTreeMap::new();
        fields.insert("surface".into(), surface.clone());
        fields.insert("os".into(), std::env::consts::OS.into());
        fields.insert("arch".into(), std::env::consts::ARCH.into());
        fields.insert("family".into(), std::env::consts::FAMILY.into());
        fields.insert(
            "cwd".into(),
            std::env::current_dir()?.to_string_lossy().into_owned(),
        );

        Ok(DesktopHandlerOutput::new(ActionOutput {
            summary: "remote desktop observation".into(),
            value: ActionValue::Fields(fields),
            evidence: vec!["source=paired-pc-agent".into()],
        }))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DesktopArtifactPolicy {
    allowed_roots: Vec<PathBuf>,
    allow_write: bool,
    max_bytes: usize,
}

impl DesktopArtifactPolicy {
    pub fn new(
        allowed_roots: Vec<PathBuf>,
        allow_write: bool,
        max_bytes: usize,
    ) -> Result<Self, PcFabricError> {
        if max_bytes == 0 {
            return Err(PcFabricError::PolicyDenied(
                "artifact byte limit must be positive".into(),
            ));
        }
        Ok(Self {
            allowed_roots: canonical_roots(allowed_roots)?,
            allow_write,
            max_bytes,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArtifactMode {
    Read,
    Write,
}

#[derive(Debug, Clone)]
pub struct FileArtifactHandler {
    capability: RemoteCapability,
    policy: DesktopArtifactPolicy,
    mode: ArtifactMode,
}

impl FileArtifactHandler {
    pub fn read(policy: DesktopArtifactPolicy) -> Self {
        Self {
            capability: RemoteCapability {
                id: "pc.artifact.read".into(),
                version: 1,
                side_effect: SideEffectClass::ReadOnly,
                verification_required: true,
                rollback_supported: false,
                resumable: false,
                required_scopes: vec!["pc.artifact.read".into()],
            },
            policy,
            mode: ArtifactMode::Read,
        }
    }

    pub fn write(policy: DesktopArtifactPolicy) -> Self {
        Self {
            capability: RemoteCapability {
                id: "pc.artifact.write".into(),
                version: 1,
                side_effect: SideEffectClass::ExternalWrite,
                verification_required: true,
                rollback_supported: false,
                resumable: false,
                required_scopes: vec!["pc.artifact.write".into()],
            },
            policy,
            mode: ArtifactMode::Write,
        }
    }
}

impl DesktopCapabilityHandler for FileArtifactHandler {
    fn capability(&self) -> RemoteCapability {
        self.capability.clone()
    }

    fn execute(&mut self, action: &RemoteAction) -> Result<DesktopHandlerOutput, PcFabricError> {
        match (self.mode, action) {
            (ArtifactMode::Read, RemoteAction::ArtifactRead { path }) => {
                let resolved = resolve_existing_path(path, &self.policy.allowed_roots, false)?;
                let metadata = fs::metadata(&resolved)?;
                let length =
                    usize::try_from(metadata.len()).map_err(|_| PcFabricError::Overflow)?;
                if length > self.policy.max_bytes {
                    return Err(PcFabricError::PolicyDenied(
                        "artifact exceeds byte limit".into(),
                    ));
                }
                let bytes = fs::read(&resolved)?;
                let name = resolved
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("artifact.bin")
                    .to_owned();

                let mut fields = BTreeMap::new();
                fields.insert("path".into(), resolved.to_string_lossy().into_owned());
                fields.insert("bytes".into(), bytes.len().to_string());

                Ok(DesktopHandlerOutput::new(ActionOutput {
                    summary: "remote artifact ready".into(),
                    value: ActionValue::Fields(fields),
                    evidence: vec!["artifact-hash-verified-on-receive".into()],
                })
                .with_artifact(name, bytes))
            }
            (ArtifactMode::Write, RemoteAction::ArtifactWrite { path, bytes }) => {
                if !self.policy.allow_write {
                    return Err(PcFabricError::PolicyDenied(
                        "artifact writes are disabled".into(),
                    ));
                }
                if bytes.len() > self.policy.max_bytes {
                    return Err(PcFabricError::PolicyDenied(
                        "artifact exceeds byte limit".into(),
                    ));
                }
                let resolved = resolve_write_path(path, &self.policy.allowed_roots)?;
                fs::write(&resolved, bytes)?;

                let mut fields = BTreeMap::new();
                fields.insert("path".into(), resolved.to_string_lossy().into_owned());
                fields.insert("bytes".into(), bytes.len().to_string());

                Ok(DesktopHandlerOutput::new(ActionOutput {
                    summary: "remote artifact written".into(),
                    value: ActionValue::Fields(fields),
                    evidence: vec!["write-completed".into()],
                }))
            }
            _ => Err(PcFabricError::InvalidMessage),
        }
    }
}

fn canonical_roots(roots: Vec<PathBuf>) -> Result<Vec<PathBuf>, PcFabricError> {
    if roots.is_empty() {
        return Err(PcFabricError::PolicyDenied(
            "at least one allowed root is required".into(),
        ));
    }
    let mut canonical = Vec::with_capacity(roots.len());
    for root in roots {
        let root = fs::canonicalize(root)?;
        if !root.is_dir() {
            return Err(PcFabricError::PolicyDenied(
                "allowed root is not a directory".into(),
            ));
        }
        canonical.push(root);
    }
    canonical.sort();
    canonical.dedup();
    Ok(canonical)
}

fn resolve_existing_path(
    requested: &str,
    roots: &[PathBuf],
    require_directory: bool,
) -> Result<PathBuf, PcFabricError> {
    let candidate = candidate_path(requested, roots)?;
    let canonical = fs::canonicalize(candidate)?;
    if !roots.iter().any(|root| canonical.starts_with(root)) {
        return Err(PcFabricError::PolicyDenied(
            "path is outside allowed roots".into(),
        ));
    }
    if require_directory && !canonical.is_dir() {
        return Err(PcFabricError::PolicyDenied(
            "working directory is not a directory".into(),
        ));
    }
    Ok(canonical)
}

fn resolve_write_path(requested: &str, roots: &[PathBuf]) -> Result<PathBuf, PcFabricError> {
    let candidate = candidate_path(requested, roots)?;
    let file_name = candidate
        .file_name()
        .ok_or_else(|| PcFabricError::PolicyDenied("missing file name".into()))?
        .to_owned();
    let parent = candidate
        .parent()
        .ok_or_else(|| PcFabricError::PolicyDenied("missing parent directory".into()))?;
    let parent = fs::canonicalize(parent)?;
    if !roots.iter().any(|root| parent.starts_with(root)) {
        return Err(PcFabricError::PolicyDenied(
            "path is outside allowed roots".into(),
        ));
    }
    Ok(parent.join(file_name))
}

fn candidate_path(requested: &str, roots: &[PathBuf]) -> Result<PathBuf, PcFabricError> {
    let requested = requested.trim();
    if requested.is_empty() {
        return Err(PcFabricError::PolicyDenied("empty path".into()));
    }
    let path = Path::new(requested);
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        roots
            .first()
            .map(|root| root.join(path))
            .ok_or_else(|| PcFabricError::PolicyDenied("no allowed root".into()))
    }
}

fn bounded_text(bytes: &[u8], max_bytes: usize) -> String {
    let bytes = &bytes[..bytes.len().min(max_bytes)];
    String::from_utf8_lossy(bytes).into_owned()
}

#[cfg(test)]
mod tests {
    use crate::{ClientHandshake, HandshakeEntropy, PairedIdentity, ServerHello};

    use super::*;

    fn sessions() -> (SecureSession, SecureSession) {
        let phone = PairedIdentity::from_seed([21; 32]);
        let pc = PairedIdentity::from_seed([22; 32]);
        let (client_state, client_hello) = ClientHandshake::begin(
            phone.clone(),
            pc.pairing_record(),
            HandshakeEntropy::from_seed([23; 32]),
        )
        .expect("begin");
        let (server_hello, server) = ServerHello::accept(
            pc,
            phone.pairing_record(),
            HandshakeEntropy::from_seed([24; 32]),
            &client_hello,
        )
        .expect("accept");
        let client = client_state.finish(&server_hello).expect("finish");
        (client, server)
    }

    #[test]
    fn agent_serves_capability_snapshot_over_encrypted_session() {
        let (mut client, server) = sessions();
        let mut agent = DesktopAgent::new(server);
        agent
            .register_handler(SystemObserveHandler)
            .expect("register");

        let request = encode_message(&RemoteMessage::CapabilityQuery).expect("encode");
        let frame = client.seal(&request).expect("seal");
        let reply = agent.handle_encrypted_frame(&frame).expect("handle");
        let plaintext = client.open(&reply).expect("open");
        let RemoteMessage::Capabilities(capabilities) = decode_message(&plaintext).expect("decode")
        else {
            panic!("expected capabilities");
        };

        assert_eq!(capabilities.len(), 1);
        assert_eq!(capabilities[0].id, "pc.system.observe");
    }

    #[test]
    fn desktop_policy_rejects_unallowlisted_program_and_path_escape() {
        let root = std::env::current_dir().expect("cwd");
        let policy = DesktopExecutionPolicy::new(
            vec!["ntd97-approved-tool".into()],
            vec![root.clone()],
            1024,
        )
        .expect("policy");
        let mut handler = ProcessExecutionHandler::new(policy);

        let denied = handler.execute(&RemoteAction::Execute {
            program: "ntd97-not-allowed".into(),
            args: Vec::new(),
            working_dir: None,
        });
        assert!(matches!(denied, Err(PcFabricError::PolicyDenied(_))));

        let roots = canonical_roots(vec![root]).expect("roots");
        let escaped = resolve_existing_path("..", &roots, true);
        assert!(matches!(escaped, Err(PcFabricError::PolicyDenied(_))));
    }
}
