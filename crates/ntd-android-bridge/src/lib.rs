#![deny(unsafe_code)]
#![allow(non_snake_case)]

use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    net::SocketAddr,
    path::{Component, Path, PathBuf},
    ptr::null_mut,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex, OnceLock,
    },
    time::Duration,
};

use jni::{
    objects::{JByteArray, JClass, JObject, JString, JValue},
    sys::{jboolean, jbyteArray, jint, jlong},
    JNIEnv, JavaVM,
};
use ntd_assimilation::{
    activate_thin_generative_capsule, verify_native_package_with_shards, AssetKind,
    FileBackedTensorResolver, FileTensorShardStore, NativePackage, ThinGenerativeActivation,
};
use ntd_capsule::{sha256, NativeTokenizerModel};
use ntd_mobile_shell::{
    decode_mobile_continuity_bundle, encode_mobile_continuity_bundle,
    restore_mobile_continuity_bundle, AvatarController, AvatarSurface, LipSyncState,
    MobileContinuityBundle, MobileContinuityState, WakeReason,
};
use ntd_pc_fabric::{
    connect_paired_tcp, HandshakeEntropy, PairedIdentity, PairedPcAdapter, PairingRecord,
    RemoteCapability, TcpFrameTransport,
};
use ntd_runtime::{
    choose_reasoning_budget, continue_verified_assistant_plan, decode_action_fabric_checkpoint,
    decode_conversation_checkpoint, encode_action_fabric_checkpoint,
    encode_conversation_checkpoint, memory_recall_limit_for_budget, model_inference_signals,
    parse_native_action_plan, run_budgeted_reasoning_cycle, sample_token, ActionFabric,
    ActionOutput, ActionPlanStatus, ActionStatus, ActionValue, ActionVerification, ActionVerifier,
    AdapterResult, AssistantActionPlan, AssistantActionRunError, AssistantPlanDecision,
    AuthorityGrant, AuthorityScope, CapabilityAdapter, CapabilityDescriptor, CapabilityDomain,
    CapabilityId, CapabilityRegistry, CognitiveContext, CognitiveIdentity, CognitiveObservation,
    CpuReferenceProvider, DistributionKind, GenerationConfig, GenerationControl, GraphGenerator,
    LlamaSpmConfig, LlamaSpmTokenizer, NativeChatPromptCompiler, NativeReasoningProbe,
    ResourceSnapshot, SamplingMode, SideEffectClass, SovereignConversationState, TaskStatus,
    ThermalState, TypedAction, NATIVE_ACTION_DIRECT, NATIVE_ACTION_PROTOCOL_V1,
};
use ntd_validation::{
    encode_physical_evidence, run_logical_continuity_soak, run_native_validation_workload,
    DeviceEvidence, EvidenceClass, PhysicalEvidenceRecord,
};

const BRIDGE_PROTOCOL_VERSION: u8 = 1;
const CHAT_EVENT_PROTOCOL_VERSION: u8 = 1;
const CHAT_EVENT_TOKEN: u8 = 1;
const CHAT_EVENT_COMPLETE: u8 = 2;
const CHAT_EVENT_CANCELLED: u8 = 3;
const CHAT_EVENT_ERROR: u8 = 4;
const CHAT_EVENT_APPROVAL_REQUIRED: u8 = 5;
const CHAT_EVENT_ACTION_CHECKPOINTED: u8 = 6;
const CHAT_STATUS_MISSING: i32 = 0;
const CHAT_STATUS_RUNNING: i32 = 1;
const CHAT_STATUS_COMPLETE: i32 = 2;
const CHAT_STATUS_CANCELLED: i32 = 3;
const CHAT_STATUS_FAILED: i32 = 4;
const CHAT_STATUS_WAITING_APPROVAL: i32 = 5;
const ACTION_PLANNER_MAX_NEW_TOKENS: usize = 20;
const MAX_PLATFORM_TEXT_BYTES: usize = 512 * 1024;
const PLATFORM_WEB_PROTOCOL_VERSION: u8 = 1;
const PLATFORM_BROWSER_PROTOCOL_VERSION: u8 = 1;
const PLATFORM_DEVICE_APP_PROTOCOL_VERSION: u8 = 1;
const PLATFORM_STORAGE_GRANT_PROTOCOL_VERSION: u8 = 1;

static JAVA_VM: OnceLock<JavaVM> = OnceLock::new();

struct NativeChatModel {
    asset_id: String,
    version: u32,
    activation: ThinGenerativeActivation,
    resolver: FileBackedTensorResolver,
    tokenizer: LlamaSpmTokenizer,
    context_limit: usize,
    capability_root: PathBuf,
}

#[derive(Clone)]
struct PcPairProfile {
    peer: String,
    address: SocketAddr,
    local_seed: [u8; 32],
    remote_peer_id: [u8; 16],
    remote_verify_key: [u8; 32],
}

fn valid_pc_peer_alias(peer: &str) -> bool {
    !peer.is_empty()
        && peer.len() <= 64
        && peer
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
}

fn decode_fixed_hex<const N: usize>(value: &str) -> Result<[u8; N], String> {
    if value.len() != N * 2 {
        return Err("paired-PC hex field has invalid length".into());
    }
    let mut out = [0u8; N];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let high = (pair[0] as char)
            .to_digit(16)
            .ok_or_else(|| "paired-PC hex field is invalid".to_owned())?;
        let low = (pair[1] as char)
            .to_digit(16)
            .ok_or_else(|| "paired-PC hex field is invalid".to_owned())?;
        out[index] = u8::try_from((high << 4) | low)
            .map_err(|_| "paired-PC hex field overflow".to_owned())?;
    }
    Ok(out)
}

fn profile_value<'a>(line: &'a str, key: &str) -> Result<&'a str, String> {
    line.strip_prefix(key)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("paired-PC profile missing {key}"))
}

fn fixed_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[usize::from(byte >> 4)] as char);
        out.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    out
}

fn os_random<const N: usize>() -> Result<[u8; N], String> {
    let mut bytes = [0u8; N];
    fs::File::open("/dev/urandom")
        .and_then(|mut file| file.read_exact(&mut bytes))
        .map_err(|error| format!("read OS random bytes: {error}"))?;
    Ok(bytes)
}

fn validate_pc_socket_address(address: SocketAddr) -> Result<SocketAddr, String> {
    if address.port() == 0 || address.ip().is_unspecified() || address.ip().is_multicast() {
        return Err("paired-PC socket address is not routable".into());
    }
    Ok(address)
}

fn canonical_pc_pair_directory(root: &Path, create: bool) -> Result<PathBuf, String> {
    if create {
        fs::create_dir_all(root)
            .map_err(|error| format!("create paired-PC capability root: {error}"))?;
    }
    let root = fs::canonicalize(root)
        .map_err(|error| format!("canonicalize paired-PC capability root: {error}"))?;
    let pairs = root.join("pc-pairs");
    if create {
        match fs::symlink_metadata(&pairs) {
            Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
                return Err("paired-PC profile directory is not a real directory".into());
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                fs::create_dir(&pairs)
                    .map_err(|error| format!("create paired-PC profile directory: {error}"))?;
            }
            Err(error) => {
                return Err(format!("inspect paired-PC profile directory: {error}"));
            }
        }
    }
    let pairs = fs::canonicalize(&pairs)
        .map_err(|error| format!("paired-PC profile directory is unavailable: {error}"))?;
    if !pairs.starts_with(&root) {
        return Err("paired-PC profile directory escaped capability root".into());
    }
    Ok(pairs)
}

fn pc_pair_public_receipt(profile: &PcPairProfile) -> String {
    let local = PairedIdentity::from_seed(profile.local_seed);
    format!(
        "NTD97_PC_PAIR_RECEIPT_V1\npeer={}\naddress={}\nlocal_peer_id={}\nlocal_verify_key={}\nEND\n",
        profile.peer,
        profile.address,
        fixed_hex(&local.peer_id()),
        fixed_hex(&local.verify_key()),
    )
}

fn describe_pc_pair_profile(root: &Path, peer: &str) -> Result<String, String> {
    let profile = load_pc_pair_profile(root, peer)?;
    Ok(pc_pair_public_receipt(&profile))
}

fn sync_pc_pair_directory(pairs: &Path) -> Result<(), String> {
    fs::File::open(pairs)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| format!("sync paired-PC profile directory: {error}"))
}

fn rollback_pc_pair_commit(pairs: &Path, target: &Path) -> Result<(), String> {
    let mut errors = Vec::new();
    match fs::remove_file(target) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => errors.push(format!("remove {}: {error}", target.display())),
    }
    if let Err(error) = sync_pc_pair_directory(pairs) {
        errors.push(error);
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}

fn provision_pc_pair_profile(
    root: &Path,
    peer: &str,
    address: &str,
    remote_peer_id_hex: &str,
    remote_verify_key_hex: &str,
) -> Result<String, String> {
    if !valid_pc_peer_alias(peer) {
        return Err("paired-PC peer alias is invalid".into());
    }
    let address = validate_pc_socket_address(
        address
            .parse::<SocketAddr>()
            .map_err(|_| "paired-PC profile address is invalid".to_owned())?,
    )?;
    let remote_peer_id = decode_fixed_hex::<16>(remote_peer_id_hex)?;
    let remote_verify_key = decode_fixed_hex::<32>(remote_verify_key_hex)?;
    PairingRecord::from_public(remote_peer_id, remote_verify_key)
        .map_err(|error| format!("paired-PC pinned identity is invalid: {error:?}"))?;

    let pairs = canonical_pc_pair_directory(root, true)?;
    let target = pairs.join(format!("{peer}.pcp97"));
    match fs::symlink_metadata(&target) {
        Ok(_) => {
            return Err("paired-PC profile already exists; revoke it before replacement".into());
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(format!("inspect paired-PC target profile: {error}")),
    }

    let local_seed = os_random::<32>()?;
    let body = format!(
        "NTD97_PC_PAIR_V1\npeer={peer}\naddress={address}\nlocal_seed={}\nremote_peer_id={}\nremote_verify_key={}\nEND\n",
        fixed_hex(&local_seed),
        fixed_hex(&remote_peer_id),
        fixed_hex(&remote_verify_key),
    );

    let write_result = (|| -> Result<(), String> {
        let mut file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&target)
            .map_err(|error| format!("create paired-PC profile without replacement: {error}"))?;
        file.write_all(body.as_bytes())
            .map_err(|error| format!("write paired-PC profile: {error}"))?;
        file.sync_all()
            .map_err(|error| format!("sync paired-PC profile: {error}"))?;
        drop(file);
        sync_pc_pair_directory(&pairs)?;
        Ok(())
    })();
    if let Err(error) = write_result {
        let rollback = rollback_pc_pair_commit(&pairs, &target);
        return Err(match rollback {
            Ok(()) => error,
            Err(rollback_error) => format!("{error}; rollback failed: {rollback_error}"),
        });
    }

    let loaded = match load_pc_pair_profile(root, peer) {
        Ok(profile) => profile,
        Err(error) => {
            let rollback = rollback_pc_pair_commit(&pairs, &target);
            return Err(match rollback {
                Ok(()) => format!("paired-PC committed profile failed reload: {error}"),
                Err(rollback_error) => format!(
                    "paired-PC committed profile failed reload: {error}; rollback failed: {rollback_error}"
                ),
            });
        }
    };
    if loaded.address != address
        || loaded.remote_peer_id != remote_peer_id
        || loaded.remote_verify_key != remote_verify_key
        || loaded.local_seed != local_seed
    {
        let rollback = rollback_pc_pair_commit(&pairs, &target);
        return Err(match rollback {
            Ok(()) => "paired-PC committed profile failed verification".into(),
            Err(rollback_error) => format!(
                "paired-PC committed profile failed verification; rollback failed: {rollback_error}"
            ),
        });
    }

    Ok(pc_pair_public_receipt(&loaded))
}

fn revoke_pc_pair_profile(root: &Path, peer: &str) -> Result<(), String> {
    if !valid_pc_peer_alias(peer) {
        return Err("paired-PC peer alias is invalid".into());
    }
    let pairs = canonical_pc_pair_directory(root, false)?;
    let target = pairs.join(format!("{peer}.pcp97"));
    let metadata = fs::symlink_metadata(&target)
        .map_err(|error| format!("paired-PC profile is unavailable: {error}"))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err("paired-PC profile is not a regular file".into());
    }
    let canonical = fs::canonicalize(&target)
        .map_err(|error| format!("canonicalize paired-PC profile: {error}"))?;
    if !canonical.starts_with(&pairs) {
        return Err("paired-PC profile escaped profile directory".into());
    }
    fs::remove_file(&canonical).map_err(|error| format!("remove paired-PC profile: {error}"))?;
    sync_pc_pair_directory(&pairs)?;
    Ok(())
}

fn list_pc_pair_profiles(root: &Path) -> Result<Vec<String>, String> {
    if !root.exists() || !root.join("pc-pairs").exists() {
        return Ok(Vec::new());
    }
    let pairs = canonical_pc_pair_directory(root, false)?;
    let mut peers = Vec::new();
    for entry in fs::read_dir(&pairs)
        .map_err(|error| format!("read paired-PC profile directory: {error}"))?
    {
        let entry = entry.map_err(|error| format!("read paired-PC profile entry: {error}"))?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let Some(peer) = name.strip_suffix(".pcp97") else {
            continue;
        };
        if valid_pc_peer_alias(peer) && load_pc_pair_profile(root, peer).is_ok() {
            peers.push(peer.to_owned());
        }
    }
    peers.sort();
    peers.dedup();
    Ok(peers)
}

fn load_pc_pair_profile(root: &Path, expected_peer: &str) -> Result<PcPairProfile, String> {
    if !valid_pc_peer_alias(expected_peer) {
        return Err("paired-PC peer alias is invalid".into());
    }
    let root = fs::canonicalize(root)
        .map_err(|error| format!("canonicalize paired-PC capability root: {error}"))?;
    let pairs = root.join("pc-pairs");
    let pairs = fs::canonicalize(&pairs)
        .map_err(|error| format!("paired-PC profile directory is unavailable: {error}"))?;
    if !pairs.starts_with(&root) {
        return Err("paired-PC profile directory escaped capability root".into());
    }

    let path = pairs.join(format!("{expected_peer}.pcp97"));
    let metadata = fs::symlink_metadata(&path)
        .map_err(|error| format!("paired-PC profile is unavailable: {error}"))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() > 4096 {
        return Err("paired-PC profile is not a bounded regular file".into());
    }
    let canonical = fs::canonicalize(&path)
        .map_err(|error| format!("canonicalize paired-PC profile: {error}"))?;
    if !canonical.starts_with(&pairs) {
        return Err("paired-PC profile escaped profile directory".into());
    }
    let text = fs::read_to_string(&canonical)
        .map_err(|error| format!("read paired-PC profile: {error}"))?;
    let lines = text.lines().collect::<Vec<_>>();
    if lines.len() != 7 || lines[0] != "NTD97_PC_PAIR_V1" || lines[6] != "END" {
        return Err("paired-PC profile framing is invalid".into());
    }

    let peer = profile_value(lines[1], "peer=")?.to_owned();
    if peer != expected_peer || !valid_pc_peer_alias(&peer) {
        return Err("paired-PC profile peer alias mismatch".into());
    }
    let address = validate_pc_socket_address(
        profile_value(lines[2], "address=")?
            .parse::<SocketAddr>()
            .map_err(|_| "paired-PC profile address is invalid".to_owned())?,
    )?;
    let local_seed = decode_fixed_hex::<32>(profile_value(lines[3], "local_seed=")?)?;
    let remote_peer_id = decode_fixed_hex::<16>(profile_value(lines[4], "remote_peer_id=")?)?;
    let remote_verify_key = decode_fixed_hex::<32>(profile_value(lines[5], "remote_verify_key=")?)?;
    PairingRecord::from_public(remote_peer_id, remote_verify_key)
        .map_err(|error| format!("paired-PC pinned identity is invalid: {error:?}"))?;

    Ok(PcPairProfile {
        peer,
        address,
        local_seed,
        remote_peer_id,
        remote_verify_key,
    })
}

fn fresh_pc_handshake_entropy() -> Result<HandshakeEntropy, String> {
    Ok(HandshakeEntropy::from_seed(os_random::<32>()?))
}

fn expected_remote_pc_capability(
    canonical: &str,
) -> Option<(&'static str, SideEffectClass, &'static str)> {
    match canonical {
        "pc.observe" => Some(("pc.system.observe", SideEffectClass::ReadOnly, "pc.observe")),
        "pc.execute" => Some((
            "pc.process.execute",
            SideEffectClass::ExternalWrite,
            "pc.execute",
        )),
        "pc.artifact.read" => Some((
            "pc.artifact.read",
            SideEffectClass::ReadOnly,
            "pc.artifact.read",
        )),
        "pc.artifact.write" => Some((
            "pc.artifact.write",
            SideEffectClass::ExternalWrite,
            "pc.artifact.write",
        )),
        _ => None,
    }
}

fn remote_pc_capability_matches(
    capability: &RemoteCapability,
    id: &str,
    side_effect: SideEffectClass,
    scope: &str,
) -> bool {
    capability.id == id
        && capability.version == 1
        && capability.side_effect == side_effect
        && capability.verification_required
        && !capability.rollback_supported
        && !capability.resumable
        && capability.required_scopes.len() == 1
        && capability.required_scopes[0] == scope
}

struct AndroidPairedPcAdapter {
    peer: String,
    remote_capability: String,
    inner: PairedPcAdapter<TcpFrameTransport>,
}

impl AndroidPairedPcAdapter {
    fn connect(root: &Path, peer: &str, canonical_capability: &str) -> Result<Self, String> {
        let profile = load_pc_pair_profile(root, peer)?;
        let (remote_capability, expected_effect, expected_scope) =
            expected_remote_pc_capability(canonical_capability)
                .ok_or_else(|| "unsupported paired-PC capability".to_owned())?;
        let local = PairedIdentity::from_seed(profile.local_seed);
        let remote = PairingRecord::from_public(profile.remote_peer_id, profile.remote_verify_key)
            .map_err(|error| format!("load paired-PC pinned identity: {error:?}"))?;
        let (session, transport) = connect_paired_tcp(
            profile.address,
            local,
            remote,
            fresh_pc_handshake_entropy()?,
            Duration::from_secs(5),
        )
        .map_err(|error| format!("establish authenticated PCF97 session: {error:?}"))?;
        let mut inner = PairedPcAdapter::new(
            profile.peer.clone(),
            remote_capability,
            1,
            session,
            transport,
        )
        .map_err(|error| format!("build paired-PC adapter: {error:?}"))?;
        let capabilities = inner
            .discover_capabilities()
            .map_err(|error| format!("discover paired-PC capabilities: {error:?}"))?;
        let advertised = capabilities
            .iter()
            .find(|capability| capability.id == remote_capability)
            .ok_or_else(|| "paired PC did not advertise required capability".to_owned())?;
        if !remote_pc_capability_matches(
            advertised,
            remote_capability,
            expected_effect,
            expected_scope,
        ) {
            return Err("paired-PC capability descriptor mismatch".into());
        }
        Ok(Self {
            peer: profile.peer,
            remote_capability: remote_capability.to_owned(),
            inner,
        })
    }
}

impl CapabilityAdapter for AndroidPairedPcAdapter {
    fn execute(
        &mut self,
        action_id: ntd_runtime::ActionId,
        action: &TypedAction,
    ) -> Result<AdapterResult, String> {
        match self.inner.execute(action_id, action)? {
            AdapterResult::Completed {
                mut output,
                rollback_token,
            } => {
                output.evidence.push("pcf97-authenticated".into());
                output.evidence.push(format!("peer:{}", self.peer));
                output
                    .evidence
                    .push(format!("remote-capability:{}", self.remote_capability));
                Ok(AdapterResult::Completed {
                    output,
                    rollback_token,
                })
            }
            other => Ok(other),
        }
    }
}

struct NativeModelReasoningProbe<'a> {
    generator: GraphGenerator<CpuReferenceProvider>,
    resolver: &'a FileBackedTensorResolver,
    scratch_tokens: Vec<u32>,
    context_limit: usize,
}

impl<'a> NativeModelReasoningProbe<'a> {
    fn new(model: &'a NativeChatModel, prompt_tokens: Vec<u32>) -> Result<Self, String> {
        let generator = GraphGenerator::new(
            model.activation.graph.clone(),
            CpuReferenceProvider,
            BTreeMap::new(),
            model.activation.manifest.token_input,
            usize::try_from(model.activation.manifest.distribution_output)
                .map_err(|_| "distribution output does not fit usize".to_owned())?,
            usize::try_from(model.activation.manifest.vocabulary_size)
                .map_err(|_| "vocabulary size does not fit usize".to_owned())?,
        )
        .map_err(|error| format!("build native reasoning generator: {error:?}"))?;

        Ok(Self {
            generator,
            resolver: &model.resolver,
            scratch_tokens: prompt_tokens,
            context_limit: model.context_limit,
        })
    }
}

impl NativeReasoningProbe for NativeModelReasoningProbe<'_> {
    fn probe(&mut self, context: CognitiveContext<'_>) -> Result<CognitiveObservation, String> {
        let distribution = self
            .generator
            .next_distribution_with_resolver(
                self.resolver,
                &self.scratch_tokens,
                self.context_limit,
            )
            .map_err(|error| format!("native reasoning forward: {error:?}"))?;
        let inference = model_inference_signals(&distribution)
            .map_err(|error| format!("native reasoning signal analysis: {error:?}"))?;
        let next = sample_token(
            &distribution,
            DistributionKind::Logits,
            SamplingMode::Greedy,
            usize::try_from(context.iteration)
                .map_err(|_| "reasoning iteration does not fit usize".to_owned())?,
        )
        .map_err(|error| format!("native reasoning token selection: {error:?}"))?;
        let token =
            u32::try_from(next).map_err(|_| "reasoning token does not fit u32".to_owned())?;
        self.scratch_tokens.push(token);

        Ok(CognitiveObservation {
            summary: format!(
                "native probe token={token} entropy={:.6} margin={:.6}",
                inference.normalized_entropy, inference.top_margin
            ),
            evidence: vec![
                format!("native-token:{token}"),
                format!("entropy:{:.6}", inference.normalized_entropy),
                format!("top-margin:{:.6}", inference.top_margin),
            ],
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum NativeActionPlanningOutcome {
    Direct,
    Actions(AssistantActionPlan),
    Invalid,
}

struct AndroidResourceAdapter {
    snapshot: ResourceSnapshot,
}

impl CapabilityAdapter for AndroidResourceAdapter {
    fn execute(
        &mut self,
        _action_id: ntd_runtime::ActionId,
        action: &TypedAction,
    ) -> Result<AdapterResult, String> {
        let TypedAction::DeviceObserve { surface } = action else {
            return Err("Android resource adapter only supports device.observe".into());
        };

        let mut fields = BTreeMap::new();
        fields.insert("surface".into(), surface.clone());
        match surface.as_str() {
            "battery" => {
                fields.insert(
                    "battery_percent".into(),
                    self.snapshot.battery_percent.to_string(),
                );
                fields.insert("charging".into(), self.snapshot.charging.to_string());
            }
            "thermal" => {
                fields.insert(
                    "thermal".into(),
                    format!("{:?}", self.snapshot.thermal).to_lowercase(),
                );
            }
            "memory" => {
                fields.insert(
                    "available_ram_bytes".into(),
                    self.snapshot.available_ram_bytes.to_string(),
                );
            }
            "resources" => {
                fields.insert(
                    "available_ram_bytes".into(),
                    self.snapshot.available_ram_bytes.to_string(),
                );
                fields.insert(
                    "battery_percent".into(),
                    self.snapshot.battery_percent.to_string(),
                );
                fields.insert("charging".into(), self.snapshot.charging.to_string());
                fields.insert(
                    "thermal".into(),
                    format!("{:?}", self.snapshot.thermal).to_lowercase(),
                );
                fields.insert(
                    "latency_budget_ms".into(),
                    self.snapshot.latency_budget_ms.to_string(),
                );
            }
            _ => return Err(format!("unsupported device observation surface: {surface}")),
        }

        Ok(AdapterResult::Completed {
            output: ActionOutput {
                summary: format!("verified local device observation for {surface}"),
                value: ActionValue::Fields(fields),
                evidence: vec!["android-resource-snapshot".into()],
            },
            rollback_token: None,
        })
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct AndroidResourceVerifier;

impl ActionVerifier for AndroidResourceVerifier {
    fn verify(
        &mut self,
        descriptor: &CapabilityDescriptor,
        action: &TypedAction,
        output: &ActionOutput,
    ) -> ActionVerification {
        if descriptor.id.0 != "device.observe"
            || !matches!(action, TypedAction::DeviceObserve { .. })
            || output.summary.trim().is_empty()
            || !output
                .evidence
                .iter()
                .any(|item| item == "android-resource-snapshot")
        {
            return ActionVerification::Reject {
                reason: "device observation lacks trusted local evidence".into(),
            };
        }
        ActionVerification::Accept
    }
}

#[derive(Debug)]
struct AndroidWebFetchResult {
    status: u32,
    final_url: String,
    content_type: String,
    body: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AndroidWebSearchItem {
    title: String,
    url: String,
    snippet: String,
}

#[derive(Debug)]
struct AndroidWebSearchResult {
    status: u32,
    source: String,
    items: Vec<AndroidWebSearchItem>,
}

fn normalized_search_value_hash(items: &[String]) -> Result<[u8; 32], String> {
    let count = u32::try_from(items.len())
        .map_err(|_| "normalized search result count overflow".to_owned())?;
    let mut canonical = Vec::new();
    canonical.extend_from_slice(&count.to_le_bytes());
    for item in items {
        let bytes = item.as_bytes();
        let len = u32::try_from(bytes.len())
            .map_err(|_| "normalized search result length overflow".to_owned())?;
        canonical.extend_from_slice(&len.to_le_bytes());
        canonical.extend_from_slice(bytes);
    }
    Ok(sha256(&canonical))
}

fn normalized_search_item_valid(item: &str) -> bool {
    if item.len() > 8192 || item.contains('\0') || item.contains('\r') {
        return false;
    }
    let mut lines = item.split('\n');
    let Some(title) = lines.next() else {
        return false;
    };
    let Some(url) = lines.next() else {
        return false;
    };
    let Some(_snippet) = lines.next() else {
        return false;
    };
    if lines.next().is_some() || title.trim().is_empty() || title != title.trim() {
        return false;
    }
    let Some(authority_and_path) = url.strip_prefix("https://") else {
        return false;
    };
    let authority = authority_and_path
        .split(['/', '?', '#'])
        .next()
        .unwrap_or_default();
    !authority.is_empty()
        && !authority.contains('@')
        && !authority.chars().any(char::is_whitespace)
        && !url.chars().any(char::is_whitespace)
}

struct AndroidWebSearchAdapter;

impl CapabilityAdapter for AndroidWebSearchAdapter {
    fn execute(
        &mut self,
        _action_id: ntd_runtime::ActionId,
        action: &TypedAction,
    ) -> Result<AdapterResult, String> {
        let TypedAction::WebSearch { query, max_results } = action else {
            return Err("Android web search adapter only supports web.search".into());
        };
        let result = android_web_search(query, *max_results)?;
        if result.items.len() > usize::from(*max_results) {
            return Err("normalized web.search exceeded requested result limit".into());
        }

        let items = result
            .items
            .iter()
            .map(|item| format!("{}\n{}\n{}", item.title, item.url, item.snippet))
            .collect::<Vec<_>>();
        if items.iter().any(|item| !normalized_search_item_valid(item)) {
            return Err("normalized web.search item framing is invalid".into());
        }
        let result_hash = digest_hex(&normalized_search_value_hash(&items)?);

        Ok(AdapterResult::Completed {
            output: ActionOutput {
                summary: format!(
                    "verified normalized HTTPS web search status {} with {} results",
                    result.status,
                    items.len()
                ),
                value: ActionValue::TextList(items),
                evidence: vec![
                    "android-https-search".into(),
                    "provider-boundary:runtime-configured".into(),
                    "normalization:field-map-v1".into(),
                    format!("status:{}", result.status),
                    format!("source:{}", result.source),
                    format!("results:{}", result.items.len()),
                    format!("results-sha256:{result_hash}"),
                    format!("max-results:{max_results}"),
                ],
            },
            rollback_token: None,
        })
    }
}

fn android_web_search(query: &str, max_results: u16) -> Result<AndroidWebSearchResult, String> {
    let vm = JAVA_VM.get().ok_or_else(|| {
        "Android JavaVM is not attached to the native capability runtime".to_owned()
    })?;
    let mut env = vm
        .attach_current_thread()
        .map_err(|error| format!("attach Android platform thread: {error}"))?;
    let jquery = env
        .new_string(query)
        .map_err(|error| format!("encode web.search query for Android: {error}"))?;
    let jquery_object = JObject::from(jquery);
    let encoded = env
        .call_static_method(
            "ai/ntd97/mobile/NtdWebPlatform",
            "search",
            "(Ljava/lang/String;I)[B",
            &[
                JValue::Object(&jquery_object),
                JValue::Int(i32::from(max_results)),
            ],
        )
        .and_then(|value| value.l())
        .map_err(|error| format!("invoke Android WebSearch platform boundary: {error}"))?;
    let encoded = JByteArray::from(encoded);
    let bytes = env
        .convert_byte_array(&encoded)
        .map_err(|error| format!("decode Android WebSearch platform response: {error}"))?;
    decode_android_web_search_result(&bytes)
}

fn decode_android_web_search_result(bytes: &[u8]) -> Result<AndroidWebSearchResult, String> {
    let mut cursor = PlatformCursor::new(bytes);
    if cursor.u8()? != PLATFORM_WEB_PROTOCOL_VERSION {
        return Err("unsupported Android WebSearch platform protocol".into());
    }
    let success = cursor.u8()?;
    if success == 0 {
        return Err(format!("Android WebSearch failed: {}", cursor.string()?));
    }
    if success != 1 {
        return Err("invalid Android WebSearch platform status".into());
    }

    let status = cursor.u32()?;
    if !(200..300).contains(&status) {
        return Err(format!("Android WebSearch returned status {status}"));
    }
    let source = cursor.string()?;
    if !source.starts_with("https://")
        || source.len() > 512
        || source.contains('?')
        || source.contains('#')
        || source.contains('@')
    {
        return Err("Android WebSearch source evidence is invalid".into());
    }
    let count = usize::try_from(cursor.u32()?)
        .map_err(|_| "Android WebSearch result count overflow".to_owned())?;
    if count > 20 {
        return Err("Android WebSearch result count exceeds platform limit".into());
    }

    let mut items = Vec::with_capacity(count);
    let mut total = 0usize;
    for _ in 0..count {
        let title = cursor.string()?;
        let url = cursor.string()?;
        let snippet = cursor.string()?;
        total = total
            .checked_add(title.len())
            .and_then(|value| value.checked_add(url.len()))
            .and_then(|value| value.checked_add(snippet.len()))
            .ok_or_else(|| "Android WebSearch normalized result size overflow".to_owned())?;
        if title.is_empty()
            || title.len() > 512
            || url.len() > 4096
            || snippet.len() > 4096
            || title.contains(['\n', '\r', '\t'])
            || url.contains(['\n', '\r', '\t'])
            || snippet.contains(['\n', '\r', '\t'])
        {
            return Err("Android WebSearch normalized item is invalid".into());
        }
        items.push(AndroidWebSearchItem {
            title,
            url,
            snippet,
        });
    }
    if total > MAX_PLATFORM_TEXT_BYTES || !cursor.finished() {
        return Err("invalid Android WebSearch response framing".into());
    }
    Ok(AndroidWebSearchResult {
        status,
        source,
        items,
    })
}

struct AndroidWebFetchAdapter;

impl CapabilityAdapter for AndroidWebFetchAdapter {
    fn execute(
        &mut self,
        _action_id: ntd_runtime::ActionId,
        action: &TypedAction,
    ) -> Result<AdapterResult, String> {
        let TypedAction::WebFetch { url } = action else {
            return Err("Android web adapter only supports web.fetch".into());
        };
        let result = android_https_fetch(url)?;
        let text = String::from_utf8(result.body)
            .map_err(|_| "web.fetch response is not valid UTF-8 text".to_owned())?;
        if text.len() > MAX_PLATFORM_TEXT_BYTES {
            return Err("web.fetch text exceeds native evidence limit".into());
        }

        Ok(AdapterResult::Completed {
            output: ActionOutput {
                summary: format!("verified HTTPS fetch status {}", result.status),
                value: ActionValue::Text(text),
                evidence: vec![
                    "android-https-fetch".into(),
                    format!("status:{}", result.status),
                    format!("url:{}", result.final_url),
                    format!("content-type:{}", result.content_type),
                ],
            },
            rollback_token: None,
        })
    }
}

fn android_https_fetch(url: &str) -> Result<AndroidWebFetchResult, String> {
    let vm = JAVA_VM.get().ok_or_else(|| {
        "Android JavaVM is not attached to the native capability runtime".to_owned()
    })?;
    let mut env = vm
        .attach_current_thread()
        .map_err(|error| format!("attach Android platform thread: {error}"))?;
    let jurl = env
        .new_string(url)
        .map_err(|error| format!("encode web.fetch URL for Android: {error}"))?;
    let jurl_object = JObject::from(jurl);
    let encoded = env
        .call_static_method(
            "ai/ntd97/mobile/NtdWebPlatform",
            "fetch",
            "(Ljava/lang/String;)[B",
            &[JValue::Object(&jurl_object)],
        )
        .and_then(|value| value.l())
        .map_err(|error| format!("invoke Android HTTPS platform boundary: {error}"))?;
    let encoded = JByteArray::from(encoded);
    let bytes = env
        .convert_byte_array(&encoded)
        .map_err(|error| format!("decode Android HTTPS platform response: {error}"))?;
    decode_android_web_fetch_result(&bytes)
}

fn android_https_put(
    url: &str,
    body: &[u8],
    digest: &[u8; 32],
) -> Result<AndroidWebFetchResult, String> {
    let vm = JAVA_VM.get().ok_or_else(|| {
        "Android JavaVM is not attached to the native capability runtime".to_owned()
    })?;
    let mut env = vm
        .attach_current_thread()
        .map_err(|error| format!("attach Android upload thread: {error}"))?;
    let jurl = env
        .new_string(url)
        .map_err(|error| format!("encode artifact.upload URL for Android: {error}"))?;
    let jbody = env
        .byte_array_from_slice(body)
        .map_err(|error| format!("encode artifact.upload body for Android: {error}"))?;
    let jdigest = env
        .new_string(digest_hex(digest))
        .map_err(|error| format!("encode artifact.upload digest for Android: {error}"))?;
    let jurl_object = JObject::from(jurl);
    let jbody_object = JObject::from(jbody);
    let jdigest_object = JObject::from(jdigest);
    let encoded = env
        .call_static_method(
            "ai/ntd97/mobile/NtdWebPlatform",
            "put",
            "(Ljava/lang/String;[BLjava/lang/String;)[B",
            &[
                JValue::Object(&jurl_object),
                JValue::Object(&jbody_object),
                JValue::Object(&jdigest_object),
            ],
        )
        .and_then(|value| value.l())
        .map_err(|error| format!("invoke Android HTTPS upload boundary: {error}"))?;
    let encoded = JByteArray::from(encoded);
    let bytes = env
        .convert_byte_array(&encoded)
        .map_err(|error| format!("decode Android HTTPS upload response: {error}"))?;
    decode_android_web_fetch_result(&bytes)
}

fn decode_android_web_fetch_result(bytes: &[u8]) -> Result<AndroidWebFetchResult, String> {
    let mut cursor = PlatformCursor::new(bytes);
    if cursor.u8()? != PLATFORM_WEB_PROTOCOL_VERSION {
        return Err("unsupported Android web platform protocol".into());
    }
    let success = cursor.u8()?;
    if success == 0 {
        return Err(format!("Android HTTPS fetch failed: {}", cursor.string()?));
    }
    if success != 1 {
        return Err("invalid Android web platform status".into());
    }

    let status = cursor.u32()?;
    if !(200..300).contains(&status) {
        return Err(format!("Android HTTPS fetch returned status {status}"));
    }
    let final_url = cursor.string()?;
    let content_type = cursor.string()?;
    let body = cursor.bytes()?;
    if body.len() > MAX_PLATFORM_TEXT_BYTES || !cursor.finished() {
        return Err("invalid Android HTTPS response framing".into());
    }
    Ok(AndroidWebFetchResult {
        status,
        final_url,
        content_type,
        body,
    })
}

struct AndroidBrowserObserveAdapter;

impl CapabilityAdapter for AndroidBrowserObserveAdapter {
    fn execute(
        &mut self,
        _action_id: ntd_runtime::ActionId,
        action: &TypedAction,
    ) -> Result<AdapterResult, String> {
        let TypedAction::BrowserObserve { target } = action else {
            return Err("Android browser observe adapter received wrong action".into());
        };
        let observation = android_browser_observe(target)?;
        if observation.len() > MAX_PLATFORM_TEXT_BYTES {
            return Err("browser observation exceeds native evidence limit".into());
        }
        let receipt = browser_receipt(&observation)?;
        if !browser_receipt_matches(&receipt, "observe", target, None) {
            return Err("browser observation receipt does not bind the requested target".into());
        }
        Ok(AdapterResult::Completed {
            output: ActionOutput {
                summary: "verified Android browser observation".into(),
                value: ActionValue::Text(observation),
                evidence: vec![
                    "android-webview-browser".into(),
                    "operation:observe".into(),
                    format!("receipt:{receipt}"),
                ],
            },
            rollback_token: None,
        })
    }
}

struct AndroidBrowserInteractAdapter;

impl CapabilityAdapter for AndroidBrowserInteractAdapter {
    fn execute(
        &mut self,
        _action_id: ntd_runtime::ActionId,
        action: &TypedAction,
    ) -> Result<AdapterResult, String> {
        let TypedAction::BrowserInteract {
            target,
            operation,
            value,
        } = action
        else {
            return Err("Android browser interaction adapter received wrong action".into());
        };
        if !matches!(
            operation.as_str(),
            "click" | "set_value" | "submit" | "navigate"
        ) {
            return Err("unsupported Android browser interaction operation".into());
        }
        if matches!(operation.as_str(), "click" | "submit" | "navigate") && value.is_some() {
            return Err("browser interaction operation must not carry a value".into());
        }
        if operation == "set_value" && !value.as_ref().is_some_and(|value| !value.is_empty()) {
            return Err("browser set_value requires a value".into());
        }

        let result = android_browser_interact(target, operation, value.as_deref())?;
        if result.len() > MAX_PLATFORM_TEXT_BYTES {
            return Err("browser interaction receipt exceeds native evidence limit".into());
        }
        let receipt = browser_receipt(&result)?;
        if !browser_receipt_matches(&receipt, operation, target, value.as_deref()) {
            return Err("browser interaction receipt does not bind the canonical action".into());
        }
        if operation == "set_value" {
            let expected_hash = value
                .as_ref()
                .map(|value| digest_hex(&sha256(value.as_bytes())))
                .ok_or_else(|| "browser set_value lost its value".to_owned())?;
            if browser_platform_field(&result, "value_sha256") != Some(expected_hash.as_str()) {
                return Err("browser set_value platform read-back hash mismatch".into());
            }
        }
        Ok(AdapterResult::Completed {
            output: ActionOutput {
                summary: format!("verified Android browser {operation} interaction"),
                value: ActionValue::Fields(BTreeMap::from([
                    ("operation".into(), operation.to_owned()),
                    ("target".into(), target.to_owned()),
                    ("receipt".into(), receipt.clone()),
                    ("platform".into(), result),
                ])),
                evidence: vec![
                    "android-webview-browser".into(),
                    format!("operation:{operation}"),
                    format!("receipt:{receipt}"),
                ],
            },
            rollback_token: None,
        })
    }
}

fn browser_receipt(result: &str) -> Result<String, String> {
    let receipt = result
        .lines()
        .next()
        .and_then(|line| line.strip_prefix("receipt="))
        .ok_or_else(|| "browser interaction is missing a platform receipt".to_owned())?;
    if !receipt.starts_with("android-webview:") || receipt.len() > 256 {
        return Err("browser interaction platform receipt is invalid".into());
    }
    Ok(receipt.to_owned())
}

fn browser_platform_field<'a>(result: &'a str, key: &str) -> Option<&'a str> {
    let prefix = format!("{key}=");
    result
        .lines()
        .find_map(|line| line.strip_prefix(&prefix))
}

fn browser_receipt_matches(
    receipt: &str,
    operation: &str,
    target: &str,
    value: Option<&str>,
) -> bool {
    let mut parts = receipt.split(':');
    let Some(platform) = parts.next() else {
        return false;
    };
    let Some(receipt_operation) = parts.next() else {
        return false;
    };
    let Some(sequence) = parts.next() else {
        return false;
    };
    let Some(receipt_digest) = parts.next() else {
        return false;
    };
    if parts.next().is_some()
        || platform != "android-webview"
        || receipt_operation != operation
        || sequence
            .parse::<u64>()
            .ok()
            .map_or(true, |sequence| sequence == 0)
        || receipt_digest.len() != 64
    {
        return false;
    }
    let canonical = format!("{operation}\n{target}\n{}", value.unwrap_or_default());
    digest_hex(&sha256(canonical.as_bytes())) == receipt_digest
}

fn android_browser_observe(target: &str) -> Result<String, String> {
    let vm = JAVA_VM.get().ok_or_else(|| {
        "Android JavaVM is not attached to the native capability runtime".to_owned()
    })?;
    let mut env = vm
        .attach_current_thread()
        .map_err(|error| format!("attach Android browser thread: {error}"))?;
    let jtarget = env
        .new_string(target)
        .map_err(|error| format!("encode browser observe target: {error}"))?;
    let jtarget_object = JObject::from(jtarget);
    let encoded = env
        .call_static_method(
            "ai/ntd97/mobile/NtdBrowserPlatform",
            "observe",
            "(Ljava/lang/String;)[B",
            &[JValue::Object(&jtarget_object)],
        )
        .and_then(|value| value.l())
        .map_err(|error| format!("invoke Android browser observe boundary: {error}"))?;
    let encoded = JByteArray::from(encoded);
    let bytes = env
        .convert_byte_array(&encoded)
        .map_err(|error| format!("decode Android browser observe response: {error}"))?;
    decode_android_browser_result(&bytes)
}

fn android_browser_drop_in_memory_session_for_test() -> Result<(), String> {
    let vm = JAVA_VM.get().ok_or_else(|| {
        "Android JavaVM is not attached to the native capability runtime".to_owned()
    })?;
    let mut env = vm
        .attach_current_thread()
        .map_err(|error| format!("attach Android browser test thread: {error}"))?;
    let dropped = env
        .call_static_method(
            "ai/ntd97/mobile/NtdBrowserPlatform",
            "dropInMemorySessionForTest",
            "()Z",
            &[],
        )
        .and_then(|value| value.z())
        .map_err(|error| format!("drop Android browser in-memory session: {error}"))?;
    if dropped {
        Ok(())
    } else {
        Err("Android browser test session reset was rejected".into())
    }
}

fn android_browser_interact(
    target: &str,
    operation: &str,
    value: Option<&str>,
) -> Result<String, String> {
    let vm = JAVA_VM.get().ok_or_else(|| {
        "Android JavaVM is not attached to the native capability runtime".to_owned()
    })?;
    let mut env = vm
        .attach_current_thread()
        .map_err(|error| format!("attach Android browser thread: {error}"))?;
    let jtarget = env
        .new_string(target)
        .map_err(|error| format!("encode browser interaction target: {error}"))?;
    let joperation = env
        .new_string(operation)
        .map_err(|error| format!("encode browser interaction operation: {error}"))?;
    let jvalue = env
        .new_string(value.unwrap_or_default())
        .map_err(|error| format!("encode browser interaction value: {error}"))?;
    let jtarget_object = JObject::from(jtarget);
    let joperation_object = JObject::from(joperation);
    let jvalue_object = JObject::from(jvalue);
    let encoded = env
        .call_static_method(
            "ai/ntd97/mobile/NtdBrowserPlatform",
            "interact",
            "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)[B",
            &[
                JValue::Object(&jtarget_object),
                JValue::Object(&joperation_object),
                JValue::Object(&jvalue_object),
            ],
        )
        .and_then(|value| value.l())
        .map_err(|error| format!("invoke Android browser interaction boundary: {error}"))?;
    let encoded = JByteArray::from(encoded);
    let bytes = env
        .convert_byte_array(&encoded)
        .map_err(|error| format!("decode Android browser interaction response: {error}"))?;
    decode_android_browser_result(&bytes)
}

fn decode_android_browser_result(bytes: &[u8]) -> Result<String, String> {
    let mut cursor = PlatformCursor::new(bytes);
    if cursor.u8()? != PLATFORM_BROWSER_PROTOCOL_VERSION {
        return Err("unsupported Android browser platform protocol".into());
    }
    let success = cursor.u8()?;
    let message = cursor.string()?;
    if !cursor.finished() || message.len() > MAX_PLATFORM_TEXT_BYTES {
        return Err("invalid Android browser response framing".into());
    }
    match success {
        1 if !message.trim().is_empty() => Ok(message),
        0 => Err(format!("Android browser operation failed: {message}")),
        _ => Err("invalid Android browser platform status".into()),
    }
}

fn android_platform_receipt(method: &str, value: &str) -> Result<String, String> {
    let vm = JAVA_VM.get().ok_or_else(|| {
        "Android JavaVM is not attached to the native capability runtime".to_owned()
    })?;
    let mut env = vm
        .attach_current_thread()
        .map_err(|error| format!("attach Android platform thread: {error}"))?;
    let jvalue = env
        .new_string(value)
        .map_err(|error| format!("encode Android platform action: {error}"))?;
    let jvalue_object = JObject::from(jvalue);
    let encoded = env
        .call_static_method(
            "ai/ntd97/mobile/NtdDeviceAppPlatform",
            method,
            "(Ljava/lang/String;)[B",
            &[JValue::Object(&jvalue_object)],
        )
        .and_then(|value| value.l())
        .map_err(|error| format!("invoke Android device/app platform boundary: {error}"))?;
    let encoded = JByteArray::from(encoded);
    let bytes = env
        .convert_byte_array(&encoded)
        .map_err(|error| format!("decode Android device/app response: {error}"))?;
    decode_android_platform_receipt(&bytes)
}

fn android_accessibility_receipt(
    app: &str,
    operation: &str,
    payload: &str,
) -> Result<String, String> {
    let vm = JAVA_VM.get().ok_or_else(|| {
        "Android JavaVM is not attached to the native capability runtime".to_owned()
    })?;
    let mut env = vm
        .attach_current_thread()
        .map_err(|error| format!("attach Android accessibility thread: {error}"))?;
    let japp = env
        .new_string(app)
        .map_err(|error| format!("encode accessibility package: {error}"))?;
    let joperation = env
        .new_string(operation)
        .map_err(|error| format!("encode accessibility operation: {error}"))?;
    let jpayload = env
        .new_string(payload)
        .map_err(|error| format!("encode accessibility payload: {error}"))?;
    let japp_object = JObject::from(japp);
    let joperation_object = JObject::from(joperation);
    let jpayload_object = JObject::from(jpayload);
    let encoded = env
        .call_static_method(
            "ai/ntd97/mobile/NtdDeviceAppPlatform",
            "accessibilityInteract",
            "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)[B",
            &[
                JValue::Object(&japp_object),
                JValue::Object(&joperation_object),
                JValue::Object(&jpayload_object),
            ],
        )
        .and_then(|value| value.l())
        .map_err(|error| format!("invoke Android accessibility boundary: {error}"))?;
    let encoded = JByteArray::from(encoded);
    let bytes = env
        .convert_byte_array(&encoded)
        .map_err(|error| format!("decode Android accessibility response: {error}"))?;
    decode_android_platform_receipt(&bytes)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AccessibilityActionSpec {
    selector_kind: String,
    selector_value: String,
    text_bytes: Option<usize>,
}

fn parse_accessibility_action(
    operation: &str,
    payload: &[u8],
) -> Result<AccessibilityActionSpec, String> {
    let payload = std::str::from_utf8(payload)
        .map_err(|_| "accessibility payload is not UTF-8".to_owned())?;
    if payload.is_empty()
        || payload.len() > MAX_PLATFORM_TEXT_BYTES
        || payload.chars().any(|ch| matches!(ch, '\r' | '\n' | '|'))
    {
        return Err("accessibility payload is invalid".into());
    }
    let parts = payload.split('\t').collect::<Vec<_>>();
    match operation {
        "accessibility.click"
            if parts.len() == 2
                && parts[0] == "view_id"
                && !parts[1].is_empty()
                && parts[1].len() <= 4096 =>
        {
            Ok(AccessibilityActionSpec {
                selector_kind: parts[0].to_owned(),
                selector_value: parts[1].to_owned(),
                text_bytes: None,
            })
        }
        "accessibility.set_text"
            if parts.len() == 3
                && parts[0] == "view_id"
                && !parts[1].is_empty()
                && parts[1].len() <= 4096
                && !parts[2].is_empty() =>
        {
            Ok(AccessibilityActionSpec {
                selector_kind: parts[0].to_owned(),
                selector_value: parts[1].to_owned(),
                text_bytes: Some(parts[2].len()),
            })
        }
        _ => Err("unsupported or malformed accessibility action".into()),
    }
}

fn decode_android_platform_receipt(bytes: &[u8]) -> Result<String, String> {
    let mut cursor = PlatformCursor::new(bytes);
    if cursor.u8()? != PLATFORM_DEVICE_APP_PROTOCOL_VERSION {
        return Err("unsupported Android device/app platform protocol".into());
    }
    let success = cursor.u8()?;
    let message = cursor.string()?;
    if !cursor.finished() {
        return Err("invalid Android device/app response framing".into());
    }
    match success {
        1 if !message.trim().is_empty() => Ok(message),
        0 => Err(format!("Android device/app action failed: {message}")),
        _ => Err("invalid Android device/app platform status".into()),
    }
}

struct AndroidDeviceInteractAdapter;

impl CapabilityAdapter for AndroidDeviceInteractAdapter {
    fn execute(
        &mut self,
        _action_id: ntd_runtime::ActionId,
        action: &TypedAction,
    ) -> Result<AdapterResult, String> {
        let TypedAction::DeviceInteract {
            surface,
            operation,
            argument,
        } = action
        else {
            return Err("Android device interaction adapter received wrong action".into());
        };
        if surface != "clipboard" || operation != "set_text" {
            return Err("unsupported Android device interaction".into());
        }
        let text = argument
            .as_deref()
            .ok_or_else(|| "clipboard set_text requires an argument".to_owned())?;
        let receipt = android_platform_receipt("clipboardSet", text)?;

        Ok(AdapterResult::Completed {
            output: ActionOutput {
                summary: "verified Android clipboard text write".into(),
                value: ActionValue::Fields(BTreeMap::from([
                    ("surface".into(), "clipboard".into()),
                    ("operation".into(), "set_text".into()),
                ])),
                evidence: vec![
                    "android-clipboard-write".into(),
                    "operation:set_text".into(),
                    receipt,
                ],
            },
            rollback_token: None,
        })
    }
}

struct AndroidAppActionAdapter;

impl CapabilityAdapter for AndroidAppActionAdapter {
    fn execute(
        &mut self,
        _action_id: ntd_runtime::ActionId,
        action: &TypedAction,
    ) -> Result<AdapterResult, String> {
        let TypedAction::AppAction {
            app,
            action,
            payload,
        } = action
        else {
            return Err("Android app adapter received wrong action".into());
        };

        if action == "launch" && payload.is_empty() {
            let receipt = android_platform_receipt("launchApp", app)?;
            return Ok(AdapterResult::Completed {
                output: ActionOutput {
                    summary: format!("verified Android app launch: {app}"),
                    value: ActionValue::Fields(BTreeMap::from([
                        ("package".into(), app.clone()),
                        ("operation".into(), "launch".into()),
                        ("receipt".into(), receipt.clone()),
                    ])),
                    evidence: vec![
                        "android-app-launch".into(),
                        "operation:launch".into(),
                        receipt,
                    ],
                },
                rollback_token: None,
            });
        }

        let spec = parse_accessibility_action(action, payload)?;
        let payload_text = std::str::from_utf8(payload)
            .map_err(|_| "accessibility payload is not UTF-8".to_owned())?;
        let receipt = android_accessibility_receipt(app, action, payload_text)?;
        let mut fields = BTreeMap::from([
            ("package".into(), app.clone()),
            ("operation".into(), action.clone()),
            ("selector_kind".into(), spec.selector_kind),
            ("selector".into(), spec.selector_value),
            ("receipt".into(), receipt.clone()),
        ]);
        if let Some(text_bytes) = spec.text_bytes {
            fields.insert("text_bytes".into(), text_bytes.to_string());
        }

        Ok(AdapterResult::Completed {
            output: ActionOutput {
                summary: format!("verified Android accessibility interaction: {action} on {app}"),
                value: ActionValue::Fields(fields),
                evidence: vec![
                    "android-accessibility-interaction".into(),
                    format!("operation:{action}"),
                    receipt,
                ],
            },
            rollback_token: None,
        })
    }
}

fn decode_android_storage_grant_result(bytes: &[u8]) -> Result<String, String> {
    let mut cursor = PlatformCursor::new(bytes);
    if cursor.u8()? != PLATFORM_STORAGE_GRANT_PROTOCOL_VERSION {
        return Err("unsupported Android storage grant protocol".into());
    }
    let success = cursor.u8()?;
    let message = cursor.string()?;
    if !cursor.finished() || message.len() > MAX_PLATFORM_TEXT_BYTES {
        return Err("invalid Android storage grant response framing".into());
    }
    match success {
        1 => Ok(message),
        0 => Err(format!("Android storage grant action failed: {message}")),
        _ => Err("invalid Android storage grant platform status".into()),
    }
}

fn android_granted_file_read(path: &str) -> Result<String, String> {
    let vm = JAVA_VM.get().ok_or_else(|| {
        "Android JavaVM is not attached to the native capability runtime".to_owned()
    })?;
    let mut env = vm
        .attach_current_thread()
        .map_err(|error| format!("attach Android storage thread: {error}"))?;
    let jpath = env
        .new_string(path)
        .map_err(|error| format!("encode granted file path: {error}"))?;
    let jpath_object = JObject::from(jpath);
    let encoded = env
        .call_static_method(
            "ai/ntd97/mobile/NtdStorageGrantPlatform",
            "read",
            "(Ljava/lang/String;)[B",
            &[JValue::Object(&jpath_object)],
        )
        .and_then(|value| value.l())
        .map_err(|error| format!("invoke Android granted file read: {error}"))?;
    let encoded = JByteArray::from(encoded);
    let bytes = env
        .convert_byte_array(&encoded)
        .map_err(|error| format!("decode Android granted file read: {error}"))?;
    decode_android_storage_grant_result(&bytes)
}

fn android_granted_file_write(path: &str, body: &[u8]) -> Result<String, String> {
    let vm = JAVA_VM.get().ok_or_else(|| {
        "Android JavaVM is not attached to the native capability runtime".to_owned()
    })?;
    let mut env = vm
        .attach_current_thread()
        .map_err(|error| format!("attach Android storage thread: {error}"))?;
    let jpath = env
        .new_string(path)
        .map_err(|error| format!("encode granted file path: {error}"))?;
    let jbody = env
        .byte_array_from_slice(body)
        .map_err(|error| format!("encode granted file bytes: {error}"))?;
    let jpath_object = JObject::from(jpath);
    let jbody_object = JObject::from(jbody);
    let encoded = env
        .call_static_method(
            "ai/ntd97/mobile/NtdStorageGrantPlatform",
            "write",
            "(Ljava/lang/String;[B)[B",
            &[JValue::Object(&jpath_object), JValue::Object(&jbody_object)],
        )
        .and_then(|value| value.l())
        .map_err(|error| format!("invoke Android granted file write: {error}"))?;
    let encoded = JByteArray::from(encoded);
    let bytes = env
        .convert_byte_array(&encoded)
        .map_err(|error| format!("decode Android granted file write: {error}"))?;
    decode_android_storage_grant_result(&bytes)
}

fn granted_file_parts(path: &str) -> Result<(&str, &str), String> {
    let (alias, relative) = path
        .split_once('\t')
        .ok_or_else(|| "granted file path must contain alias and relative path".to_owned())?;
    if alias.trim().is_empty()
        || relative.trim().is_empty()
        || alias != alias.trim()
        || relative != relative.trim()
    {
        return Err("invalid granted file path".into());
    }
    Ok((alias, relative))
}

struct AndroidUserGrantedFileAdapter;

impl CapabilityAdapter for AndroidUserGrantedFileAdapter {
    fn execute(
        &mut self,
        _action_id: ntd_runtime::ActionId,
        action: &TypedAction,
    ) -> Result<AdapterResult, String> {
        match action {
            TypedAction::FileRead { path } => {
                let (alias, relative) = granted_file_parts(path)?;
                let text = android_granted_file_read(path)?;
                Ok(AdapterResult::Completed {
                    output: ActionOutput {
                        summary: format!("verified user-granted file read: {alias}/{relative}"),
                        value: ActionValue::Text(text),
                        evidence: vec![
                            "android-user-granted-file".into(),
                            format!("grant:{alias}"),
                            format!("path:{relative}"),
                            "operation:read".into(),
                        ],
                    },
                    rollback_token: None,
                })
            }
            TypedAction::FileWrite { path, bytes } => {
                let (alias, relative) = granted_file_parts(path)?;
                let receipt = android_granted_file_write(path, bytes)?;
                Ok(AdapterResult::Completed {
                    output: ActionOutput {
                        summary: format!("verified user-granted file write: {alias}/{relative}"),
                        value: ActionValue::Fields(BTreeMap::from([
                            ("grant".into(), alias.to_owned()),
                            ("path".into(), relative.to_owned()),
                            ("bytes".into(), bytes.len().to_string()),
                            ("receipt".into(), receipt.clone()),
                        ])),
                        evidence: vec![
                            "android-user-granted-file".into(),
                            format!("grant:{alias}"),
                            format!("path:{relative}"),
                            "operation:write".into(),
                            format!("receipt:{receipt}"),
                        ],
                    },
                    rollback_token: None,
                })
            }
            _ => Err("Android user-granted file adapter received wrong action".into()),
        }
    }
}

#[derive(Debug, Clone)]
struct AndroidScopedFileAdapter {
    root: PathBuf,
}

impl AndroidScopedFileAdapter {
    fn new(root: PathBuf) -> Result<Self, String> {
        fs::create_dir_all(&root)
            .map_err(|error| format!("create app-private capability root: {error}"))?;
        let root = fs::canonicalize(root)
            .map_err(|error| format!("canonicalize app-private capability root: {error}"))?;
        Ok(Self { root })
    }

    fn scoped_path(&self, relative: &str) -> Result<PathBuf, String> {
        let path = Path::new(relative);
        if relative.trim().is_empty() || path.is_absolute() {
            return Err("file capability path must be relative to app-private storage".into());
        }
        let mut target = self.root.clone();
        for component in path.components() {
            match component {
                Component::Normal(name) => target.push(name),
                _ => return Err("file capability path traversal is not allowed".into()),
            }
        }

        let mut current = self.root.clone();
        if let Ok(relative_target) = target.strip_prefix(&self.root) {
            for component in relative_target.components() {
                current.push(component.as_os_str());
                if current.exists()
                    && fs::symlink_metadata(&current)
                        .map_err(|error| format!("inspect app-private path: {error}"))?
                        .file_type()
                        .is_symlink()
                {
                    return Err(
                        "symbolic links are not allowed in app-private capability paths".into(),
                    );
                }
            }
        }
        Ok(target)
    }

    fn read_text(&self, relative: &str) -> Result<String, String> {
        let target = self.scoped_path(relative)?;
        let bytes = fs::read(&target).map_err(|error| format!("read app-private file: {error}"))?;
        if bytes.len() > MAX_PLATFORM_TEXT_BYTES {
            return Err("app-private file exceeds native evidence limit".into());
        }
        String::from_utf8(bytes).map_err(|_| "app-private file is not UTF-8 text".into())
    }

    fn write_atomic(
        &self,
        action_id: ntd_runtime::ActionId,
        relative: &str,
        bytes: &[u8],
    ) -> Result<Vec<u8>, String> {
        if bytes.len() > MAX_PLATFORM_TEXT_BYTES {
            return Err("app-private file write exceeds native limit".into());
        }
        let target = self.scoped_path(relative)?;
        let parent = target
            .parent()
            .ok_or_else(|| "app-private file path has no parent".to_owned())?;
        fs::create_dir_all(parent)
            .map_err(|error| format!("create app-private file directory: {error}"))?;
        let canonical_parent = fs::canonicalize(parent)
            .map_err(|error| format!("canonicalize app-private file directory: {error}"))?;
        if !canonical_parent.starts_with(&self.root) {
            return Err("app-private file directory escaped capability root".into());
        }

        let rollback = match fs::read(&target) {
            Ok(previous) => {
                if previous.len() > MAX_PLATFORM_TEXT_BYTES {
                    return Err("existing app-private file exceeds rollback limit".into());
                }
                let mut token = Vec::with_capacity(previous.len() + 5);
                token.push(1);
                token.extend_from_slice(
                    &u32::try_from(previous.len())
                        .map_err(|_| "rollback length overflow".to_owned())?
                        .to_le_bytes(),
                );
                token.extend_from_slice(&previous);
                token
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => vec![0],
            Err(error) => return Err(format!("snapshot app-private file for rollback: {error}")),
        };

        let file_name = target
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| "invalid app-private file name".to_owned())?;
        let temp = canonical_parent.join(format!(".{file_name}.ntd97-{}", action_id.0));
        let mut output = fs::File::create(&temp)
            .map_err(|error| format!("create app-private temp file: {error}"))?;
        output
            .write_all(bytes)
            .map_err(|error| format!("write app-private temp file: {error}"))?;
        output
            .sync_all()
            .map_err(|error| format!("sync app-private temp file: {error}"))?;
        drop(output);
        fs::rename(&temp, &target).map_err(|error| format!("commit app-private file: {error}"))?;
        Ok(rollback)
    }

    fn rollback_write(&self, relative: &str, token: &[u8]) -> Result<(), String> {
        let target = self.scoped_path(relative)?;
        match token.first().copied() {
            Some(0) if token.len() == 1 => match fs::remove_file(&target) {
                Ok(()) => Ok(()),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
                Err(error) => Err(format!("rollback app-private file removal: {error}")),
            },
            Some(1) if token.len() >= 5 => {
                let len = u32::from_le_bytes([token[1], token[2], token[3], token[4]]) as usize;
                if len > MAX_PLATFORM_TEXT_BYTES || token.len() != len + 5 {
                    return Err("invalid app-private rollback token".into());
                }
                let parent = target
                    .parent()
                    .ok_or_else(|| "rollback path has no parent".to_owned())?;
                fs::create_dir_all(parent)
                    .map_err(|error| format!("create rollback directory: {error}"))?;
                fs::write(target, &token[5..])
                    .map_err(|error| format!("restore app-private rollback bytes: {error}"))
            }
            _ => Err("invalid app-private rollback token".into()),
        }
    }
}

impl CapabilityAdapter for AndroidScopedFileAdapter {
    fn execute(
        &mut self,
        action_id: ntd_runtime::ActionId,
        action: &TypedAction,
    ) -> Result<AdapterResult, String> {
        match action {
            TypedAction::FileRead { path } => {
                let text = self.read_text(path)?;
                Ok(AdapterResult::Completed {
                    output: ActionOutput {
                        summary: format!("verified app-private file read: {path}"),
                        value: ActionValue::Text(text),
                        evidence: vec![
                            "android-app-private-file".into(),
                            format!("path:{path}"),
                            "operation:read".into(),
                        ],
                    },
                    rollback_token: None,
                })
            }
            TypedAction::FileWrite { path, bytes } => {
                let rollback = self.write_atomic(action_id, path, bytes)?;
                Ok(AdapterResult::Completed {
                    output: ActionOutput {
                        summary: format!("verified app-private file write: {path}"),
                        value: ActionValue::Text(format!("{} bytes", bytes.len())),
                        evidence: vec![
                            "android-app-private-file".into(),
                            format!("path:{path}"),
                            "operation:write".into(),
                        ],
                    },
                    rollback_token: Some(rollback),
                })
            }
            _ => Err("Android scoped file adapter only supports file.read/file.write".into()),
        }
    }

    fn rollback(
        &mut self,
        _action_id: ntd_runtime::ActionId,
        action: &TypedAction,
        rollback_token: &[u8],
    ) -> Result<(), String> {
        let TypedAction::FileWrite { path, .. } = action else {
            return Err("rollback is only supported for file.write".into());
        };
        self.rollback_write(path, rollback_token)
    }
}

#[derive(Debug, Clone)]
struct AndroidArtifactDownloadAdapter {
    file: AndroidScopedFileAdapter,
}

impl AndroidArtifactDownloadAdapter {
    fn new(root: PathBuf) -> Result<Self, String> {
        Ok(Self {
            file: AndroidScopedFileAdapter::new(root)?,
        })
    }

    fn rollback_snapshot(&self, relative: &str) -> Result<Vec<u8>, String> {
        let target = self.file.scoped_path(relative)?;
        match fs::read(&target) {
            Ok(previous) => {
                if previous.len() > MAX_PLATFORM_TEXT_BYTES {
                    return Err("existing artifact exceeds rollback limit".into());
                }
                let mut token = Vec::with_capacity(previous.len() + 5);
                token.push(1);
                token.extend_from_slice(
                    &u32::try_from(previous.len())
                        .map_err(|_| "artifact rollback length overflow".to_owned())?
                        .to_le_bytes(),
                );
                token.extend_from_slice(&previous);
                Ok(token)
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(vec![0]),
            Err(error) => Err(format!("snapshot artifact for rollback: {error}")),
        }
    }

    fn stage_path(&self, action_id: ntd_runtime::ActionId) -> Result<PathBuf, String> {
        let staging = self.file.root.join(".ntd97-download");
        fs::create_dir_all(&staging)
            .map_err(|error| format!("create artifact staging directory: {error}"))?;
        let canonical = fs::canonicalize(&staging)
            .map_err(|error| format!("canonicalize artifact staging directory: {error}"))?;
        if !canonical.starts_with(&self.file.root) {
            return Err("artifact staging escaped capability root".into());
        }
        Ok(canonical.join(format!("{}.part", action_id.0)))
    }

    fn write_staging(
        &self,
        action_id: ntd_runtime::ActionId,
        body: &[u8],
    ) -> Result<PathBuf, String> {
        if body.len() > MAX_PLATFORM_TEXT_BYTES {
            return Err("artifact exceeds native download limit".into());
        }
        let stage = self.stage_path(action_id)?;
        let mut output = fs::File::create(&stage)
            .map_err(|error| format!("create artifact staging file: {error}"))?;
        output
            .write_all(body)
            .map_err(|error| format!("write artifact staging file: {error}"))?;
        output
            .sync_all()
            .map_err(|error| format!("sync artifact staging file: {error}"))?;
        Ok(stage)
    }

    fn commit_staging(
        &self,
        action_id: ntd_runtime::ActionId,
        relative: &str,
        expected_hash: &[u8; 32],
    ) -> Result<(usize, [u8; 32]), String> {
        let stage = self.stage_path(action_id)?;
        let bytes = fs::read(&stage).map_err(|error| format!("read staged artifact: {error}"))?;
        let digest = sha256(&bytes);
        if &digest != expected_hash {
            return Err("staged artifact hash mismatch".into());
        }

        let target = self.file.scoped_path(relative)?;
        let parent = target
            .parent()
            .ok_or_else(|| "artifact path has no parent".to_owned())?;
        fs::create_dir_all(parent)
            .map_err(|error| format!("create artifact target directory: {error}"))?;
        let canonical_parent = fs::canonicalize(parent)
            .map_err(|error| format!("canonicalize artifact target directory: {error}"))?;
        if !canonical_parent.starts_with(&self.file.root) {
            return Err("artifact target escaped capability root".into());
        }

        fs::rename(&stage, &target).map_err(|error| format!("commit staged artifact: {error}"))?;
        fs::File::open(&canonical_parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| format!("sync artifact directory: {error}"))?;
        Ok((bytes.len(), digest))
    }
}

fn encode_artifact_resume_token(
    digest: &[u8; 32],
    rollback: &[u8],
    final_url: &str,
) -> Result<Vec<u8>, String> {
    let rollback_len =
        u32::try_from(rollback.len()).map_err(|_| "artifact rollback token overflow".to_owned())?;
    let url = final_url.as_bytes();
    let url_len = u32::try_from(url.len()).map_err(|_| "artifact final URL overflow".to_owned())?;
    let mut out = Vec::with_capacity(1 + 32 + 4 + rollback.len() + 4 + url.len());
    out.push(1);
    out.extend_from_slice(digest);
    out.extend_from_slice(&rollback_len.to_le_bytes());
    out.extend_from_slice(rollback);
    out.extend_from_slice(&url_len.to_le_bytes());
    out.extend_from_slice(url);
    Ok(out)
}

fn decode_artifact_resume_token(token: &[u8]) -> Result<([u8; 32], Vec<u8>, String), String> {
    if token.len() < 1 + 32 + 4 + 4 || token[0] != 1 {
        return Err("invalid artifact resume token".into());
    }
    let mut digest = [0u8; 32];
    digest.copy_from_slice(&token[1..33]);
    let rollback_len = u32::from_le_bytes([token[33], token[34], token[35], token[36]]) as usize;
    let rollback_start = 37usize;
    let rollback_end = rollback_start
        .checked_add(rollback_len)
        .ok_or_else(|| "artifact resume token overflow".to_owned())?;
    let url_len_end = rollback_end
        .checked_add(4)
        .ok_or_else(|| "artifact resume token overflow".to_owned())?;
    if url_len_end > token.len() {
        return Err("truncated artifact resume token".into());
    }
    let url_len = u32::from_le_bytes([
        token[rollback_end],
        token[rollback_end + 1],
        token[rollback_end + 2],
        token[rollback_end + 3],
    ]) as usize;
    let url_end = url_len_end
        .checked_add(url_len)
        .ok_or_else(|| "artifact resume token overflow".to_owned())?;
    if url_end != token.len() {
        return Err("non-canonical artifact resume token".into());
    }
    let final_url = String::from_utf8(token[url_len_end..url_end].to_vec())
        .map_err(|_| "artifact resume URL is not UTF-8".to_owned())?;
    Ok((
        digest,
        token[rollback_start..rollback_end].to_vec(),
        final_url,
    ))
}

impl CapabilityAdapter for AndroidArtifactDownloadAdapter {
    fn execute(
        &mut self,
        action_id: ntd_runtime::ActionId,
        action: &TypedAction,
    ) -> Result<AdapterResult, String> {
        let TypedAction::ArtifactDownload { url, path } = action else {
            return Err("Android artifact adapter received wrong action".into());
        };
        let result = android_https_fetch(url)?;
        if result.body.len() > MAX_PLATFORM_TEXT_BYTES {
            return Err("artifact download exceeds native limit".into());
        }

        let rollback = self.rollback_snapshot(path)?;
        self.write_staging(action_id, &result.body)?;
        let digest = sha256(&result.body);
        let resume_token = encode_artifact_resume_token(&digest, &rollback, &result.final_url)?;
        Ok(AdapterResult::Suspended {
            resume_token,
            note: "artifact downloaded and staged for verified commit".into(),
        })
    }

    fn resume(
        &mut self,
        action_id: ntd_runtime::ActionId,
        action: &TypedAction,
        resume_token: &[u8],
    ) -> Result<AdapterResult, String> {
        let TypedAction::ArtifactDownload { path, .. } = action else {
            return Err("Android artifact adapter received wrong resume action".into());
        };
        let (expected_hash, rollback, final_url) = decode_artifact_resume_token(resume_token)?;
        let (size, digest) = self.commit_staging(action_id, path, &expected_hash)?;
        Ok(AdapterResult::Completed {
            output: ActionOutput {
                summary: format!("verified artifact download committed: {path}"),
                value: ActionValue::Fields(BTreeMap::from([
                    ("path".into(), path.clone()),
                    ("bytes".into(), size.to_string()),
                    ("sha256".into(), digest_hex(&digest)),
                    ("url".into(), final_url),
                ])),
                evidence: vec![
                    "android-artifact-download".into(),
                    "operation:download".into(),
                    format!("sha256:{}", digest_hex(&digest)),
                ],
            },
            rollback_token: Some(rollback),
        })
    }

    fn rollback(
        &mut self,
        _action_id: ntd_runtime::ActionId,
        action: &TypedAction,
        rollback_token: &[u8],
    ) -> Result<(), String> {
        let TypedAction::ArtifactDownload { path, .. } = action else {
            return Err("artifact rollback received wrong action".into());
        };
        self.file.rollback_write(path, rollback_token)
    }
}

#[derive(Debug, Clone)]
struct AndroidArtifactUploadAdapter {
    file: AndroidScopedFileAdapter,
}

impl AndroidArtifactUploadAdapter {
    fn new(root: PathBuf) -> Result<Self, String> {
        Ok(Self {
            file: AndroidScopedFileAdapter::new(root)?,
        })
    }

    fn source_bytes(&self, relative: &str) -> Result<Vec<u8>, String> {
        let target = self.file.scoped_path(relative)?;
        let bytes = fs::read(&target).map_err(|error| format!("read upload source: {error}"))?;
        if bytes.is_empty() || bytes.len() > MAX_PLATFORM_TEXT_BYTES {
            return Err("upload source is empty or exceeds native limit".into());
        }
        Ok(bytes)
    }
}

impl CapabilityAdapter for AndroidArtifactUploadAdapter {
    fn execute(
        &mut self,
        _action_id: ntd_runtime::ActionId,
        action: &TypedAction,
    ) -> Result<AdapterResult, String> {
        let TypedAction::ArtifactUpload { path, .. } = action else {
            return Err("Android artifact upload adapter received wrong action".into());
        };
        let bytes = self.source_bytes(path)?;
        let digest = sha256(&bytes);
        let mut resume_token = Vec::with_capacity(33);
        resume_token.push(1);
        resume_token.extend_from_slice(&digest);
        Ok(AdapterResult::Suspended {
            resume_token,
            note: "artifact upload source hashed and ready for idempotent PUT".into(),
        })
    }

    fn resume(
        &mut self,
        _action_id: ntd_runtime::ActionId,
        action: &TypedAction,
        resume_token: &[u8],
    ) -> Result<AdapterResult, String> {
        let TypedAction::ArtifactUpload { url, path } = action else {
            return Err("Android artifact upload adapter received wrong resume action".into());
        };
        if resume_token.len() != 33 || resume_token[0] != 1 {
            return Err("invalid artifact upload resume token".into());
        }
        let mut expected = [0u8; 32];
        expected.copy_from_slice(&resume_token[1..]);
        let bytes = self.source_bytes(path)?;
        let digest = sha256(&bytes);
        if digest != expected {
            return Err("artifact upload source changed after suspension".into());
        }

        let result = match android_https_put(url, &bytes, &digest) {
            Ok(result) => result,
            Err(reason) => {
                return Ok(AdapterResult::Retryable {
                    reason,
                    resume_token: Some(resume_token.to_vec()),
                })
            }
        };

        Ok(AdapterResult::Completed {
            output: ActionOutput {
                summary: format!("verified idempotent artifact upload: {path}"),
                value: ActionValue::Fields(BTreeMap::from([
                    ("path".into(), path.clone()),
                    ("bytes".into(), bytes.len().to_string()),
                    ("sha256".into(), digest_hex(&digest)),
                    ("url".into(), result.final_url.clone()),
                    ("status".into(), result.status.to_string()),
                ])),
                evidence: vec![
                    "android-artifact-upload".into(),
                    "transport:https-put".into(),
                    format!("status:{}", result.status),
                    format!("sha256:{}", digest_hex(&digest)),
                    format!("bytes:{}", bytes.len()),
                    format!("url:{}", result.final_url),
                ],
            },
            rollback_token: None,
        })
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct AndroidProductionVerifier;

impl ActionVerifier for AndroidProductionVerifier {
    fn verify(
        &mut self,
        descriptor: &CapabilityDescriptor,
        action: &TypedAction,
        output: &ActionOutput,
    ) -> ActionVerification {
        let trusted = match (descriptor.id.0.as_str(), action) {
            ("device.observe", TypedAction::DeviceObserve { .. }) => output
                .evidence
                .iter()
                .any(|item| item == "android-resource-snapshot"),
            ("web.fetch", TypedAction::WebFetch { .. }) => {
                output
                    .evidence
                    .iter()
                    .any(|item| item == "android-https-fetch")
                    && output
                        .evidence
                        .iter()
                        .any(|item| item.starts_with("status:2"))
            }
            ("web.search", TypedAction::WebSearch { max_results, .. }) => {
                let ActionValue::TextList(items) = &output.value else {
                    return ActionVerification::Reject {
                        reason: "normalized WebSearch output is not a text list".into(),
                    };
                };
                let Ok(hash) = normalized_search_value_hash(items) else {
                    return ActionVerification::Reject {
                        reason: "normalized WebSearch output hash failed".into(),
                    };
                };
                let expected_hash = format!("results-sha256:{}", digest_hex(&hash));
                let expected_count = format!("results:{}", items.len());
                items.len() <= usize::from(*max_results)
                    && items.iter().all(|item| normalized_search_item_valid(item))
                    && output
                        .evidence
                        .iter()
                        .any(|item| item == "android-https-search")
                    && output
                        .evidence
                        .iter()
                        .any(|item| item == "provider-boundary:runtime-configured")
                    && output
                        .evidence
                        .iter()
                        .any(|item| item == "normalization:field-map-v1")
                    && output
                        .evidence
                        .iter()
                        .any(|item| item.starts_with("status:2"))
                    && output.evidence.iter().any(|item| item == &expected_count)
                    && output.evidence.iter().any(|item| item == &expected_hash)
                    && output.evidence.iter().any(|item| {
                        item.strip_prefix("source:")
                            .is_some_and(|source| source.starts_with("https://"))
                    })
            }
            ("browser.observe", TypedAction::BrowserObserve { target }) => {
                let ActionValue::Text(platform) = &output.value else {
                    return ActionVerification::Reject {
                        reason: "browser observe output is not platform text".into(),
                    };
                };
                let Ok(receipt) = browser_receipt(platform) else {
                    return ActionVerification::Reject {
                        reason: "browser observe output lacks receipt".into(),
                    };
                };
                browser_receipt_matches(&receipt, "observe", target, None)
                    && browser_platform_field(platform, "operation") == Some("observe")
                    && output
                        .evidence
                        .iter()
                        .any(|item| item == "android-webview-browser")
                    && output
                        .evidence
                        .iter()
                        .any(|item| item == "operation:observe")
                    && output
                        .evidence
                        .iter()
                        .any(|item| item == &format!("receipt:{receipt}"))
            }
            (
                "browser.interact",
                TypedAction::BrowserInteract {
                    target,
                    operation,
                    value,
                },
            ) => {
                let semantics_valid = match operation.as_str() {
                    "click" | "submit" | "navigate" => value.is_none(),
                    "set_value" => value.as_ref().is_some_and(|value| !value.is_empty()),
                    _ => false,
                };
                let ActionValue::Fields(fields) = &output.value else {
                    return ActionVerification::Reject {
                        reason: "browser interaction output is not fields".into(),
                    };
                };
                let Some(receipt) = fields.get("receipt") else {
                    return ActionVerification::Reject {
                        reason: "browser interaction output lacks receipt".into(),
                    };
                };
                let Some(platform) = fields.get("platform") else {
                    return ActionVerification::Reject {
                        reason: "browser interaction output lacks platform result".into(),
                    };
                };
                let value_hash_valid = if operation == "set_value" {
                    value.as_ref().is_some_and(|value| {
                        let expected_hash = digest_hex(&sha256(value.as_bytes()));
                        browser_platform_field(platform, "value_sha256")
                            == Some(expected_hash.as_str())
                    })
                } else {
                    true
                };
                semantics_valid
                    && browser_receipt_matches(
                        receipt,
                        operation,
                        target,
                        value.as_deref(),
                    )
                    && browser_platform_field(platform, "operation")
                        == Some(operation.as_str())
                    && value_hash_valid
                    && fields.get("operation") == Some(operation)
                    && fields.get("target") == Some(target)
                    && output
                        .evidence
                        .iter()
                        .any(|item| item == "android-webview-browser")
                    && output
                        .evidence
                        .iter()
                        .any(|item| item.strip_prefix("operation:") == Some(operation.as_str()))
                    && output
                        .evidence
                        .iter()
                        .any(|item| item == &format!("receipt:{receipt}"))
            }
            ("file.read", TypedAction::FileRead { .. }) => {
                output
                    .evidence
                    .iter()
                    .any(|item| item == "android-app-private-file")
                    && output.evidence.iter().any(|item| item == "operation:read")
            }
            ("file.write", TypedAction::FileWrite { .. }) => {
                output
                    .evidence
                    .iter()
                    .any(|item| item == "android-app-private-file")
                    && output.evidence.iter().any(|item| item == "operation:write")
            }
            ("file.grant.read", TypedAction::FileRead { path }) => {
                let path_matches = granted_file_parts(path)
                    .map(|(alias, relative)| {
                        output
                            .evidence
                            .iter()
                            .any(|item| item.strip_prefix("grant:") == Some(alias))
                            && output
                                .evidence
                                .iter()
                                .any(|item| item.strip_prefix("path:") == Some(relative))
                    })
                    .unwrap_or(false);
                path_matches
                    && output
                        .evidence
                        .iter()
                        .any(|item| item == "android-user-granted-file")
                    && output.evidence.iter().any(|item| item == "operation:read")
            }
            ("file.grant.write", TypedAction::FileWrite { path, bytes }) => {
                let digest = digest_hex(&sha256(bytes));
                let receipt_suffix = format!(":{}:{digest}", bytes.len());
                let path_matches = granted_file_parts(path)
                    .map(|(alias, relative)| {
                        output
                            .evidence
                            .iter()
                            .any(|item| item.strip_prefix("grant:") == Some(alias))
                            && output
                                .evidence
                                .iter()
                                .any(|item| item.strip_prefix("path:") == Some(relative))
                    })
                    .unwrap_or(false);
                path_matches
                    && output
                        .evidence
                        .iter()
                        .any(|item| item == "android-user-granted-file")
                    && output.evidence.iter().any(|item| item == "operation:write")
                    && output
                        .evidence
                        .iter()
                        .filter_map(|item| item.strip_prefix("receipt:grant-write:"))
                        .any(|receipt| receipt.ends_with(&receipt_suffix))
                    && matches!(
                        &output.value,
                        ActionValue::Fields(fields)
                            if fields.get("bytes") == Some(&bytes.len().to_string())
                                && fields
                                    .get("receipt")
                                    .is_some_and(|receipt| receipt.ends_with(&receipt_suffix))
                    )
            }
            ("artifact.download", TypedAction::ArtifactDownload { .. }) => {
                output
                    .evidence
                    .iter()
                    .any(|item| item == "android-artifact-download")
                    && output
                        .evidence
                        .iter()
                        .any(|item| item == "operation:download")
                    && output
                        .evidence
                        .iter()
                        .any(|item| item.starts_with("sha256:"))
            }
            ("artifact.upload", TypedAction::ArtifactUpload { url, .. }) => {
                let fields = match &output.value {
                    ActionValue::Fields(fields) => Some(fields),
                    _ => None,
                };
                let field_url = fields.and_then(|fields| fields.get("url"));
                let field_status = fields.and_then(|fields| fields.get("status"));
                let field_hash = fields.and_then(|fields| fields.get("sha256"));
                let field_bytes = fields.and_then(|fields| fields.get("bytes"));
                output
                    .evidence
                    .iter()
                    .any(|item| item == "android-artifact-upload")
                    && output
                        .evidence
                        .iter()
                        .any(|item| item == "transport:https-put")
                    && field_url == Some(url)
                    && field_status.is_some_and(|status| status.starts_with('2'))
                    && field_hash.is_some_and(|hash| hash.len() == 64)
                    && field_bytes
                        .and_then(|bytes| bytes.parse::<usize>().ok())
                        .is_some_and(|bytes| bytes > 0 && bytes <= MAX_PLATFORM_TEXT_BYTES)
                    && output.evidence.iter().any(|item| {
                        field_status.is_some_and(|status| {
                            item.strip_prefix("status:") == Some(status.as_str())
                        })
                    })
                    && output.evidence.iter().any(|item| {
                        field_hash
                            .is_some_and(|hash| item.strip_prefix("sha256:") == Some(hash.as_str()))
                    })
                    && output.evidence.iter().any(|item| {
                        field_bytes.is_some_and(|bytes| {
                            item.strip_prefix("bytes:") == Some(bytes.as_str())
                        })
                    })
                    && output
                        .evidence
                        .iter()
                        .any(|item| item.strip_prefix("url:") == Some(url.as_str()))
            }
            (
                "device.interact",
                TypedAction::DeviceInteract {
                    surface,
                    operation,
                    argument,
                },
            ) => argument.as_ref().is_some_and(|text| {
                let expected_receipt = format!(
                    "clipboard-set:{}:{}",
                    text.len(),
                    digest_hex(&sha256(text.as_bytes()))
                );
                surface == "clipboard"
                    && operation == "set_text"
                    && output
                        .evidence
                        .iter()
                        .any(|item| item == "android-clipboard-write")
                    && output
                        .evidence
                        .iter()
                        .any(|item| item == "operation:set_text")
                    && output.evidence.iter().any(|item| item == &expected_receipt)
            }),
            (
                "app.action",
                TypedAction::AppAction {
                    app,
                    action,
                    payload,
                },
            ) => {
                if action == "launch" && payload.is_empty() {
                    let expected_receipt = format!("app-launch:{app}");
                    output
                        .evidence
                        .iter()
                        .any(|item| item == "android-app-launch")
                        && output
                            .evidence
                            .iter()
                            .any(|item| item == "operation:launch")
                        && output.evidence.iter().any(|item| item == &expected_receipt)
                        && matches!(
                            &output.value,
                            ActionValue::Fields(fields)
                                if fields.get("package") == Some(app)
                                    && fields.get("operation") == Some(action)
                                    && fields.get("receipt") == Some(&expected_receipt)
                        )
                } else {
                    let Ok(spec) = parse_accessibility_action(action, payload) else {
                        return ActionVerification::Reject {
                            reason: "Android accessibility action payload is invalid".into(),
                        };
                    };
                    let expected_receipt = format!(
                        "accessibility:{action}:{app}:{}",
                        digest_hex(&sha256(payload))
                    );
                    output
                        .evidence
                        .iter()
                        .any(|item| item == "android-accessibility-interaction")
                        && output
                            .evidence
                            .iter()
                            .any(|item| item == &format!("operation:{action}"))
                        && output.evidence.iter().any(|item| item == &expected_receipt)
                        && matches!(
                            &output.value,
                            ActionValue::Fields(fields)
                                if fields.get("package") == Some(app)
                                    && fields.get("operation") == Some(action)
                                    && fields.get("selector_kind") == Some(&spec.selector_kind)
                                    && fields.get("selector") == Some(&spec.selector_value)
                                    && fields.get("receipt") == Some(&expected_receipt)
                                    && match spec.text_bytes {
                                        None => true,
                                        Some(text_bytes) => {
                                            fields.get("text_bytes")
                                                == Some(&text_bytes.to_string())
                                        }
                                    }
                        )
                }
            }
            ("pc.observe", TypedAction::PcObserve { peer, .. }) => {
                output
                    .evidence
                    .iter()
                    .any(|item| item == "pcf97-authenticated")
                    && output
                        .evidence
                        .iter()
                        .any(|item| item.strip_prefix("peer:") == Some(peer.as_str()))
                    && output
                        .evidence
                        .iter()
                        .any(|item| item == "remote-capability:pc.system.observe")
            }
            ("pc.execute", TypedAction::PcExecute { peer, .. }) => {
                output
                    .evidence
                    .iter()
                    .any(|item| item == "pcf97-authenticated")
                    && output
                        .evidence
                        .iter()
                        .any(|item| item.strip_prefix("peer:") == Some(peer.as_str()))
                    && output
                        .evidence
                        .iter()
                        .any(|item| item == "remote-capability:pc.process.execute")
            }
            ("pc.artifact.read", TypedAction::PcArtifactRead { peer, .. }) => {
                output
                    .evidence
                    .iter()
                    .any(|item| item == "pcf97-authenticated")
                    && output
                        .evidence
                        .iter()
                        .any(|item| item.strip_prefix("peer:") == Some(peer.as_str()))
                    && output
                        .evidence
                        .iter()
                        .any(|item| item == "remote-capability:pc.artifact.read")
                    && matches!(&output.value, ActionValue::Bytes(_))
            }
            ("pc.artifact.write", TypedAction::PcArtifactWrite { peer, bytes, .. }) => {
                output
                    .evidence
                    .iter()
                    .any(|item| item == "pcf97-authenticated")
                    && output
                        .evidence
                        .iter()
                        .any(|item| item.strip_prefix("peer:") == Some(peer.as_str()))
                    && output
                        .evidence
                        .iter()
                        .any(|item| item == "remote-capability:pc.artifact.write")
                    && matches!(
                        &output.value,
                        ActionValue::Fields(fields)
                            if fields.get("bytes") == Some(&bytes.len().to_string())
                    )
            }
            _ => false,
        };
        if !trusted || output.summary.trim().is_empty() {
            ActionVerification::Reject {
                reason: "Android production capability output lacks trusted evidence".into(),
            }
        } else {
            ActionVerification::Accept
        }
    }
}

struct PlatformCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> PlatformCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], String> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| "platform response length overflow".to_owned())?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| "truncated platform response".to_owned())?;
        self.offset = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, String> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn bytes(&mut self) -> Result<Vec<u8>, String> {
        let len = usize::try_from(self.u32()?)
            .map_err(|_| "platform response length does not fit usize".to_owned())?;
        Ok(self.take(len)?.to_vec())
    }

    fn string(&mut self) -> Result<String, String> {
        String::from_utf8(self.bytes()?).map_err(|_| "platform response is not UTF-8".into())
    }

    fn finished(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeChatSessionStatus {
    Running,
    WaitingApproval,
    Complete,
    Cancelled,
    Failed,
}

struct NativeChatSession {
    request_id: u64,
    task_id: u64,
    cancel: Arc<AtomicBool>,
    status: NativeChatSessionStatus,
}

struct NativeState {
    bundle: Option<MobileContinuityBundle>,
    resources: ResourceSnapshot,
    input_peak_milli: u16,
    chat_model: Option<Arc<NativeChatModel>>,
    chat_session: Option<NativeChatSession>,
    chat_submit_in_progress: bool,
    chat_last_error: String,
    conversation: SovereignConversationState,
    next_chat_request_id: u64,
}

impl Default for NativeState {
    fn default() -> Self {
        Self {
            bundle: None,
            resources: ResourceSnapshot {
                available_ram_bytes: 2 * 1024 * 1024 * 1024,
                battery_percent: 50,
                charging: false,
                thermal: ThermalState::Nominal,
                latency_budget_ms: 100,
            },
            input_peak_milli: 0,
            chat_model: None,
            chat_session: None,
            chat_submit_in_progress: false,
            chat_last_error: String::new(),
            conversation: SovereignConversationState::new(CognitiveIdentity(*b"NTD97-ASSISTANT1")),
            next_chat_request_id: 1,
        }
    }
}

static STATE: OnceLock<Mutex<NativeState>> = OnceLock::new();

fn state() -> &'static Mutex<NativeState> {
    STATE.get_or_init(|| Mutex::new(NativeState::default()))
}

fn lock_state() -> std::sync::MutexGuard<'static, NativeState> {
    state()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn wake_reason(value: &str) -> WakeReason {
    match value {
        "user_interaction" => WakeReason::UserInteraction,
        "foreground" | "foreground_execution" => WakeReason::ForegroundExecution,
        "retry" | "retry_backoff" => WakeReason::RetryBackoff,
        "reboot" => WakeReason::Reboot,
        "approval_resolved" => WakeReason::ApprovalResolved,
        "network_available" => WakeReason::NetworkAvailable,
        "charging" => WakeReason::Charging,
        "manual_recovery" => WakeReason::ManualRecovery,
        _ => WakeReason::ScheduledWork,
    }
}

fn has_eligible_work(state: MobileContinuityState) -> bool {
    !matches!(
        state,
        MobileContinuityState::Completed | MobileContinuityState::FailedRecoverable
    )
}

fn requires_foreground(state: MobileContinuityState) -> bool {
    state == MobileContinuityState::ActiveExecution
}

fn status_for(state: MobileContinuityState) -> &'static str {
    match state {
        MobileContinuityState::Interactive => "Native runtime ready",
        MobileContinuityState::ActiveExecution => "Native task execution active",
        MobileContinuityState::Checkpointed => "Native task checkpoint restored",
        MobileContinuityState::SuspendedByOs => "Native task suspended by Android",
        MobileContinuityState::WaitingCondition => "Native task waiting for condition",
        MobileContinuityState::WaitingApproval => "Native action requires approval",
        MobileContinuityState::Reconstructing => "Native runtime reconstructing task",
        MobileContinuityState::VerifyingResume => "Native runtime verifying resume",
        MobileContinuityState::Completed => "Native task completed",
        MobileContinuityState::FailedRecoverable => "Native task requires recovery",
    }
}

fn push_string(out: &mut Vec<u8>, value: &str) {
    let bytes = value.as_bytes();
    let len = u32::try_from(bytes.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&bytes[..usize::try_from(len).unwrap_or(bytes.len()).min(bytes.len())]);
}

fn encode_restore_result(
    restored: bool,
    eligible: bool,
    foreground: bool,
    status: &str,
    bundle: Option<&MobileContinuityBundle>,
) -> Vec<u8> {
    let mut out = vec![
        BRIDGE_PROTOCOL_VERSION,
        u8::from(restored),
        u8::from(eligible),
        u8::from(foreground),
    ];
    push_string(&mut out, status);

    if let Some(approval) = bundle.and_then(|bundle| bundle.pending_approval.as_ref()) {
        out.push(1);
        push_string(&mut out, &approval.capability);
        push_string(&mut out, &approval.rationale);
    } else {
        out.push(0);
    }
    out
}

fn encode_avatar_state(
    bundle: Option<&MobileContinuityBundle>,
    resources: ResourceSnapshot,
    input_peak_milli: u16,
) -> Vec<u8> {
    let mut controller = AvatarController::new(AvatarSurface::InApp);
    controller.set_lip_sync(LipSyncState {
        amplitude_milli: input_peak_milli,
        viseme: 0,
        speaking: false,
    });

    let (continuity, restored) = match bundle {
        Some(bundle) => (
            bundle.state,
            restore_mobile_continuity_bundle(bundle.clone()).ok(),
        ),
        None => (MobileContinuityState::Interactive, None),
    };
    let plan = restored
        .as_ref()
        .and_then(|session| session.actions.state().plans.values().next());
    let frame = controller.frame(continuity, plan, resources);

    let mut out = vec![
        BRIDGE_PROTOCOL_VERSION,
        frame.mode as u8,
        frame.expression as u8,
        frame.gesture as u8,
    ];
    out.extend_from_slice(&frame.gaze.x_milli.to_le_bytes());
    out.extend_from_slice(&frame.gaze.y_milli.to_le_bytes());
    out.extend_from_slice(&frame.gaze.z_milli.to_le_bytes());
    out.extend_from_slice(&frame.lip_sync.amplitude_milli.to_le_bytes());
    out.push(frame.lip_sync.viseme);
    out.push(u8::from(frame.lip_sync.speaking));
    out.extend_from_slice(&frame.render.target_fps.to_le_bytes());
    push_string(&mut out, &frame.status_text);
    out
}

fn java_bytes(env: &JNIEnv<'_>, bytes: &[u8]) -> jbyteArray {
    env.byte_array_from_slice(bytes)
        .map(JByteArray::into_raw)
        .unwrap_or_else(|_| null_mut())
}

#[allow(unsafe_code)]
#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdNativeRuntimeHost_nativeRestoreAndVerify(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    mcs97: JByteArray<'_>,
    wake_reason_value: JString<'_>,
) -> jbyteArray {
    let bytes = match env.convert_byte_array(&mcs97) {
        Ok(bytes) => bytes,
        Err(_) => {
            return java_bytes(
                &env,
                &encode_restore_result(
                    false,
                    false,
                    false,
                    "Native checkpoint transfer failed",
                    None,
                ),
            )
        }
    };
    let reason = env
        .get_string(&wake_reason_value)
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "scheduled".to_owned());

    let mut bundle = match decode_mobile_continuity_bundle(&bytes) {
        Ok(bundle) => bundle,
        Err(_) => {
            return java_bytes(
                &env,
                &encode_restore_result(
                    false,
                    false,
                    false,
                    "Native continuity decode failed",
                    None,
                ),
            )
        }
    };

    if restore_mobile_continuity_bundle(bundle.clone()).is_err() {
        return java_bytes(
            &env,
            &encode_restore_result(
                false,
                false,
                false,
                "Native continuity verification failed",
                None,
            ),
        );
    }

    bundle.wake_reason = wake_reason(&reason);
    let encoded = encode_restore_result(
        true,
        has_eligible_work(bundle.state),
        requires_foreground(bundle.state),
        status_for(bundle.state),
        Some(&bundle),
    );
    lock_state().bundle = Some(bundle);
    java_bytes(&env, &encoded)
}

#[allow(unsafe_code)]
#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdNativeRuntimeHost_nativeCheckpoint(
    env: JNIEnv<'_>,
    _class: JClass<'_>,
) -> jbyteArray {
    let mut guard = lock_state();
    let Some(mut bundle) = guard.bundle.clone() else {
        return java_bytes(&env, &[]);
    };
    let Some(sequence) = bundle.checkpoint_sequence.checked_add(1) else {
        return java_bytes(&env, &[]);
    };
    bundle.checkpoint_sequence = sequence;

    match encode_mobile_continuity_bundle(&bundle) {
        Ok(bytes) => {
            guard.bundle = Some(bundle);
            java_bytes(&env, &bytes)
        }
        Err(_) => java_bytes(&env, &[]),
    }
}

#[allow(unsafe_code)]
#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdNativeRuntimeHost_nativeResolveApproval(
    _env: JNIEnv<'_>,
    _class: JClass<'_>,
    approved: jboolean,
) -> jboolean {
    let mut guard = lock_state();
    let Some(current) = guard.bundle.as_ref() else {
        return 0;
    };
    if current.state != MobileContinuityState::WaitingApproval || current.pending_approval.is_none()
    {
        return 0;
    }

    let mut candidate = current.clone();
    candidate.pending_approval = None;
    candidate.wake_reason = WakeReason::ApprovalResolved;
    candidate.state = if approved != 0 {
        MobileContinuityState::Checkpointed
    } else {
        MobileContinuityState::FailedRecoverable
    };

    if encode_mobile_continuity_bundle(&candidate).is_err() {
        return 0;
    }
    guard.bundle = Some(candidate);
    1
}

#[allow(unsafe_code)]
#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdNativeRuntimeHost_nativeAvatarState(
    env: JNIEnv<'_>,
    _class: JClass<'_>,
) -> jbyteArray {
    let guard = lock_state();
    let encoded = encode_avatar_state(
        guard.bundle.as_ref(),
        guard.resources,
        guard.input_peak_milli,
    );
    java_bytes(&env, &encoded)
}

#[allow(clippy::too_many_arguments)]
#[allow(unsafe_code)]
#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdNativeRuntimeHost_nativeUpdateResources(
    _env: JNIEnv<'_>,
    _class: JClass<'_>,
    available_ram_bytes: jlong,
    battery_percent: jint,
    charging: jboolean,
    thermal: jint,
    latency_budget_ms: jint,
) {
    let thermal = match thermal {
        2 => ThermalState::Warm,
        3 => ThermalState::Hot,
        4 => ThermalState::Critical,
        _ => ThermalState::Nominal,
    };
    let available_ram_bytes = u64::try_from(available_ram_bytes).unwrap_or_default();
    let latency_budget_ms = u32::try_from(latency_budget_ms.max(0)).unwrap_or_default();

    lock_state().resources = ResourceSnapshot {
        available_ram_bytes,
        battery_percent: battery_percent.clamp(0, 100) as u8,
        charging: charging != 0,
        thermal,
        latency_budget_ms,
    };
}

#[allow(unsafe_code)]
#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdNativeRuntimeHost_nativeAcceptMicrophonePcm(
    env: JNIEnv<'_>,
    _class: JClass<'_>,
    pcm_le: JByteArray<'_>,
    _sample_rate_hz: jint,
) {
    let Ok(bytes) = env.convert_byte_array(&pcm_le) else {
        return;
    };

    let peak = bytes
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]).unsigned_abs())
        .max()
        .unwrap_or(0);
    let peak_milli = u32::from(peak)
        .saturating_mul(1000)
        .checked_div(u32::from(i16::MAX as u16))
        .unwrap_or(0)
        .min(1000) as u16;
    lock_state().input_peak_milli = peak_milli;
}

#[allow(unsafe_code)]
#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdNativeRuntimeHost_nativePullSpeakerPcm(
    env: JNIEnv<'_>,
    _class: JClass<'_>,
    _max_samples: jint,
    _sample_rate_hz: jint,
) -> jbyteArray {
    java_bytes(&env, &[])
}

const STORIES260K_ASSET_ID: &str = "model.ntd97.stories260k";
const STORIES260K_VERSION: u32 = 1;
const STORIES260K_CONTEXT_LIMIT: usize = 128;
const STORIES260K_ANDROID_PROBE_TOKENS: usize = 16;

fn format_token_ids(tokens: &[u32]) -> String {
    tokens
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn digest_hex(digest: &[u8; 32]) -> String {
    use std::fmt::Write as _;

    let mut out = String::with_capacity(64);
    for byte in digest {
        write!(&mut out, "{byte:02x}").expect("write digest");
    }
    out
}

fn build_llama_tokenizer(
    activation: &ThinGenerativeActivation,
) -> Result<LlamaSpmTokenizer, String> {
    let NativeTokenizerModel::LlamaSpm {
        score_bits,
        token_types,
        add_space_prefix,
        add_bos_token,
        add_eos_token,
    } = &activation.tokenizer.model
    else {
        return Err("native chat currently requires a LLaMA SPM tokenizer".into());
    };

    LlamaSpmTokenizer::new(
        activation.tokenizer.tokens.clone(),
        score_bits.clone(),
        token_types.clone(),
        LlamaSpmConfig {
            bos_token: activation.tokenizer.bos_token,
            eos_token: activation.tokenizer.eos_token,
            unknown_token: activation.tokenizer.unknown_token,
            add_space_prefix: *add_space_prefix,
            add_bos_token: *add_bos_token,
            add_eos_token: *add_eos_token,
        },
    )
    .map_err(|error| format!("build tokenizer: {error:?}"))
}

fn open_chat_model(
    asset_id: &str,
    version: u32,
    capsule_path: &str,
    shard_root: &str,
    verify_key: &[u8],
    context_limit: usize,
    capability_root: &str,
) -> Result<(), String> {
    if asset_id.trim().is_empty()
        || version == 0
        || context_limit == 0
        || capability_root.trim().is_empty()
    {
        return Err("invalid native chat model identity or context limit".into());
    }
    let verify_key: [u8; 32] = verify_key
        .try_into()
        .map_err(|_| "verify key must contain exactly 32 bytes".to_owned())?;
    let native_capsule =
        fs::read(capsule_path).map_err(|error| format!("read native capsule: {error}"))?;
    let package = NativePackage {
        asset_id: asset_id.to_owned(),
        version,
        kind: AssetKind::Intelligence,
        capsule_hash: sha256(&native_capsule),
        native_capsule,
        capability: None,
    };
    let shard_store = FileTensorShardStore::open(shard_root)
        .map_err(|error| format!("open shards: {error:?}"))?;

    verify_native_package_with_shards(&package, &verify_key, &shard_store)
        .map_err(|error| format!("verify signed package: {error:?}"))?;
    let activation = activate_thin_generative_capsule(&package.native_capsule, &shard_store)
        .map_err(|error| format!("activate package: {error:?}"))?;
    let tokenizer = build_llama_tokenizer(&activation)?;
    if tokenizer.vocab_size()
        != usize::try_from(activation.manifest.vocabulary_size)
            .map_err(|_| "vocabulary size does not fit usize".to_owned())?
    {
        return Err("native chat tokenizer vocabulary mismatch".into());
    }
    let resolver = FileBackedTensorResolver::from_activation(shard_store, &activation);
    let capability_root = PathBuf::from(capability_root);
    fs::create_dir_all(&capability_root)
        .map_err(|error| format!("create native capability root: {error}"))?;
    let capability_root = fs::canonicalize(capability_root)
        .map_err(|error| format!("canonicalize native capability root: {error}"))?;

    let mut guard = lock_state();
    guard.chat_model = Some(Arc::new(NativeChatModel {
        asset_id: asset_id.to_owned(),
        version,
        activation,
        resolver,
        tokenizer,
        context_limit,
        capability_root,
    }));
    Ok(())
}

fn chat_session_active(status: NativeChatSessionStatus) -> bool {
    matches!(
        status,
        NativeChatSessionStatus::Running | NativeChatSessionStatus::WaitingApproval
    )
}

fn submit_chat(user_message: &str, max_new_tokens: usize) -> Result<u64, String> {
    if max_new_tokens == 0 {
        return Err("max_new_tokens must be greater than zero".into());
    }

    let (model, history, prior_failure, resources) = {
        let mut guard = lock_state();
        if guard.chat_submit_in_progress
            || guard
                .chat_session
                .as_ref()
                .is_some_and(|session| chat_session_active(session.status))
        {
            let session = guard
                .chat_session
                .as_ref()
                .map(|session| {
                    format!(
                        "request={} task={} status={:?}",
                        session.request_id, session.task_id, session.status
                    )
                })
                .unwrap_or_else(|| "none".to_owned());
            let active_task = guard
                .conversation
                .active()
                .map(|active| active.task_id.to_string())
                .unwrap_or_else(|| "none".to_owned());
            return Err(format!(
                "another native chat request is still running: submit_in_progress={} session={} active_task={}",
                guard.chat_submit_in_progress, session, active_task
            ));
        }
        let model = guard
            .chat_model
            .clone()
            .ok_or_else(|| "no verified native chat model is loaded".to_owned())?;
        guard.chat_submit_in_progress = true;
        (
            model,
            guard.conversation.turns().to_vec(),
            guard.conversation.prior_conversation_failure(),
            guard.resources,
        )
    };

    let result = submit_chat_reserved(
        &model,
        &history,
        prior_failure,
        resources,
        user_message,
        max_new_tokens,
    );

    lock_state().chat_submit_in_progress = false;
    result
}

fn action_planning_prompt(user_message: &str) -> String {
    format!(
        "Choose local action. Reply exactly DIRECT or NTD97_ACTIONS_V1\n1|device.observe|surface\nEND. User: {user_message}\nPlan:"
    )
}

fn run_native_action_planner(
    model: &NativeChatModel,
    user_message: &str,
) -> Result<NativeActionPlanningOutcome, String> {
    let prompt = action_planning_prompt(user_message);
    let prompt_tokens = model
        .tokenizer
        .encode(&prompt, true)
        .map_err(|error| format!("encode native action planner prompt: {error:?}"))?;
    if prompt_tokens.is_empty() || prompt_tokens.len() >= model.context_limit {
        return Ok(NativeActionPlanningOutcome::Invalid);
    }

    let available = model.context_limit - prompt_tokens.len();
    let max_new_tokens = ACTION_PLANNER_MAX_NEW_TOKENS.min(available);
    if max_new_tokens == 0 {
        return Ok(NativeActionPlanningOutcome::Invalid);
    }

    let generator = GraphGenerator::new(
        model.activation.graph.clone(),
        CpuReferenceProvider,
        BTreeMap::new(),
        model.activation.manifest.token_input,
        usize::try_from(model.activation.manifest.distribution_output)
            .map_err(|_| "distribution output does not fit usize".to_owned())?,
        usize::try_from(model.activation.manifest.vocabulary_size)
            .map_err(|_| "vocabulary size does not fit usize".to_owned())?,
    )
    .map_err(|error| format!("build native action planner generator: {error:?}"))?;
    let previous_token = prompt_tokens.last().copied();
    let mut early_decision = None;
    let mut decode_failed = false;
    let generated = generator
        .generate_tokens_with_resolver_streaming(
            &model.resolver,
            &prompt_tokens,
            GenerationConfig {
                max_new_tokens,
                context_limit: model.context_limit,
                eos_token: model.tokenizer.eos_token(),
                distribution: DistributionKind::Logits,
                sampling: SamplingMode::Greedy,
            },
            |_, generated_tokens| {
                let decoded =
                    match model
                        .tokenizer
                        .decode_after(previous_token, generated_tokens, true)
                    {
                        Ok(decoded) => decoded,
                        Err(_) => {
                            decode_failed = true;
                            return GenerationControl::Cancel;
                        }
                    };
                let candidate = decoded.trim();
                if candidate == NATIVE_ACTION_DIRECT {
                    early_decision = Some(NativeActionPlanningOutcome::Direct);
                    return GenerationControl::Cancel;
                }
                if candidate.starts_with(NATIVE_ACTION_PROTOCOL_V1)
                    && candidate.lines().any(|line| line == "END")
                {
                    early_decision = Some(match parse_native_action_plan(candidate) {
                        Ok(AssistantPlanDecision::Actions(plan)) => {
                            NativeActionPlanningOutcome::Actions(plan)
                        }
                        _ => NativeActionPlanningOutcome::Invalid,
                    });
                    return GenerationControl::Cancel;
                }

                let direct_prefix = NATIVE_ACTION_DIRECT.starts_with(candidate);
                let action_prefix = NATIVE_ACTION_PROTOCOL_V1.starts_with(candidate)
                    || candidate.starts_with(NATIVE_ACTION_PROTOCOL_V1);
                if candidate.is_empty() || direct_prefix || action_prefix {
                    GenerationControl::Continue
                } else {
                    early_decision = Some(NativeActionPlanningOutcome::Invalid);
                    GenerationControl::Cancel
                }
            },
        )
        .map_err(|error| format!("native action planner generation: {error:?}"))?;

    if decode_failed {
        return Ok(NativeActionPlanningOutcome::Invalid);
    }
    if let Some(decision) = early_decision {
        return Ok(decision);
    }

    let decoded = model
        .tokenizer
        .decode_after(previous_token, &generated.generation.generated_tokens, true)
        .map_err(|error| format!("decode native action planner output: {error:?}"))?;
    Ok(match parse_native_action_plan(&decoded) {
        Ok(AssistantPlanDecision::Direct) => NativeActionPlanningOutcome::Direct,
        Ok(AssistantPlanDecision::Actions(plan)) => NativeActionPlanningOutcome::Actions(plan),
        Err(_) => NativeActionPlanningOutcome::Invalid,
    })
}

fn governed_device_surfaces(user_message: &str) -> Option<Vec<&'static str>> {
    let normalized = user_message.to_ascii_lowercase();
    if normalized.contains("battery") || normalized.contains("charging") {
        return Some(vec!["battery", "resources"]);
    }
    if normalized.contains("thermal")
        || normalized.contains("device temperature")
        || normalized.contains("phone temperature")
    {
        return Some(vec!["thermal", "resources"]);
    }
    if normalized.contains("ram")
        || normalized.contains("device memory")
        || normalized.contains("phone memory")
        || normalized.contains("available memory")
    {
        return Some(vec!["memory", "resources"]);
    }
    if normalized.contains("device status")
        || normalized.contains("phone status")
        || normalized.contains("device resources")
        || normalized.contains("phone resources")
    {
        return Some(vec!["resources"]);
    }
    None
}

fn constrained_device_planning_prompt(user_message: &str, surfaces: &[&str]) -> String {
    format!(
        "Choose the most relevant live device observation. Options: {}. User: {user_message}\nChoice: ",
        surfaces.join(" ")
    )
}

fn run_constrained_device_planner(
    model: &NativeChatModel,
    user_message: &str,
    surfaces: &[&str],
) -> Result<NativeActionPlanningOutcome, String> {
    if surfaces.is_empty() {
        return Ok(NativeActionPlanningOutcome::Invalid);
    }
    let prompt = constrained_device_planning_prompt(user_message, surfaces);
    let prompt_tokens = model
        .tokenizer
        .encode(&prompt, true)
        .map_err(|error| format!("encode constrained planner prompt: {error:?}"))?;
    if prompt_tokens.is_empty() || prompt_tokens.len() >= model.context_limit {
        return Ok(NativeActionPlanningOutcome::Invalid);
    }

    let generator = GraphGenerator::new(
        model.activation.graph.clone(),
        CpuReferenceProvider,
        BTreeMap::new(),
        model.activation.manifest.token_input,
        usize::try_from(model.activation.manifest.distribution_output)
            .map_err(|_| "distribution output does not fit usize".to_owned())?,
        usize::try_from(model.activation.manifest.vocabulary_size)
            .map_err(|_| "vocabulary size does not fit usize".to_owned())?,
    )
    .map_err(|error| format!("build constrained planner generator: {error:?}"))?;

    let mut best: Option<(&str, f32)> = None;
    for &surface in surfaces {
        let candidate_tokens = model
            .tokenizer
            .encode(surface, false)
            .map_err(|error| format!("encode constrained planner candidate: {error:?}"))?;
        let score = generator
            .score_continuation_with_resolver(
                &model.resolver,
                &prompt_tokens,
                &candidate_tokens,
                model.context_limit,
            )
            .map_err(|error| format!("score constrained planner candidate: {error:?}"))?;
        if !score.is_finite() {
            return Ok(NativeActionPlanningOutcome::Invalid);
        }
        let replace = match best {
            None => true,
            Some((_, best_score)) => score > best_score,
        };
        if replace {
            best = Some((surface, score));
        }
    }

    let Some((surface, _)) = best else {
        return Ok(NativeActionPlanningOutcome::Invalid);
    };
    let canonical = format!("{NATIVE_ACTION_PROTOCOL_V1}\n1|device.observe|{surface}\nEND");
    Ok(match parse_native_action_plan(&canonical) {
        Ok(AssistantPlanDecision::Actions(plan)) => NativeActionPlanningOutcome::Actions(plan),
        _ => NativeActionPlanningOutcome::Invalid,
    })
}

fn valid_android_package(value: &str) -> bool {
    if value.is_empty() || value.len() > 255 {
        return false;
    }
    let mut segments = value.split('.');
    let Some(first) = segments.next() else {
        return false;
    };
    if first.is_empty()
        || !first.as_bytes()[0].is_ascii_alphabetic()
        || !first
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        return false;
    }
    let rest = segments.collect::<Vec<_>>();
    !rest.is_empty()
        && rest.iter().all(|segment| {
            !segment.is_empty()
                && segment
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        })
}

fn granted_command_path(value: &str) -> Option<String> {
    let (alias, relative) = value.split_once('/')?;
    if alias.trim().is_empty()
        || relative.trim().is_empty()
        || alias != alias.trim()
        || relative != relative.trim()
        || alias
            .chars()
            .chain(relative.chars())
            .any(|ch| matches!(ch, '\r' | '\n' | '|' | '\t'))
    {
        return None;
    }
    Some(format!("{alias}\t{relative}"))
}

fn governed_explicit_action_plan(
    user_message: &str,
) -> Result<Option<AssistantActionPlan>, String> {
    let trimmed = user_message.trim();
    let lower = trimmed.to_ascii_lowercase();

    let canonical = if lower.starts_with("search web for ") {
        let query = trimmed
            .get("search web for ".len()..)
            .ok_or_else(|| "web search command boundary failed".to_owned())?;
        if query.trim().is_empty()
            || query
                .chars()
                .any(|ch| matches!(ch, '\r' | '\n' | '|' | '\t'))
        {
            return Ok(None);
        }
        Some(format!(
            "{NATIVE_ACTION_PROTOCOL_V1}\n1|web.search|{query}\nEND"
        ))
    } else if lower.starts_with("observe browser ") {
        let target = trimmed
            .get("observe browser ".len()..)
            .ok_or_else(|| "browser observe command boundary failed".to_owned())?;
        if target.trim().is_empty()
            || target
                .chars()
                .any(|ch| matches!(ch, '\r' | '\n' | '|' | '\t'))
        {
            return Ok(None);
        }
        Some(format!(
            "{NATIVE_ACTION_PROTOCOL_V1}\n1|browser.observe|{target}\nEND"
        ))
    } else if lower == "resume browser" {
        Some(format!(
            "{NATIVE_ACTION_PROTOCOL_V1}\n1|browser.observe|session\nEND"
        ))
    } else if lower.starts_with("browser navigate ") {
        let target = trimmed
            .get("browser navigate ".len()..)
            .ok_or_else(|| "browser navigation command boundary failed".to_owned())?;
        if target.trim().is_empty()
            || target != target.trim()
            || target
                .chars()
                .any(|ch| matches!(ch, '\r' | '\n' | '|' | '\t'))
        {
            return Ok(None);
        }
        Some(format!(
            "{NATIVE_ACTION_PROTOCOL_V1}\n1|browser.interact|{target}\tnavigate\nEND"
        ))
    } else if lower.starts_with("browser submit ") {
        let target = trimmed
            .get("browser submit ".len()..)
            .ok_or_else(|| "browser submit command boundary failed".to_owned())?;
        if target.trim().is_empty()
            || target != target.trim()
            || target
                .chars()
                .any(|ch| matches!(ch, '\r' | '\n' | '|' | '\t'))
        {
            return Ok(None);
        }
        Some(format!(
            "{NATIVE_ACTION_PROTOCOL_V1}\n1|browser.interact|{target}\tsubmit\nEND"
        ))
    } else if lower.starts_with("browser set ") {
        let rest = trimmed
            .get("browser set ".len()..)
            .ok_or_else(|| "browser set_value command boundary failed".to_owned())?;
        let Some((target, value)) = rest.split_once(" to ") else {
            return Ok(None);
        };
        if target.trim().is_empty()
            || target != target.trim()
            || value.is_empty()
            || target
                .chars()
                .chain(value.chars())
                .any(|ch| matches!(ch, '\r' | '\n' | '|' | '\t'))
        {
            return Ok(None);
        }
        Some(format!(
            "{NATIVE_ACTION_PROTOCOL_V1}\n1|browser.interact|{target}\tset_value\t{value}\nEND"
        ))
    } else if lower.starts_with("browser click ") {
        let target = trimmed
            .get("browser click ".len()..)
            .ok_or_else(|| "browser interaction command boundary failed".to_owned())?;
        if target.trim().is_empty()
            || target
                .chars()
                .any(|ch| matches!(ch, '\r' | '\n' | '|' | '\t'))
        {
            return Ok(None);
        }
        Some(format!(
            "{NATIVE_ACTION_PROTOCOL_V1}\n1|browser.interact|{target}\tclick\nEND"
        ))
    } else if lower.starts_with("read granted file ") {
        let path = trimmed
            .get("read granted file ".len()..)
            .and_then(granted_command_path);
        path.map(|path| format!("{NATIVE_ACTION_PROTOCOL_V1}\n1|file.grant.read|{path}\nEND"))
    } else if lower.starts_with("write granted file ") {
        let rest = trimmed
            .get("write granted file ".len()..)
            .ok_or_else(|| "granted file write command boundary failed".to_owned())?;
        let Some((path, text)) = rest.split_once(" to ") else {
            return Ok(None);
        };
        let Some(path) = granted_command_path(path) else {
            return Ok(None);
        };
        if text.is_empty()
            || text
                .chars()
                .any(|ch| matches!(ch, '\r' | '\n' | '|' | '\t'))
        {
            return Ok(None);
        }
        Some(format!(
            "{NATIVE_ACTION_PROTOCOL_V1}\n1|file.grant.write|{path}\t{text}\nEND"
        ))
    } else if lower.starts_with("upload artifact ") {
        let rest = trimmed
            .get("upload artifact ".len()..)
            .ok_or_else(|| "artifact upload command boundary failed".to_owned())?;
        let Some((path, url)) = rest.rsplit_once(" to ") else {
            return Ok(None);
        };
        let path = path.trim();
        let url = url.trim();
        let path_value = Path::new(path);
        if path.is_empty()
            || url.is_empty()
            || !url.to_ascii_lowercase().starts_with("https://")
            || path_value.is_absolute()
            || path_value
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
            || path
                .chars()
                .chain(url.chars())
                .any(|ch| matches!(ch, '\r' | '\n' | '|' | '\t'))
        {
            return Ok(None);
        }
        Some(format!(
            "{NATIVE_ACTION_PROTOCOL_V1}\n1|artifact.upload|{url}\t{path}\nEND"
        ))
    } else if lower.starts_with("observe pc ") {
        let rest = trimmed
            .get("observe pc ".len()..)
            .ok_or_else(|| "paired-PC observe command boundary failed".to_owned())?;
        let Some((peer, surface)) = rest.split_once(' ') else {
            return Ok(None);
        };
        if !valid_pc_peer_alias(peer)
            || surface.trim().is_empty()
            || surface != surface.trim()
            || surface
                .chars()
                .any(|ch| matches!(ch, '\r' | '\n' | '|' | '\t'))
        {
            return Ok(None);
        }
        Some(format!(
            "{NATIVE_ACTION_PROTOCOL_V1}\n1|pc.observe|{peer}\t{surface}\nEND"
        ))
    } else if lower.starts_with("execute pc ") {
        let rest = trimmed
            .get("execute pc ".len()..)
            .ok_or_else(|| "paired-PC execute command boundary failed".to_owned())?;
        let Some((peer, program)) = rest.split_once(' ') else {
            return Ok(None);
        };
        if !valid_pc_peer_alias(peer)
            || program.trim().is_empty()
            || program != program.trim()
            || program
                .chars()
                .any(|ch| matches!(ch, '\r' | '\n' | '|' | '\t' | '\u{1f}' | ' '))
        {
            return Ok(None);
        }
        Some(format!(
            "{NATIVE_ACTION_PROTOCOL_V1}\n1|pc.execute|{peer}\t{program}\nEND"
        ))
    } else if lower.starts_with("read pc file ") {
        let rest = trimmed
            .get("read pc file ".len()..)
            .ok_or_else(|| "paired-PC artifact read command boundary failed".to_owned())?;
        let Some((peer, path)) = rest.split_once(' ') else {
            return Ok(None);
        };
        if !valid_pc_peer_alias(peer)
            || path.trim().is_empty()
            || path != path.trim()
            || path
                .chars()
                .any(|ch| matches!(ch, '\r' | '\n' | '|' | '\t'))
        {
            return Ok(None);
        }
        Some(format!(
            "{NATIVE_ACTION_PROTOCOL_V1}\n1|pc.artifact.read|{peer}\t{path}\nEND"
        ))
    } else if lower.starts_with("write pc file ") {
        let rest = trimmed
            .get("write pc file ".len()..)
            .ok_or_else(|| "paired-PC artifact write command boundary failed".to_owned())?;
        let Some((target, text)) = rest.split_once(" to ") else {
            return Ok(None);
        };
        let Some((peer, path)) = target.split_once(' ') else {
            return Ok(None);
        };
        if !valid_pc_peer_alias(peer)
            || path.trim().is_empty()
            || path != path.trim()
            || text.is_empty()
            || path
                .chars()
                .chain(text.chars())
                .any(|ch| matches!(ch, '\r' | '\n' | '|' | '\t'))
        {
            return Ok(None);
        }
        Some(format!(
            "{NATIVE_ACTION_PROTOCOL_V1}\n1|pc.artifact.write|{peer}\t{path}\t{text}\nEND"
        ))
    } else if lower.starts_with("set clipboard to ") {
        let value = trimmed
            .get("set clipboard to ".len()..)
            .ok_or_else(|| "clipboard command boundary failed".to_owned())?;
        if value.trim().is_empty()
            || value
                .chars()
                .any(|ch| matches!(ch, '\r' | '\n' | '|' | '\t'))
        {
            return Ok(None);
        }
        Some(format!(
            "{NATIVE_ACTION_PROTOCOL_V1}\n1|device.interact|clipboard\tset_text\t{value}\nEND"
        ))
    } else if lower.starts_with("accessibility click ") {
        let rest = trimmed
            .get("accessibility click ".len()..)
            .ok_or_else(|| "accessibility click command boundary failed".to_owned())?;
        let Some((package, view_id)) = rest.split_once(' ') else {
            return Ok(None);
        };
        if !valid_android_package(package)
            || view_id.trim().is_empty()
            || view_id != view_id.trim()
            || view_id.len() > 4096
            || view_id
                .chars()
                .any(|ch| matches!(ch, '\r' | '\n' | '|' | '\t' | ' '))
        {
            return Ok(None);
        }
        Some(format!(
            "{NATIVE_ACTION_PROTOCOL_V1}\n1|app.action|{package}\taccessibility.click\tview_id\t{view_id}\nEND"
        ))
    } else if lower.starts_with("accessibility set text ") {
        let rest = trimmed
            .get("accessibility set text ".len()..)
            .ok_or_else(|| "accessibility set_text command boundary failed".to_owned())?;
        let Some((target, text)) = rest.rsplit_once(" to ") else {
            return Ok(None);
        };
        let Some((package, view_id)) = target.split_once(' ') else {
            return Ok(None);
        };
        if !valid_android_package(package)
            || view_id.trim().is_empty()
            || view_id != view_id.trim()
            || view_id.len() > 4096
            || text.is_empty()
            || view_id
                .chars()
                .chain(text.chars())
                .any(|ch| matches!(ch, '\r' | '\n' | '|' | '\t'))
        {
            return Ok(None);
        }
        Some(format!(
            "{NATIVE_ACTION_PROTOCOL_V1}\n1|app.action|{package}\taccessibility.set_text\tview_id\t{view_id}\t{text}\nEND"
        ))
    } else if lower.starts_with("open app ") {
        let package = trimmed
            .get("open app ".len()..)
            .ok_or_else(|| "app command boundary failed".to_owned())?;
        if !valid_android_package(package) {
            return Ok(None);
        }
        Some(format!(
            "{NATIVE_ACTION_PROTOCOL_V1}\n1|app.action|{package}\tlaunch\nEND"
        ))
    } else {
        None
    };

    let Some(canonical) = canonical else {
        return Ok(None);
    };
    Ok(match parse_native_action_plan(&canonical) {
        Ok(AssistantPlanDecision::Actions(plan)) => Some(plan),
        _ => None,
    })
}

fn external_write_approval(
    action_plan: &AssistantActionPlan,
) -> Result<Option<(String, String)>, String> {
    let mut capabilities = std::collections::BTreeSet::new();
    let mut rationales = Vec::new();

    for node in &action_plan.graph.actions {
        if node.side_effect != SideEffectClass::ExternalWrite {
            continue;
        }
        let action = action_plan
            .payloads
            .get(&node.id)
            .ok_or_else(|| "external-write action payload is missing".to_owned())?;
        match (node.capability.0.as_str(), action) {
            ("artifact.upload", TypedAction::ArtifactUpload { url, path })
                if !url.trim().is_empty() && !path.trim().is_empty() =>
            {
                capabilities.insert("artifact.upload".to_owned());
                rationales.push(format!(
                    "upload app-private artifact {path} to {url} using idempotent HTTPS PUT"
                ));
            }
            ("file.grant.write", TypedAction::FileWrite { path, bytes })
                if !path.trim().is_empty() && !bytes.is_empty() =>
            {
                capabilities.insert("file.grant.write".to_owned());
                rationales.push(format!(
                    "write {} bytes to user-granted Android storage",
                    bytes.len()
                ));
            }
            (
                "browser.interact",
                TypedAction::BrowserInteract {
                    target,
                    operation,
                    value,
                },
            ) if (matches!(operation.as_str(), "click" | "submit" | "navigate")
                && value.is_none())
                || (operation == "set_value"
                    && value.as_ref().is_some_and(|value| !value.is_empty())) =>
            {
                capabilities.insert("browser.interact".to_owned());
                let subject = if operation == "navigate" {
                    "public HTTPS destination"
                } else {
                    "unique DOM selector"
                };
                rationales.push(format!(
                    "interact with the controlled browser using {operation} on {subject} {target}"
                ));
            }
            (
                "device.interact",
                TypedAction::DeviceInteract {
                    surface,
                    operation,
                    argument,
                },
            ) if surface == "clipboard"
                && operation == "set_text"
                && argument.as_ref().is_some_and(|value| !value.is_empty()) =>
            {
                capabilities.insert("device.interact".to_owned());
                rationales.push("write text to the Android clipboard".to_owned());
            }
            (
                "app.action",
                TypedAction::AppAction {
                    app,
                    action,
                    payload,
                },
            ) if action == "launch" && payload.is_empty() && valid_android_package(app) => {
                capabilities.insert("app.action".to_owned());
                rationales.push(format!("launch Android app package {app}"));
            }
            (
                "app.action",
                TypedAction::AppAction {
                    app,
                    action,
                    payload,
                },
            ) if valid_android_package(app)
                && matches!(
                    action.as_str(),
                    "accessibility.click" | "accessibility.set_text"
                ) =>
            {
                let spec = parse_accessibility_action(action, payload)?;
                capabilities.insert("app.action".to_owned());
                rationales.push(format!(
                    "perform {action} in foreground Android package {app} on {} selector {}",
                    spec.selector_kind, spec.selector_value
                ));
            }
            (
                "pc.execute",
                TypedAction::PcExecute {
                    peer,
                    program,
                    args,
                    working_dir,
                },
            ) if valid_pc_peer_alias(peer) && !program.trim().is_empty() => {
                capabilities.insert("pc.execute".to_owned());
                rationales.push(format!(
                    "execute paired-PC program {program} on {peer} with {} typed arguments{}",
                    args.len(),
                    working_dir
                        .as_ref()
                        .map(|dir| format!(" in {dir}"))
                        .unwrap_or_default()
                ));
            }
            ("pc.artifact.write", TypedAction::PcArtifactWrite { peer, path, bytes })
                if valid_pc_peer_alias(peer) && !path.trim().is_empty() && !bytes.is_empty() =>
            {
                capabilities.insert("pc.artifact.write".to_owned());
                rationales.push(format!(
                    "write {} bytes to paired-PC artifact {path} on {peer}",
                    bytes.len()
                ));
            }
            _ => return Err("unsupported external-write action requested".into()),
        }
    }

    if capabilities.is_empty() {
        return Ok(None);
    }
    Ok(Some((
        capabilities.into_iter().collect::<Vec<_>>().join(","),
        rationales.join("; "),
    )))
}

fn app_action_scope_names(
    action_plan: &AssistantActionPlan,
) -> Result<std::collections::BTreeSet<&'static str>, String> {
    let mut scopes = std::collections::BTreeSet::new();
    for node in &action_plan.graph.actions {
        if node.capability.0 != "app.action" {
            continue;
        }
        let action = action_plan
            .payloads
            .get(&node.id)
            .ok_or_else(|| "app.action payload is missing".to_owned())?;
        let TypedAction::AppAction {
            app,
            action,
            payload,
        } = action
        else {
            return Err("app.action capability/action mismatch".into());
        };
        if !valid_android_package(app) {
            return Err("app.action package is invalid".into());
        }
        match action.as_str() {
            "launch" if payload.is_empty() => {
                scopes.insert("app.launch");
            }
            "accessibility.click" | "accessibility.set_text" => {
                parse_accessibility_action(action, payload)?;
                scopes.insert("app.accessibility.interact");
            }
            _ => return Err("unsupported Android app action".into()),
        }
    }
    Ok(scopes)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum AndroidActionExecutionOutcome {
    Prepared,
    Checkpointed {
        note: String,
        requires_reconfirm: bool,
    },
    Completed,
}

fn action_checkpoint_requires_reconfirm(
    fabric: &ActionFabric,
    plan_id: ntd_runtime::ActionPlanId,
) -> bool {
    fabric
        .state()
        .plans
        .get(&plan_id.0)
        .and_then(|plan| plan.actions.get(plan.cursor))
        .is_some_and(|action| {
            action.side_effect == SideEffectClass::ExternalWrite
                && action.status == ActionStatus::Retryable
                && action.resume_token.is_none()
        })
}

struct AndroidActionExecutionContext<'a> {
    model: &'a NativeChatModel,
    task_id: u64,
    user_message: &'a str,
    prompt_limit: usize,
    resources: ResourceSnapshot,
    external_approved: bool,
    prepare_only: bool,
}

fn pc_peer_for_capability(
    action_plan: &AssistantActionPlan,
    capability: &str,
) -> Result<String, String> {
    let mut peer: Option<String> = None;
    for node in &action_plan.graph.actions {
        if node.capability.0 != capability {
            continue;
        }
        let action = action_plan
            .payloads
            .get(&node.id)
            .ok_or_else(|| "paired-PC action payload is missing".to_owned())?;
        let action_peer = match action {
            TypedAction::PcObserve { peer, .. }
            | TypedAction::PcExecute { peer, .. }
            | TypedAction::PcArtifactRead { peer, .. }
            | TypedAction::PcArtifactWrite { peer, .. } => peer,
            _ => return Err("paired-PC capability/action mismatch".into()),
        };
        if !valid_pc_peer_alias(action_peer) {
            return Err("paired-PC action peer alias is invalid".into());
        }
        match &peer {
            None => peer = Some(action_peer.clone()),
            Some(existing) if existing == action_peer => {}
            Some(_) => {
                return Err(
                    "one paired-PC capability cannot target multiple peers in one plan".into(),
                )
            }
        }
    }
    peer.ok_or_else(|| "paired-PC capability has no target peer".to_owned())
}

fn execute_android_verified_actions(
    conversation: &mut SovereignConversationState,
    action_plan: &AssistantActionPlan,
    context: AndroidActionExecutionContext<'_>,
) -> Result<AndroidActionExecutionOutcome, String> {
    let AndroidActionExecutionContext {
        model,
        task_id,
        user_message,
        prompt_limit,
        resources,
        external_approved,
        prepare_only,
    } = context;
    let supported = [
        "device.observe",
        "web.search",
        "web.fetch",
        "browser.observe",
        "browser.interact",
        "file.read",
        "file.write",
        "file.grant.read",
        "file.grant.write",
        "artifact.download",
        "artifact.upload",
        "device.interact",
        "app.action",
        "pc.observe",
        "pc.execute",
        "pc.artifact.read",
        "pc.artifact.write",
    ];
    if action_plan
        .graph
        .actions
        .iter()
        .any(|node| !supported.contains(&node.capability.0.as_str()))
    {
        conversation
            .record_action_planner_status(task_id, "unsupported")
            .map_err(|error| format!("record unsupported action plan: {error:?}"))?;
        return Err("model requested a capability without a production Android adapter".into());
    }

    let mut registry = CapabilityRegistry::new();
    let mut fabric_capabilities = std::collections::BTreeSet::new();
    for node in &action_plan.graph.actions {
        if !fabric_capabilities.insert(node.capability.0.clone()) {
            continue;
        }

        let mut descriptor = match node.capability.0.as_str() {
            "device.observe" => CapabilityDescriptor::new(
                CapabilityId("device.observe".into()),
                1,
                CapabilityDomain::Device,
                SideEffectClass::ReadOnly,
            ),
            "web.search" => {
                let mut descriptor = CapabilityDescriptor::new(
                    CapabilityId("web.search".into()),
                    1,
                    CapabilityDomain::Web,
                    SideEffectClass::ReadOnly,
                );
                if let Ok(descriptor) = descriptor.as_mut() {
                    descriptor.required_scopes.push(
                        AuthorityScope::new("network.read")
                            .map_err(|error| format!("network scope: {error:?}"))?,
                    );
                }
                descriptor
            }
            "web.fetch" => {
                let mut descriptor = CapabilityDescriptor::new(
                    CapabilityId("web.fetch".into()),
                    1,
                    CapabilityDomain::Web,
                    SideEffectClass::ReadOnly,
                );
                if let Ok(descriptor) = descriptor.as_mut() {
                    descriptor.required_scopes.push(
                        AuthorityScope::new("network.read")
                            .map_err(|error| format!("network scope: {error:?}"))?,
                    );
                }
                descriptor
            }
            "browser.observe" => {
                let mut descriptor = CapabilityDescriptor::new(
                    CapabilityId("browser.observe".into()),
                    1,
                    CapabilityDomain::Browser,
                    SideEffectClass::ReadOnly,
                );
                if let Ok(descriptor) = descriptor.as_mut() {
                    descriptor.required_scopes.extend([
                        AuthorityScope::new("network.read")
                            .map_err(|error| format!("network scope: {error:?}"))?,
                        AuthorityScope::new("browser.observe")
                            .map_err(|error| format!("browser observe scope: {error:?}"))?,
                    ]);
                }
                descriptor
            }
            "browser.interact" => {
                let mut descriptor = CapabilityDescriptor::new(
                    CapabilityId("browser.interact".into()),
                    1,
                    CapabilityDomain::Browser,
                    SideEffectClass::ExternalWrite,
                );
                if let Ok(descriptor) = descriptor.as_mut() {
                    descriptor.required_scopes.extend([
                        AuthorityScope::new("network.read")
                            .map_err(|error| format!("network scope: {error:?}"))?,
                        AuthorityScope::new("browser.interact")
                            .map_err(|error| format!("browser interaction scope: {error:?}"))?,
                    ]);
                }
                descriptor
            }
            "file.read" => {
                let mut descriptor = CapabilityDescriptor::new(
                    CapabilityId("file.read".into()),
                    1,
                    CapabilityDomain::File,
                    SideEffectClass::ReadOnly,
                );
                if let Ok(descriptor) = descriptor.as_mut() {
                    descriptor.required_scopes.push(
                        AuthorityScope::new("file.app_private")
                            .map_err(|error| format!("file scope: {error:?}"))?,
                    );
                }
                descriptor
            }
            "file.grant.read" => {
                let mut descriptor = CapabilityDescriptor::new(
                    CapabilityId("file.grant.read".into()),
                    1,
                    CapabilityDomain::File,
                    SideEffectClass::ReadOnly,
                );
                if let Ok(descriptor) = descriptor.as_mut() {
                    descriptor.required_scopes.push(
                        AuthorityScope::new("file.user_grant.read")
                            .map_err(|error| format!("granted file read scope: {error:?}"))?,
                    );
                }
                descriptor
            }
            "file.grant.write" => {
                let mut descriptor = CapabilityDescriptor::new(
                    CapabilityId("file.grant.write".into()),
                    1,
                    CapabilityDomain::File,
                    SideEffectClass::ExternalWrite,
                );
                if let Ok(descriptor) = descriptor.as_mut() {
                    descriptor.required_scopes.push(
                        AuthorityScope::new("file.user_grant.write")
                            .map_err(|error| format!("granted file write scope: {error:?}"))?,
                    );
                }
                descriptor
            }
            "file.write" => {
                let mut descriptor = CapabilityDescriptor::new(
                    CapabilityId("file.write".into()),
                    1,
                    CapabilityDomain::File,
                    SideEffectClass::Reversible,
                );
                if let Ok(descriptor) = descriptor.as_mut() {
                    descriptor.rollback_supported = true;
                    descriptor.required_scopes.push(
                        AuthorityScope::new("file.app_private")
                            .map_err(|error| format!("file scope: {error:?}"))?,
                    );
                }
                descriptor
            }
            "artifact.download" => {
                let mut descriptor = CapabilityDescriptor::new(
                    CapabilityId("artifact.download".into()),
                    1,
                    CapabilityDomain::Web,
                    SideEffectClass::Reversible,
                );
                if let Ok(descriptor) = descriptor.as_mut() {
                    descriptor.rollback_supported = true;
                    descriptor.resumable = true;
                    descriptor.required_scopes.push(
                        AuthorityScope::new("network.read")
                            .map_err(|error| format!("network scope: {error:?}"))?,
                    );
                    descriptor.required_scopes.push(
                        AuthorityScope::new("file.app_private")
                            .map_err(|error| format!("file scope: {error:?}"))?,
                    );
                }
                descriptor
            }
            "artifact.upload" => {
                let mut descriptor = CapabilityDescriptor::new(
                    CapabilityId("artifact.upload".into()),
                    1,
                    CapabilityDomain::Web,
                    SideEffectClass::ExternalWrite,
                );
                if let Ok(descriptor) = descriptor.as_mut() {
                    descriptor.resumable = true;
                    descriptor.required_scopes.push(
                        AuthorityScope::new("network.write")
                            .map_err(|error| format!("network write scope: {error:?}"))?,
                    );
                    descriptor.required_scopes.push(
                        AuthorityScope::new("file.app_private")
                            .map_err(|error| format!("file scope: {error:?}"))?,
                    );
                }
                descriptor
            }
            "device.interact" => {
                let mut descriptor = CapabilityDescriptor::new(
                    CapabilityId("device.interact".into()),
                    1,
                    CapabilityDomain::Device,
                    SideEffectClass::ExternalWrite,
                );
                if let Ok(descriptor) = descriptor.as_mut() {
                    descriptor.required_scopes.push(
                        AuthorityScope::new("device.clipboard.write")
                            .map_err(|error| format!("clipboard scope: {error:?}"))?,
                    );
                }
                descriptor
            }
            "app.action" => {
                let mut descriptor = CapabilityDescriptor::new(
                    CapabilityId("app.action".into()),
                    1,
                    CapabilityDomain::App,
                    SideEffectClass::ExternalWrite,
                );
                if let Ok(descriptor) = descriptor.as_mut() {
                    for scope in app_action_scope_names(action_plan)? {
                        descriptor.required_scopes.push(
                            AuthorityScope::new(scope)
                                .map_err(|error| format!("app action scope: {error:?}"))?,
                        );
                    }
                }
                descriptor
            }
            "pc.observe" => {
                let mut descriptor = CapabilityDescriptor::new(
                    CapabilityId("pc.observe".into()),
                    1,
                    CapabilityDomain::Pc,
                    SideEffectClass::ReadOnly,
                );
                if let Ok(descriptor) = descriptor.as_mut() {
                    descriptor.required_scopes.push(
                        AuthorityScope::new("pc.observe")
                            .map_err(|error| format!("paired-PC observe scope: {error:?}"))?,
                    );
                }
                descriptor
            }
            "pc.execute" => {
                let mut descriptor = CapabilityDescriptor::new(
                    CapabilityId("pc.execute".into()),
                    1,
                    CapabilityDomain::Pc,
                    SideEffectClass::ExternalWrite,
                );
                if let Ok(descriptor) = descriptor.as_mut() {
                    descriptor.required_scopes.push(
                        AuthorityScope::new("pc.execute")
                            .map_err(|error| format!("paired-PC execute scope: {error:?}"))?,
                    );
                }
                descriptor
            }
            "pc.artifact.read" => {
                let mut descriptor = CapabilityDescriptor::new(
                    CapabilityId("pc.artifact.read".into()),
                    1,
                    CapabilityDomain::Pc,
                    SideEffectClass::ReadOnly,
                );
                if let Ok(descriptor) = descriptor.as_mut() {
                    descriptor
                        .required_scopes
                        .push(AuthorityScope::new("pc.artifact.read").map_err(|error| {
                            format!("paired-PC artifact read scope: {error:?}")
                        })?);
                }
                descriptor
            }
            "pc.artifact.write" => {
                let mut descriptor = CapabilityDescriptor::new(
                    CapabilityId("pc.artifact.write".into()),
                    1,
                    CapabilityDomain::Pc,
                    SideEffectClass::ExternalWrite,
                );
                if let Ok(descriptor) = descriptor.as_mut() {
                    descriptor.required_scopes.push(
                        AuthorityScope::new("pc.artifact.write").map_err(|error| {
                            format!("paired-PC artifact write scope: {error:?}")
                        })?,
                    );
                }
                descriptor
            }
            _ => unreachable!("unsupported capabilities rejected above"),
        }
        .map_err(|error| format!("build Android capability descriptor: {error:?}"))?;
        descriptor
            .normalize()
            .map_err(|error| format!("normalize Android capability descriptor: {error:?}"))?;
        registry
            .register(descriptor)
            .map_err(|error| format!("register Android capability: {error:?}"))?;
    }

    let restored_state = conversation
        .action_fabric_checkpoint_for_task(task_id)
        .map(|checkpoint| {
            decode_action_fabric_checkpoint(&registry, checkpoint)
                .map_err(|error| format!("decode persisted TAF97 action state: {error:?}"))
        })
        .transpose()?;
    let mut fabric = match restored_state {
        Some(state) => ActionFabric::from_state(registry, state)
            .map_err(|error| format!("restore persisted ActionFabric state: {error:?}"))?,
        None => ActionFabric::new(registry),
    };
    if fabric_capabilities.contains("device.observe") {
        fabric
            .register_adapter(
                CapabilityId("device.observe".into()),
                AndroidResourceAdapter {
                    snapshot: resources,
                },
            )
            .map_err(|error| format!("register Android resource adapter: {error:?}"))?;
    }
    if fabric_capabilities.contains("web.search") {
        fabric
            .register_adapter(CapabilityId("web.search".into()), AndroidWebSearchAdapter)
            .map_err(|error| format!("register Android WebSearch adapter: {error:?}"))?;
    }
    if fabric_capabilities.contains("web.fetch") {
        fabric
            .register_adapter(CapabilityId("web.fetch".into()), AndroidWebFetchAdapter)
            .map_err(|error| format!("register Android HTTPS adapter: {error:?}"))?;
    }
    if fabric_capabilities.contains("browser.observe") {
        fabric
            .register_adapter(
                CapabilityId("browser.observe".into()),
                AndroidBrowserObserveAdapter,
            )
            .map_err(|error| format!("register Android browser observe adapter: {error:?}"))?;
    }
    if fabric_capabilities.contains("browser.interact") {
        fabric
            .register_adapter(
                CapabilityId("browser.interact".into()),
                AndroidBrowserInteractAdapter,
            )
            .map_err(|error| format!("register Android browser interaction adapter: {error:?}"))?;
    }
    if fabric_capabilities.contains("file.grant.read") {
        fabric
            .register_adapter(
                CapabilityId("file.grant.read".into()),
                AndroidUserGrantedFileAdapter,
            )
            .map_err(|error| format!("register user-granted file.read adapter: {error:?}"))?;
    }
    if fabric_capabilities.contains("file.grant.write") {
        fabric
            .register_adapter(
                CapabilityId("file.grant.write".into()),
                AndroidUserGrantedFileAdapter,
            )
            .map_err(|error| format!("register user-granted file.write adapter: {error:?}"))?;
    }
    if fabric_capabilities.contains("file.read") || fabric_capabilities.contains("file.write") {
        let adapter = AndroidScopedFileAdapter::new(model.capability_root.clone())?;
        if fabric_capabilities.contains("file.read") {
            fabric
                .register_adapter(CapabilityId("file.read".into()), adapter.clone())
                .map_err(|error| format!("register app-private file.read adapter: {error:?}"))?;
        }
        if fabric_capabilities.contains("file.write") {
            fabric
                .register_adapter(CapabilityId("file.write".into()), adapter)
                .map_err(|error| format!("register app-private file.write adapter: {error:?}"))?;
        }
    }
    if fabric_capabilities.contains("artifact.download") {
        fabric
            .register_adapter(
                CapabilityId("artifact.download".into()),
                AndroidArtifactDownloadAdapter::new(model.capability_root.clone())?,
            )
            .map_err(|error| format!("register artifact download adapter: {error:?}"))?;
    }
    if fabric_capabilities.contains("artifact.upload") {
        fabric
            .register_adapter(
                CapabilityId("artifact.upload".into()),
                AndroidArtifactUploadAdapter::new(model.capability_root.clone())?,
            )
            .map_err(|error| format!("register artifact upload adapter: {error:?}"))?;
    }
    if fabric_capabilities.contains("device.interact") {
        fabric
            .register_adapter(
                CapabilityId("device.interact".into()),
                AndroidDeviceInteractAdapter,
            )
            .map_err(|error| format!("register device interaction adapter: {error:?}"))?;
    }
    if fabric_capabilities.contains("app.action") {
        fabric
            .register_adapter(CapabilityId("app.action".into()), AndroidAppActionAdapter)
            .map_err(|error| format!("register app action adapter: {error:?}"))?;
    }
    for capability in [
        "pc.observe",
        "pc.execute",
        "pc.artifact.read",
        "pc.artifact.write",
    ] {
        if fabric_capabilities.contains(capability) {
            let peer = pc_peer_for_capability(action_plan, capability)?;
            let adapter =
                AndroidPairedPcAdapter::connect(&model.capability_root, &peer, capability)?;
            fabric
                .register_adapter(CapabilityId(capability.into()), adapter)
                .map_err(|error| format!("register paired-PC adapter {capability}: {error:?}"))?;
        }
    }

    let mut authority = AuthorityGrant::new()
        .with_scope(
            AuthorityScope::new("network.read")
                .map_err(|error| format!("network authority scope: {error:?}"))?,
        )
        .with_scope(
            AuthorityScope::new("file.app_private")
                .map_err(|error| format!("file authority scope: {error:?}"))?,
        );
    if fabric_capabilities.contains("browser.observe") {
        authority = authority.with_scope(
            AuthorityScope::new("browser.observe")
                .map_err(|error| format!("browser observe authority scope: {error:?}"))?,
        );
    }
    if fabric_capabilities.contains("file.grant.read") {
        authority = authority.with_scope(
            AuthorityScope::new("file.user_grant.read")
                .map_err(|error| format!("granted file read authority scope: {error:?}"))?,
        );
    }
    if fabric_capabilities.contains("pc.observe") {
        authority = authority.with_scope(
            AuthorityScope::new("pc.observe")
                .map_err(|error| format!("paired-PC observe authority scope: {error:?}"))?,
        );
    }
    if fabric_capabilities.contains("pc.artifact.read") {
        authority = authority.with_scope(
            AuthorityScope::new("pc.artifact.read")
                .map_err(|error| format!("paired-PC artifact read authority scope: {error:?}"))?,
        );
    }
    if action_plan
        .graph
        .actions
        .iter()
        .any(|node| node.side_effect == SideEffectClass::ExternalWrite)
    {
        if !external_approved {
            return Err("external-write action requires explicit chat approval".into());
        }
        authority.allow_external_write = true;
        if fabric_capabilities.contains("artifact.upload") {
            authority = authority.with_scope(
                AuthorityScope::new("network.write")
                    .map_err(|error| format!("network write authority scope: {error:?}"))?,
            );
        }
        if fabric_capabilities.contains("file.grant.write") {
            authority = authority.with_scope(
                AuthorityScope::new("file.user_grant.write")
                    .map_err(|error| format!("granted file write authority scope: {error:?}"))?,
            );
        }
        if fabric_capabilities.contains("browser.interact") {
            authority = authority.with_scope(
                AuthorityScope::new("browser.interact")
                    .map_err(|error| format!("browser interaction authority scope: {error:?}"))?,
            );
        }
        if fabric_capabilities.contains("device.interact") {
            authority = authority.with_scope(
                AuthorityScope::new("device.clipboard.write")
                    .map_err(|error| format!("clipboard authority scope: {error:?}"))?,
            );
        }
        if fabric_capabilities.contains("app.action") {
            for scope in app_action_scope_names(action_plan)? {
                authority = authority.with_scope(
                    AuthorityScope::new(scope)
                        .map_err(|error| format!("app authority scope: {error:?}"))?,
                );
            }
        }
        if fabric_capabilities.contains("pc.execute") {
            authority =
                authority
                    .with_scope(AuthorityScope::new("pc.execute").map_err(|error| {
                        format!("paired-PC execute authority scope: {error:?}")
                    })?);
        }
        if fabric_capabilities.contains("pc.artifact.write") {
            authority =
                authority.with_scope(AuthorityScope::new("pc.artifact.write").map_err(
                    |error| format!("paired-PC artifact write authority scope: {error:?}"),
                )?);
        }
    }

    let task = conversation
        .cognition()
        .state()
        .tasks
        .get(&task_id)
        .cloned()
        .ok_or_else(|| "model-grounded cognitive task disappeared".to_owned())?;
    let plan_id = if conversation
        .action_fabric_checkpoint_for_task(task_id)
        .is_some()
    {
        if fabric.state().plans.len() != 1 {
            return Err("persisted ActionFabric must contain exactly one chat plan".into());
        }
        let plan = fabric
            .state()
            .plans
            .values()
            .next()
            .ok_or_else(|| "persisted ActionFabric plan is missing".to_owned())?;
        if plan.task_id != task_id {
            return Err("persisted ActionFabric task identity mismatch".into());
        }
        plan.id
    } else {
        fabric
            .prepare_cognitive_task(&task, action_plan.payloads.clone())
            .map_err(|error| format!("prepare Android ActionFabric plan: {error:?}"))?
    };

    let persist_fabric = |conversation: &mut SovereignConversationState,
                          fabric: &ActionFabric|
     -> Result<(), String> {
        let checkpoint = encode_action_fabric_checkpoint(fabric.registry(), fabric.state())
            .map_err(|error| format!("encode TAF97 action state: {error:?}"))?;
        conversation
            .install_action_fabric_checkpoint(task_id, checkpoint)
            .map_err(|error| format!("persist TAF97 in NCS97: {error:?}"))
    };

    persist_fabric(conversation, &fabric)?;
    if prepare_only {
        return Ok(AndroidActionExecutionOutcome::Prepared);
    }

    let run = match continue_verified_assistant_plan(
        &mut fabric,
        plan_id,
        &task,
        action_plan,
        &authority,
        &mut AndroidProductionVerifier,
        user_message,
    ) {
        Ok(run) => run,
        Err(AssistantActionRunError::Incomplete {
            plan_status,
            action_status,
            summary,
        }) if plan_status == ActionPlanStatus::Suspended
            || matches!(
                action_status,
                Some(ActionStatus::Retryable | ActionStatus::Suspended)
            ) =>
        {
            let requires_reconfirm = action_checkpoint_requires_reconfirm(&fabric, plan_id);
            persist_fabric(conversation, &fabric)?;
            return Ok(AndroidActionExecutionOutcome::Checkpointed {
                note: summary,
                requires_reconfirm,
            });
        }
        Err(error) => {
            return Err(format!("execute verified Android action plan: {error:?}"));
        }
    };

    conversation
        .clear_action_fabric_checkpoint(task_id)
        .map_err(|error| format!("clear completed TAF97 action state: {error:?}"))?;

    let synthesis_tokens = model
        .tokenizer
        .encode(&run.synthesis_prompt, true)
        .map_err(|error| format!("encode verified synthesis prompt: {error:?}"))?;
    if synthesis_tokens.is_empty() || synthesis_tokens.len() > prompt_limit {
        return Err("verified synthesis prompt exceeds native context budget".into());
    }

    conversation
        .replace_active_prompt_tokens(task_id, synthesis_tokens)
        .map_err(|error| format!("install verified synthesis prompt: {error:?}"))?;
    conversation
        .record_verified_synthesis_ready(task_id)
        .map_err(|error| format!("record verified synthesis provenance: {error:?}"))?;
    conversation
        .record_verified_action_count(task_id, run.evidence.len())
        .map_err(|error| format!("record verified action count: {error:?}"))?;
    conversation
        .record_action_planner_status(task_id, "actions")
        .map_err(|error| format!("record action planner status: {error:?}"))?;
    Ok(AndroidActionExecutionOutcome::Completed)
}

fn submit_chat_reserved(
    model: &Arc<NativeChatModel>,
    history: &[ntd_runtime::ConversationTurn],
    prior_failure: bool,
    resources: ResourceSnapshot,
    user_message: &str,
    max_new_tokens: usize,
) -> Result<u64, String> {
    let prompt_limit = model.context_limit.saturating_sub(max_new_tokens).max(1);
    let base = NativeChatPromptCompiler
        .compile(history, user_message, &model.tokenizer, prompt_limit)
        .map_err(|error| format!("compile preflight chat prompt: {error:?}"))?;

    let generator = GraphGenerator::new(
        model.activation.graph.clone(),
        CpuReferenceProvider,
        BTreeMap::new(),
        model.activation.manifest.token_input,
        usize::try_from(model.activation.manifest.distribution_output)
            .map_err(|_| "distribution output does not fit usize".to_owned())?,
        usize::try_from(model.activation.manifest.vocabulary_size)
            .map_err(|_| "vocabulary size does not fit usize".to_owned())?,
    )
    .map_err(|error| format!("build reasoning preflight generator: {error:?}"))?;
    let distribution = generator
        .next_distribution_with_resolver(&model.resolver, &base.token_ids, model.context_limit)
        .map_err(|error| format!("native reasoning preflight: {error:?}"))?;
    let inference = model_inference_signals(&distribution)
        .map_err(|error| format!("analyze native reasoning logits: {error:?}"))?;
    let signals =
        inference.cognitive_signals(base.token_ids.len(), model.context_limit, prior_failure);
    let budget = choose_reasoning_budget(signals);
    let recall_limit = memory_recall_limit_for_budget(budget);

    let conversation_snapshot = {
        let guard = lock_state();
        if guard
            .chat_session
            .as_ref()
            .is_some_and(|session| chat_session_active(session.status))
        {
            return Err("another native chat request started during preflight".into());
        }
        let loaded_model = guard
            .chat_model
            .as_ref()
            .ok_or_else(|| "native chat model was unloaded during preflight".to_owned())?;
        if loaded_model.asset_id != model.asset_id || loaded_model.version != model.version {
            return Err("native chat model changed during reasoning preflight".into());
        }
        guard.conversation.clone()
    };

    let mut conversation = conversation_snapshot.clone();
    let recalled = conversation
        .recall_conversation_context(user_message, recall_limit)
        .map_err(|error| format!("recall sovereign conversation memory: {error:?}"))?;
    let compiled = NativeChatPromptCompiler
        .compile_with_memory(
            conversation.turns(),
            &recalled,
            user_message,
            &model.tokenizer,
            prompt_limit,
        )
        .map_err(|error| format!("compile memory-augmented chat prompt: {error:?}"))?;

    let reasoning_prompt_tokens = compiled.token_ids.clone();
    let task_id = conversation
        .begin_turn(
            model.asset_id.clone(),
            model.version,
            user_message,
            compiled.token_ids,
            max_new_tokens,
        )
        .map_err(|error| format!("begin sovereign conversation turn: {error:?}"))?;
    conversation
        .record_reasoning_profile(task_id, budget, signals, compiled.retained_memory_items)
        .map_err(|error| format!("record sovereign reasoning profile: {error:?}"))?;

    let mut probe = NativeModelReasoningProbe::new(model, reasoning_prompt_tokens)?;
    let report =
        run_budgeted_reasoning_cycle(conversation.cognition_mut(), task_id, signals, &mut probe)
            .map_err(|error| format!("run budgeted native reasoning cycle: {error:?}"))?;
    if report.budget != budget || report.status != TaskStatus::Running {
        return Err(format!(
            "native reasoning cycle did not remain runnable: budget={:?} status={:?}",
            report.budget, report.status
        ));
    }
    conversation
        .record_reasoning_cycle_report(task_id, &report)
        .map_err(|error| format!("record reasoning cycle evidence: {error:?}"))?;

    let planner_outcome = if let Some(plan) = governed_explicit_action_plan(user_message)? {
        NativeActionPlanningOutcome::Actions(plan)
    } else {
        match governed_device_surfaces(user_message) {
            Some(surfaces) => run_constrained_device_planner(model, user_message, &surfaces)?,
            None => run_native_action_planner(model, user_message)?,
        }
    };
    let mut session_status = NativeChatSessionStatus::Running;
    match planner_outcome {
        NativeActionPlanningOutcome::Direct => {
            conversation
                .record_action_planner_status(task_id, "direct")
                .map_err(|error| format!("record direct planner status: {error:?}"))?;
            conversation
                .record_verified_action_count(task_id, 0)
                .map_err(|error| format!("record direct verified action count: {error:?}"))?;
        }
        NativeActionPlanningOutcome::Invalid => {
            conversation
                .record_action_planner_status(task_id, "invalid")
                .map_err(|error| format!("record invalid planner status: {error:?}"))?;
            conversation
                .record_verified_action_count(task_id, 0)
                .map_err(|error| format!("record invalid verified action count: {error:?}"))?;
        }
        NativeActionPlanningOutcome::Actions(action_plan) => {
            conversation
                .install_action_plan(task_id, &action_plan)
                .map_err(|error| format!("install model-grounded action plan: {error:?}"))?;
            if let Some((capability, rationale)) = external_write_approval(&action_plan)? {
                conversation
                    .record_chat_approval(task_id, &capability, &rationale, "pending")
                    .map_err(|error| format!("record pending chat approval: {error:?}"))?;
                conversation
                    .record_action_planner_status(task_id, "actions")
                    .map_err(|error| format!("record action planner status: {error:?}"))?;
                conversation
                    .record_verified_action_count(task_id, 0)
                    .map_err(|error| format!("record pending verified action count: {error:?}"))?;
                session_status = NativeChatSessionStatus::WaitingApproval;
            } else {
                execute_android_verified_actions(
                    &mut conversation,
                    &action_plan,
                    AndroidActionExecutionContext {
                        model,
                        task_id,
                        user_message,
                        prompt_limit,
                        resources,
                        external_approved: false,
                        prepare_only: false,
                    },
                )?;
            }
        }
    }

    let mut guard = lock_state();
    let loaded_model = guard
        .chat_model
        .as_ref()
        .ok_or_else(|| "native chat model was unloaded before reasoning commit".to_owned())?;
    if loaded_model.asset_id != model.asset_id || loaded_model.version != model.version {
        return Err("native chat model changed before reasoning commit".into());
    }
    if guard.conversation != conversation_snapshot
        || guard
            .chat_session
            .as_ref()
            .is_some_and(|session| chat_session_active(session.status))
    {
        return Err("sovereign conversation changed during native reasoning".into());
    }

    let request_id = guard.next_chat_request_id;
    guard.next_chat_request_id = guard
        .next_chat_request_id
        .checked_add(1)
        .ok_or_else(|| "chat request id overflow".to_owned())?;
    guard.conversation = conversation;
    guard.chat_session = Some(NativeChatSession {
        request_id,
        task_id,
        cancel: Arc::new(AtomicBool::new(false)),
        status: session_status,
    });
    Ok(request_id)
}

fn resolve_chat_approval(request_id: u64, approved: bool) -> Result<bool, String> {
    let (task_id, model, resources, mut conversation) = {
        let guard = lock_state();
        let session = guard
            .chat_session
            .as_ref()
            .filter(|session| {
                session.request_id == request_id
                    && session.status == NativeChatSessionStatus::WaitingApproval
            })
            .ok_or_else(|| "chat request is not waiting for approval".to_owned())?;
        let model = guard
            .chat_model
            .clone()
            .ok_or_else(|| "native chat model is not loaded".to_owned())?;
        (
            session.task_id,
            model,
            guard.resources,
            guard.conversation.clone(),
        )
    };

    let (_, _, approval_status) = conversation
        .chat_approval_for_task(task_id)
        .ok_or_else(|| "pending chat approval state is missing".to_owned())?;
    if !matches!(
        approval_status,
        "pending" | "reconfirm" | "approved-executing"
    ) {
        return Err("chat approval is no longer pending".into());
    }

    if !approved {
        conversation
            .set_chat_approval_status(task_id, "denied")
            .map_err(|error| format!("record denied chat approval: {error:?}"))?;
        conversation
            .cancel_turn(task_id, "user denied external-write approval")
            .map_err(|error| format!("cancel denied chat task: {error:?}"))?;

        let mut guard = lock_state();
        if !guard
            .chat_session
            .as_ref()
            .is_some_and(|session| session.request_id == request_id)
        {
            return Err("chat request changed during denial".into());
        }
        guard.conversation = conversation;
        if let Some(session) = guard
            .chat_session
            .as_mut()
            .filter(|session| session.request_id == request_id)
        {
            session.status = NativeChatSessionStatus::Cancelled;
        }
        return Ok(true);
    }

    conversation
        .set_chat_approval_status(task_id, "approved-executing")
        .map_err(|error| format!("record executing chat approval: {error:?}"))?;

    let active = conversation
        .active()
        .cloned()
        .ok_or_else(|| "approved chat task has no active turn".to_owned())?;
    if active.task_id != task_id {
        return Err("approved chat task identity mismatch".into());
    }
    let protocol = conversation
        .action_plan_protocol_for_task(task_id)
        .ok_or_else(|| "approved chat task has no canonical action protocol".to_owned())?
        .to_owned();
    let action_plan = match parse_native_action_plan(&protocol)
        .map_err(|error| format!("reconstruct approved action plan: {error:?}"))?
    {
        AssistantPlanDecision::Actions(plan) => plan,
        AssistantPlanDecision::Direct => {
            return Err("approved chat protocol did not contain actions".into())
        }
    };
    external_write_approval(&action_plan)?
        .ok_or_else(|| "approved chat plan no longer contains external writes".to_owned())?;

    let prompt_limit = model
        .context_limit
        .saturating_sub(active.max_new_tokens)
        .max(1);
    let prepared = execute_android_verified_actions(
        &mut conversation,
        &action_plan,
        AndroidActionExecutionContext {
            model: &model,
            task_id,
            user_message: &active.user_message,
            prompt_limit,
            resources,
            external_approved: true,
            prepare_only: true,
        },
    )?;
    if prepared != AndroidActionExecutionOutcome::Prepared {
        return Err("approved external write did not stop at durable prepare boundary".into());
    }

    let mut guard = lock_state();
    let session = guard
        .chat_session
        .as_ref()
        .filter(|session| {
            session.request_id == request_id
                && session.status == NativeChatSessionStatus::WaitingApproval
        })
        .ok_or_else(|| "chat request changed before durable action preparation".to_owned())?;
    if session.task_id != task_id {
        return Err("chat task changed before durable action preparation".into());
    }
    guard.conversation = conversation;
    if let Some(session) = guard
        .chat_session
        .as_mut()
        .filter(|session| session.request_id == request_id)
    {
        session.status = NativeChatSessionStatus::Running;
    }
    Ok(true)
}

fn chat_status(request_id: u64) -> i32 {
    let guard = lock_state();
    let Some(session) = guard
        .chat_session
        .as_ref()
        .filter(|session| session.request_id == request_id)
    else {
        return CHAT_STATUS_MISSING;
    };
    match session.status {
        NativeChatSessionStatus::Running => CHAT_STATUS_RUNNING,
        NativeChatSessionStatus::WaitingApproval => CHAT_STATUS_WAITING_APPROVAL,
        NativeChatSessionStatus::Complete => CHAT_STATUS_COMPLETE,
        NativeChatSessionStatus::Cancelled => CHAT_STATUS_CANCELLED,
        NativeChatSessionStatus::Failed => CHAT_STATUS_FAILED,
    }
}

fn chat_reasoning_budget(request_id: u64) -> i32 {
    let guard = lock_state();
    let Some(session) = guard
        .chat_session
        .as_ref()
        .filter(|session| session.request_id == request_id)
    else {
        return 0;
    };
    match guard
        .conversation
        .reasoning_budget_for_task(session.task_id)
    {
        Some(ntd_runtime::ReasoningBudget::Reflex) => 1,
        Some(ntd_runtime::ReasoningBudget::Standard) => 2,
        Some(ntd_runtime::ReasoningBudget::Deep) => 3,
        Some(ntd_runtime::ReasoningBudget::Recovery) => 4,
        None => 0,
    }
}

fn chat_recalled_memory_items(request_id: u64) -> i32 {
    let guard = lock_state();
    let Some(session) = guard
        .chat_session
        .as_ref()
        .filter(|session| session.request_id == request_id)
    else {
        return 0;
    };
    guard
        .conversation
        .recalled_memory_items_for_task(session.task_id)
        .and_then(|value| i32::try_from(value).ok())
        .unwrap_or(0)
}

fn chat_memory_record_count(request_id: u64) -> i32 {
    let guard = lock_state();
    if !guard
        .chat_session
        .as_ref()
        .is_some_and(|session| session.request_id == request_id)
    {
        return 0;
    }
    i32::try_from(guard.conversation.sovereign_memory_record_count()).unwrap_or(i32::MAX)
}

fn chat_reasoning_iterations(request_id: u64) -> i32 {
    let guard = lock_state();
    let Some(session) = guard
        .chat_session
        .as_ref()
        .filter(|session| session.request_id == request_id)
    else {
        return 0;
    };
    guard
        .conversation
        .reasoning_iterations_for_task(session.task_id)
        .and_then(|value| i32::try_from(value).ok())
        .unwrap_or(0)
}
fn chat_action_planner_status(request_id: u64) -> i32 {
    let guard = lock_state();
    let Some(session) = guard
        .chat_session
        .as_ref()
        .filter(|session| session.request_id == request_id)
    else {
        return 0;
    };
    match guard
        .conversation
        .action_planner_status_for_task(session.task_id)
    {
        Some("direct") => 1,
        Some("actions") => 2,
        Some("invalid") => 3,
        Some("unsupported") => 4,
        _ => 0,
    }
}

fn chat_action_count(request_id: u64) -> i32 {
    let guard = lock_state();
    let Some(session) = guard
        .chat_session
        .as_ref()
        .filter(|session| session.request_id == request_id)
    else {
        return 0;
    };
    guard
        .conversation
        .action_count_for_task(session.task_id)
        .and_then(|value| i32::try_from(value).ok())
        .unwrap_or(0)
}

fn chat_verified_synthesis_ready(request_id: u64) -> bool {
    let guard = lock_state();
    let Some(session) = guard
        .chat_session
        .as_ref()
        .filter(|session| session.request_id == request_id)
    else {
        return false;
    };
    guard
        .conversation
        .verified_synthesis_ready_for_task(session.task_id)
}

fn chat_verified_action_count(request_id: u64) -> i32 {
    let guard = lock_state();
    let Some(session) = guard
        .chat_session
        .as_ref()
        .filter(|session| session.request_id == request_id)
    else {
        return 0;
    };
    guard
        .conversation
        .verified_action_count_for_task(session.task_id)
        .and_then(|value| i32::try_from(value).ok())
        .unwrap_or(0)
}

fn cancel_chat(request_id: u64) -> bool {
    let (task_id, cancel) = {
        let guard = lock_state();
        let Some(session) = guard.chat_session.as_ref().filter(|session| {
            session.request_id == request_id && chat_session_active(session.status)
        }) else {
            return false;
        };
        (session.task_id, Arc::clone(&session.cancel))
    };
    cancel.store(true, Ordering::Release);

    let mut guard = lock_state();
    if guard
        .conversation
        .active()
        .is_some_and(|active| active.task_id == task_id)
        && guard
            .conversation
            .cancel_turn(task_id, "user cancelled native generation")
            .is_err()
    {
        if let Some(session) = guard
            .chat_session
            .as_mut()
            .filter(|session| session.request_id == request_id)
        {
            session.status = NativeChatSessionStatus::Failed;
        }
        return false;
    }
    if let Some(session) = guard
        .chat_session
        .as_mut()
        .filter(|session| session.request_id == request_id)
    {
        session.status = NativeChatSessionStatus::Cancelled;
    }
    true
}

fn encode_chat_event(kind: u8, token: Option<u32>, text: &str) -> Vec<u8> {
    let mut out = vec![CHAT_EVENT_PROTOCOL_VERSION, kind];
    out.extend_from_slice(&token.unwrap_or(u32::MAX).to_le_bytes());
    push_string(&mut out, text);
    out
}

fn terminal_chat_event(request_id: u64) -> Option<Vec<u8>> {
    let guard = lock_state();
    let session = guard
        .chat_session
        .as_ref()
        .filter(|session| session.request_id == request_id)?;

    match session.status {
        NativeChatSessionStatus::Running => None,
        NativeChatSessionStatus::WaitingApproval => {
            match guard.conversation.chat_approval_for_task(session.task_id) {
                Some((capability, rationale, _)) => Some(encode_chat_event(
                    CHAT_EVENT_APPROVAL_REQUIRED,
                    None,
                    &format!("{capability}\n{rationale}"),
                )),
                None => Some(encode_chat_event(
                    CHAT_EVENT_ERROR,
                    None,
                    "pending approval state is missing",
                )),
            }
        }
        NativeChatSessionStatus::Complete => Some(encode_chat_event(CHAT_EVENT_COMPLETE, None, "")),
        NativeChatSessionStatus::Cancelled => {
            Some(encode_chat_event(CHAT_EVENT_CANCELLED, None, ""))
        }
        NativeChatSessionStatus::Failed => Some(encode_chat_event(
            CHAT_EVENT_ERROR,
            None,
            "native chat generation failed",
        )),
    }
}

fn advance_pending_chat_actions(request_id: u64) -> Result<Option<Vec<u8>>, String> {
    let (task_id, model, resources, conversation) = {
        let guard = lock_state();
        let session = guard
            .chat_session
            .as_ref()
            .filter(|session| {
                session.request_id == request_id
                    && session.status == NativeChatSessionStatus::Running
            })
            .ok_or_else(|| "chat request is not running".to_owned())?;
        if guard
            .conversation
            .action_fabric_checkpoint_for_task(session.task_id)
            .is_none()
        {
            return Ok(None);
        }
        (
            session.task_id,
            guard
                .chat_model
                .clone()
                .ok_or_else(|| "native chat model is not loaded".to_owned())?,
            guard.resources,
            guard.conversation.clone(),
        )
    };

    let active = conversation
        .active()
        .cloned()
        .ok_or_else(|| "persisted action state has no active turn".to_owned())?;
    if active.task_id != task_id {
        return Err("persisted action state task identity mismatch".into());
    }
    let protocol = conversation
        .action_plan_protocol_for_task(task_id)
        .ok_or_else(|| "persisted action state has no canonical action protocol".to_owned())?;
    let action_plan = match parse_native_action_plan(protocol)
        .map_err(|error| format!("reconstruct persisted action plan: {error:?}"))?
    {
        AssistantPlanDecision::Actions(plan) => plan,
        AssistantPlanDecision::Direct => {
            return Err("persisted action protocol unexpectedly resolved to DIRECT".into())
        }
    };
    let external_write = external_write_approval(&action_plan)?.is_some();
    let approval_status = conversation
        .chat_approval_for_task(task_id)
        .map(|(_, _, status)| status);
    let external_approved = approval_status == Some("approved-executing");
    if external_write && !external_approved {
        return Err("persisted external-write plan requires reconfirmation before resume".into());
    }

    let prompt_limit = model
        .context_limit
        .saturating_sub(active.max_new_tokens)
        .max(1);
    let mut advanced = conversation.clone();
    let outcome = execute_android_verified_actions(
        &mut advanced,
        &action_plan,
        AndroidActionExecutionContext {
            model: &model,
            task_id,
            user_message: &active.user_message,
            prompt_limit,
            resources,
            external_approved,
            prepare_only: false,
        },
    )?;

    let note = match outcome {
        AndroidActionExecutionOutcome::Prepared => {
            return Err("running action execution returned to prepare-only boundary".into())
        }
        AndroidActionExecutionOutcome::Checkpointed {
            note,
            requires_reconfirm,
        } => {
            if requires_reconfirm {
                advanced
                    .set_chat_approval_status(task_id, "reconfirm")
                    .map_err(|error| {
                        format!("require reconfirmation before external retry: {error:?}")
                    })?;
            }
            note
        }
        AndroidActionExecutionOutcome::Completed => {
            if external_write {
                advanced
                    .set_chat_approval_status(task_id, "approved")
                    .map_err(|error| format!("commit approved action status: {error:?}"))?;
            }
            "verified action execution committed".to_owned()
        }
    };

    let mut guard = lock_state();
    if guard.conversation != conversation {
        return Err("sovereign conversation changed during action continuation".into());
    }
    if !guard.chat_session.as_ref().is_some_and(|session| {
        session.request_id == request_id
            && session.task_id == task_id
            && session.status == NativeChatSessionStatus::Running
    }) {
        return Err("chat request changed during action continuation".into());
    }
    guard.conversation = advanced;
    if guard
        .conversation
        .chat_approval_for_task(task_id)
        .is_some_and(|(_, _, status)| status == "reconfirm")
    {
        if let Some(session) = guard
            .chat_session
            .as_mut()
            .filter(|session| session.request_id == request_id)
        {
            session.status = NativeChatSessionStatus::WaitingApproval;
        }
    }
    Ok(Some(encode_chat_event(
        CHAT_EVENT_ACTION_CHECKPOINTED,
        None,
        &note,
    )))
}

fn next_chat_event(request_id: u64) -> Result<Vec<u8>, String> {
    if let Some(event) = terminal_chat_event(request_id) {
        return Ok(event);
    }
    if let Some(event) = advance_pending_chat_actions(request_id)? {
        return Ok(event);
    }

    let (model, task_id, all_tokens, generated_len, max_new_tokens, cancel) = {
        let guard = lock_state();
        let session = guard
            .chat_session
            .as_ref()
            .filter(|session| session.request_id == request_id)
            .ok_or_else(|| "unknown chat request".to_owned())?;
        let active = guard
            .conversation
            .active()
            .filter(|active| active.task_id == session.task_id)
            .ok_or_else(|| "active sovereign conversation turn is missing".to_owned())?;
        let model = guard
            .chat_model
            .clone()
            .ok_or_else(|| "native chat model was unloaded".to_owned())?;
        if active.model_asset_id != model.asset_id || active.model_version != model.version {
            return Err("active NCS97 model identity does not match loaded native model".into());
        }
        (
            model,
            session.task_id,
            guard
                .conversation
                .all_tokens_for_active()
                .map_err(|error| format!("restore active token prefix: {error:?}"))?,
            active.generated_tokens.len(),
            active.max_new_tokens,
            Arc::clone(&session.cancel),
        )
    };

    if generated_len >= max_new_tokens {
        let mut guard = lock_state();
        guard
            .conversation
            .complete_turn(task_id)
            .map_err(|error| format!("commit sovereign conversation turn: {error:?}"))?;
        if let Some(session) = guard
            .chat_session
            .as_mut()
            .filter(|session| session.request_id == request_id)
        {
            session.status = NativeChatSessionStatus::Complete;
        }
        return Ok(encode_chat_event(CHAT_EVENT_COMPLETE, None, ""));
    }

    if cancel.load(Ordering::Acquire) {
        cancel_chat(request_id);
        return Ok(encode_chat_event(CHAT_EVENT_CANCELLED, None, ""));
    }

    let previous_token = all_tokens.last().copied();
    let generator = GraphGenerator::new(
        model.activation.graph.clone(),
        CpuReferenceProvider,
        BTreeMap::new(),
        model.activation.manifest.token_input,
        usize::try_from(model.activation.manifest.distribution_output)
            .map_err(|_| "distribution output does not fit usize".to_owned())?,
        usize::try_from(model.activation.manifest.vocabulary_size)
            .map_err(|_| "vocabulary size does not fit usize".to_owned())?,
    )
    .map_err(|error| format!("build graph generator: {error:?}"))?;
    let generated = generator
        .generate_tokens_with_resolver(
            &model.resolver,
            &all_tokens,
            GenerationConfig {
                max_new_tokens: 1,
                context_limit: model.context_limit,
                eos_token: model.tokenizer.eos_token(),
                distribution: DistributionKind::Logits,
                sampling: SamplingMode::Greedy,
            },
        )
        .map_err(|error| format!("generate next chat token: {error:?}"))?;
    let token = generated
        .generated_tokens
        .first()
        .copied()
        .ok_or_else(|| "native chat generation returned no token".to_owned())?;
    let piece = model
        .tokenizer
        .decode_after(previous_token, &[token], true)
        .map_err(|error| format!("decode chat token: {error:?}"))?;

    let mut guard = lock_state();
    let cancelled = match guard
        .chat_session
        .as_ref()
        .filter(|session| session.request_id == request_id)
    {
        Some(session) => {
            session.status != NativeChatSessionStatus::Running
                || session.cancel.load(Ordering::Acquire)
        }
        None => true,
    };
    if cancelled {
        return Ok(encode_chat_event(CHAT_EVENT_CANCELLED, None, ""));
    }

    guard
        .conversation
        .append_generated(task_id, token, &piece)
        .map_err(|error| format!("checkpoint generated token: {error:?}"))?;
    let active = guard
        .conversation
        .active()
        .ok_or_else(|| "active sovereign conversation turn disappeared".to_owned())?;
    let reached_eos = model.tokenizer.eos_token() == Some(token);
    let complete = reached_eos || active.generated_tokens.len() >= active.max_new_tokens;
    if complete {
        guard
            .conversation
            .complete_turn(task_id)
            .map_err(|error| format!("commit sovereign conversation turn: {error:?}"))?;
        if let Some(session) = guard
            .chat_session
            .as_mut()
            .filter(|session| session.request_id == request_id)
        {
            session.status = NativeChatSessionStatus::Complete;
        }
    }

    Ok(encode_chat_event(CHAT_EVENT_TOKEN, Some(token), &piece))
}

fn checkpoint_chat() -> Result<Vec<u8>, String> {
    let guard = lock_state();
    encode_conversation_checkpoint(&guard.conversation)
        .map_err(|error| format!("encode NCS97: {error:?}"))
}

fn restore_chat_checkpoint(bytes: &[u8]) -> Result<u64, String> {
    let conversation = decode_conversation_checkpoint(bytes)
        .map_err(|error| format!("decode NCS97: {error:?}"))?;
    let active_task = conversation.active().map(|active| active.task_id);

    let mut guard = lock_state();
    if let Some(session) = guard.chat_session.as_ref() {
        session.cancel.store(true, Ordering::Release);
    }
    guard.conversation = conversation;

    let Some(task_id) = active_task else {
        guard.chat_session = None;
        return Ok(0);
    };

    let request_id = guard.next_chat_request_id;
    guard.next_chat_request_id = guard
        .next_chat_request_id
        .checked_add(1)
        .ok_or_else(|| "chat request id overflow".to_owned())?;
    let approval_status = guard
        .conversation
        .chat_approval_for_task(task_id)
        .map(|(_, _, status)| status.to_owned());
    if approval_status.as_deref() == Some("approved-executing") {
        guard
            .conversation
            .set_chat_approval_status(task_id, "reconfirm")
            .map_err(|error| {
                format!("mark interrupted external write for reconfirmation: {error:?}")
            })?;
    }
    let waiting_approval = matches!(
        approval_status.as_deref(),
        Some("pending" | "reconfirm" | "approved-executing")
    );
    guard.chat_session = Some(NativeChatSession {
        request_id,
        task_id,
        cancel: Arc::new(AtomicBool::new(false)),
        status: if waiting_approval {
            NativeChatSessionStatus::WaitingApproval
        } else {
            NativeChatSessionStatus::Running
        },
    });
    Ok(request_id)
}

fn chat_transcript() -> String {
    let guard = lock_state();
    let mut text = String::new();
    for turn in guard.conversation.turns() {
        match turn.role {
            ntd_runtime::ConversationRole::User => text.push_str("You: "),
            ntd_runtime::ConversationRole::Assistant => text.push_str("NTD97: "),
        }
        text.push_str(&turn.content);
        text.push('\n');
    }
    if let Some(active) = guard.conversation.active() {
        text.push_str("You: ");
        text.push_str(&active.user_message);
        text.push_str("\nNTD97: ");
        text.push_str(&active.generated_text);
    }
    text
}

#[allow(clippy::too_many_arguments)]
#[allow(unsafe_code)]
#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdNativeRuntimeHost_nativeOpenChatModel(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    asset_id: JString<'_>,
    version: jint,
    capsule_path: JString<'_>,
    shard_root: JString<'_>,
    capability_root: JString<'_>,
    verify_key: JByteArray<'_>,
    context_limit: jint,
) -> jboolean {
    let Some(asset_id) = java_string(&mut env, &asset_id) else {
        return 0;
    };
    let Some(capsule_path) = java_string(&mut env, &capsule_path) else {
        return 0;
    };
    let Some(shard_root) = java_string(&mut env, &shard_root) else {
        return 0;
    };
    let Some(capability_root) = java_string(&mut env, &capability_root) else {
        return 0;
    };
    if JAVA_VM.get().is_none() {
        let Ok(vm) = env.get_java_vm() else {
            return 0;
        };
        let _ = JAVA_VM.set(vm);
    }
    let Ok(version) = u32::try_from(version) else {
        return 0;
    };
    let Ok(context_limit) = usize::try_from(context_limit) else {
        return 0;
    };
    let verify_key = match env.convert_byte_array(&verify_key) {
        Ok(bytes) => bytes,
        Err(_) => return 0,
    };

    match open_chat_model(
        &asset_id,
        version,
        &capsule_path,
        &shard_root,
        &verify_key,
        context_limit,
        &capability_root,
    ) {
        Ok(()) => {
            lock_state().chat_last_error.clear();
            1
        }
        Err(error) => {
            let mut guard = lock_state();
            guard.chat_last_error = error.chars().take(2048).collect();
            0
        }
    }
}

#[allow(unsafe_code)]
#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdNativeRuntimeHost_nativeSubmitChat(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    prompt: JString<'_>,
    max_new_tokens: jint,
) -> jlong {
    let Some(prompt) = java_string(&mut env, &prompt) else {
        return -1;
    };
    let Ok(max_new_tokens) = usize::try_from(max_new_tokens) else {
        return -1;
    };
    match submit_chat(&prompt, max_new_tokens) {
        Ok(request_id) => {
            lock_state().chat_last_error.clear();
            i64::try_from(request_id).unwrap_or(-1)
        }
        Err(error) => {
            let mut guard = lock_state();
            guard.chat_last_error = error.chars().take(2048).collect();
            -1
        }
    }
}

#[allow(unsafe_code)]
#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdNativeRuntimeHost_nativeNextChatEvent(
    env: JNIEnv<'_>,
    _class: JClass<'_>,
    request_id: jlong,
) -> jbyteArray {
    let Ok(request_id) = u64::try_from(request_id) else {
        return java_bytes(
            &env,
            &encode_chat_event(CHAT_EVENT_ERROR, None, "invalid chat request id"),
        );
    };

    match next_chat_event(request_id) {
        Ok(event) => java_bytes(&env, &event),
        Err(error) => {
            {
                let mut guard = lock_state();
                if let Some(session) = guard
                    .chat_session
                    .as_mut()
                    .filter(|session| session.request_id == request_id)
                {
                    session.status = NativeChatSessionStatus::Failed;
                }
            }
            java_bytes(&env, &encode_chat_event(CHAT_EVENT_ERROR, None, &error))
        }
    }
}

#[allow(unsafe_code)]
#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdNativeRuntimeHost_nativeChatReasoningBudget(
    _env: JNIEnv<'_>,
    _class: JClass<'_>,
    request_id: jlong,
) -> jint {
    let Ok(request_id) = u64::try_from(request_id) else {
        return 0;
    };
    chat_reasoning_budget(request_id)
}

#[allow(unsafe_code)]
#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdNativeRuntimeHost_nativeChatRecalledMemoryItems(
    _env: JNIEnv<'_>,
    _class: JClass<'_>,
    request_id: jlong,
) -> jint {
    let Ok(request_id) = u64::try_from(request_id) else {
        return 0;
    };
    chat_recalled_memory_items(request_id)
}

#[allow(unsafe_code)]
#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdNativeRuntimeHost_nativeChatMemoryRecordCount(
    _env: JNIEnv<'_>,
    _class: JClass<'_>,
    request_id: jlong,
) -> jint {
    let Ok(request_id) = u64::try_from(request_id) else {
        return 0;
    };
    chat_memory_record_count(request_id)
}

#[allow(unsafe_code)]
#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdNativeRuntimeHost_nativeChatReasoningIterations(
    _env: JNIEnv<'_>,
    _class: JClass<'_>,
    request_id: jlong,
) -> jint {
    let Ok(request_id) = u64::try_from(request_id) else {
        return 0;
    };
    chat_reasoning_iterations(request_id)
}

#[allow(unsafe_code)]
#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdNativeRuntimeHost_nativeChatActionPlannerStatus(
    _env: JNIEnv<'_>,
    _class: JClass<'_>,
    request_id: jlong,
) -> jint {
    let Ok(request_id) = u64::try_from(request_id) else {
        return 0;
    };
    chat_action_planner_status(request_id)
}

#[allow(unsafe_code)]
#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdNativeRuntimeHost_nativeChatActionCount(
    _env: JNIEnv<'_>,
    _class: JClass<'_>,
    request_id: jlong,
) -> jint {
    let Ok(request_id) = u64::try_from(request_id) else {
        return 0;
    };
    chat_action_count(request_id)
}

#[allow(unsafe_code)]
#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdNativeRuntimeHost_nativeChatVerifiedActionCount(
    _env: JNIEnv<'_>,
    _class: JClass<'_>,
    request_id: jlong,
) -> jint {
    let Ok(request_id) = u64::try_from(request_id) else {
        return 0;
    };
    chat_verified_action_count(request_id)
}

#[allow(unsafe_code)]
#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdNativeRuntimeHost_nativeChatVerifiedSynthesisReady(
    _env: JNIEnv<'_>,
    _class: JClass<'_>,
    request_id: jlong,
) -> jboolean {
    let Ok(request_id) = u64::try_from(request_id) else {
        return 0;
    };
    u8::from(chat_verified_synthesis_ready(request_id))
}

#[allow(unsafe_code)]
#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdNativeRuntimeHost_nativeResolveChatApproval(
    _env: JNIEnv<'_>,
    _class: JClass<'_>,
    request_id: jlong,
    approved: jboolean,
) -> jboolean {
    let Ok(request_id) = u64::try_from(request_id) else {
        return 0;
    };
    match resolve_chat_approval(request_id, approved != 0) {
        Ok(value) => {
            lock_state().chat_last_error.clear();
            u8::from(value)
        }
        Err(error) => {
            let mut guard = lock_state();
            guard.chat_last_error = error.chars().take(2048).collect();
            0
        }
    }
}

#[allow(unsafe_code)]
#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdNativeRuntimeHost_nativeChatLastError(
    env: JNIEnv<'_>,
    _class: JClass<'_>,
) -> jbyteArray {
    let error = lock_state().chat_last_error.clone();
    java_bytes(&env, error.as_bytes())
}

#[allow(unsafe_code)]
#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdNativeRuntimeHost_nativeCancelChat(
    _env: JNIEnv<'_>,
    _class: JClass<'_>,
    request_id: jlong,
) -> jboolean {
    let Ok(request_id) = u64::try_from(request_id) else {
        return 0;
    };
    u8::from(cancel_chat(request_id))
}

#[allow(unsafe_code)]
#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdNativeRuntimeHost_nativeChatStatus(
    _env: JNIEnv<'_>,
    _class: JClass<'_>,
    request_id: jlong,
) -> jint {
    let Ok(request_id) = u64::try_from(request_id) else {
        return CHAT_STATUS_MISSING;
    };
    chat_status(request_id)
}

#[allow(unsafe_code)]
#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdNativeRuntimeHost_nativeChatCheckpoint(
    env: JNIEnv<'_>,
    _class: JClass<'_>,
) -> jbyteArray {
    match checkpoint_chat() {
        Ok(bytes) => java_bytes(&env, &bytes),
        Err(_) => java_bytes(&env, &[]),
    }
}

#[allow(unsafe_code)]
#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdNativeRuntimeHost_nativeRestoreChatCheckpoint(
    env: JNIEnv<'_>,
    _class: JClass<'_>,
    checkpoint: JByteArray<'_>,
) -> jlong {
    let bytes = match env.convert_byte_array(&checkpoint) {
        Ok(bytes) => bytes,
        Err(_) => return -1,
    };
    restore_chat_checkpoint(&bytes)
        .ok()
        .and_then(|request_id| i64::try_from(request_id).ok())
        .unwrap_or(-1)
}

#[allow(unsafe_code)]
#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdNativeRuntimeHost_nativeChatTranscript(
    env: JNIEnv<'_>,
    _class: JClass<'_>,
) -> jbyteArray {
    java_bytes(&env, chat_transcript().as_bytes())
}

fn production_capability_probe(
    capability_root: &str,
    package_name: &str,
) -> Result<String, String> {
    let root = PathBuf::from(capability_root);
    let mut file = AndroidScopedFileAdapter::new(root.clone())?;
    let mut verifier = AndroidProductionVerifier;

    let mut web = AndroidWebFetchAdapter;
    let web_action = TypedAction::WebFetch {
        url: "https://example.com/".into(),
    };
    let web_result = web
        .execute(ntd_runtime::ActionId(90), &web_action)
        .map_err(|error| format!("production web.fetch probe: {error}"))?;
    let AdapterResult::Completed {
        output: web_output, ..
    } = web_result
    else {
        return Err("production web.fetch probe did not complete".into());
    };
    let web_descriptor = CapabilityDescriptor::new(
        CapabilityId("web.fetch".into()),
        1,
        CapabilityDomain::Web,
        SideEffectClass::ReadOnly,
    )
    .map_err(|error| format!("web.fetch probe descriptor: {error:?}"))?;
    if verifier.verify(&web_descriptor, &web_action, &web_output) != ActionVerification::Accept {
        return Err("production web.fetch evidence verification failed".into());
    }

    let private_blocked = web
        .execute(
            ntd_runtime::ActionId(91),
            &TypedAction::WebFetch {
                url: "https://127.0.0.1/".into(),
            },
        )
        .is_err();
    if !private_blocked {
        return Err("private-network web.fetch was not rejected".into());
    }

    let search_action = TypedAction::WebSearch {
        query: "android".into(),
        max_results: 5,
    };
    let mut search_descriptor = CapabilityDescriptor::new(
        CapabilityId("web.search".into()),
        1,
        CapabilityDomain::Web,
        SideEffectClass::ReadOnly,
    )
    .map_err(|error| format!("web.search descriptor: {error:?}"))?;
    let search_network_scope = AuthorityScope::new("network.read")
        .map_err(|error| format!("web.search network scope: {error:?}"))?;
    search_descriptor
        .required_scopes
        .push(search_network_scope.clone());
    search_descriptor
        .normalize()
        .map_err(|error| format!("normalize web.search descriptor: {error:?}"))?;
    if AuthorityGrant::new().permits(&search_descriptor).is_ok() {
        return Err("web.search was not denied without network authority".into());
    }
    AuthorityGrant::new()
        .with_scope(search_network_scope)
        .permits(&search_descriptor)
        .map_err(|error| format!("web.search explicit authority rejected: {error:?}"))?;
    let mut search = AndroidWebSearchAdapter;
    let search_result = search
        .execute(ntd_runtime::ActionId(97), &search_action)
        .map_err(|error| format!("production web.search probe: {error}"))?;
    let AdapterResult::Completed {
        output: search_output,
        ..
    } = search_result
    else {
        return Err("production web.search probe did not complete".into());
    };
    if verifier.verify(&search_descriptor, &search_action, &search_output)
        != ActionVerification::Accept
    {
        return Err("production web.search evidence verification failed".into());
    }
    let ActionValue::TextList(search_items) = &search_output.value else {
        return Err("production web.search did not return normalized TextList".into());
    };
    if search_items.is_empty()
        || search_items.len() > 5
        || search_items
            .iter()
            .any(|item| !normalized_search_item_valid(item))
    {
        return Err("production web.search normalized result set is invalid".into());
    }
    if search_output.evidence.iter().any(|item| {
        item.contains("?q=") || item.contains("per_page=") || item.contains("search/repositories")
    }) {
        return Err("production web.search leaked endpoint path/query into evidence".into());
    }

    let browser_observe_action = TypedAction::BrowserObserve {
        target: "https://example.com/".into(),
    };
    let mut browser_observe_descriptor = CapabilityDescriptor::new(
        CapabilityId("browser.observe".into()),
        1,
        CapabilityDomain::Browser,
        SideEffectClass::ReadOnly,
    )
    .map_err(|error| format!("browser.observe descriptor: {error:?}"))?;
    let browser_network_scope = AuthorityScope::new("network.read")
        .map_err(|error| format!("browser network scope: {error:?}"))?;
    let browser_observe_scope = AuthorityScope::new("browser.observe")
        .map_err(|error| format!("browser observe scope: {error:?}"))?;
    browser_observe_descriptor
        .required_scopes
        .extend([browser_network_scope.clone(), browser_observe_scope.clone()]);
    browser_observe_descriptor
        .normalize()
        .map_err(|error| format!("normalize browser.observe descriptor: {error:?}"))?;
    if AuthorityGrant::new()
        .with_scope(browser_network_scope.clone())
        .permits(&browser_observe_descriptor)
        .is_ok()
    {
        return Err("browser observe was not denied without browser scope".into());
    }
    AuthorityGrant::new()
        .with_scope(browser_network_scope.clone())
        .with_scope(browser_observe_scope)
        .permits(&browser_observe_descriptor)
        .map_err(|error| format!("browser observe explicit authority rejected: {error:?}"))?;
    let mut browser_observe = AndroidBrowserObserveAdapter;
    let browser_observe_result = browser_observe
        .execute(ntd_runtime::ActionId(98), &browser_observe_action)
        .map_err(|error| format!("production browser.observe probe: {error}"))?;
    let AdapterResult::Completed {
        output: browser_observe_output,
        ..
    } = browser_observe_result
    else {
        return Err("production browser.observe probe did not complete".into());
    };
    if verifier.verify(
        &browser_observe_descriptor,
        &browser_observe_action,
        &browser_observe_output,
    ) != ActionVerification::Accept
    {
        return Err("production browser.observe evidence verification failed".into());
    }

    let browser_private_blocked = browser_observe
        .execute(
            ntd_runtime::ActionId(99),
            &TypedAction::BrowserObserve {
                target: "https://127.0.0.1/".into(),
            },
        )
        .is_err();
    if !browser_private_blocked {
        return Err("private-network browser target was not rejected".into());
    }

    let browser_interact_scope = AuthorityScope::new("browser.interact")
        .map_err(|error| format!("browser interaction scope: {error:?}"))?;
    let mut browser_interact_descriptor = CapabilityDescriptor::new(
        CapabilityId("browser.interact".into()),
        1,
        CapabilityDomain::Browser,
        SideEffectClass::ExternalWrite,
    )
    .map_err(|error| format!("browser.interact descriptor: {error:?}"))?;
    browser_interact_descriptor
        .required_scopes
        .extend([browser_network_scope, browser_interact_scope.clone()]);
    browser_interact_descriptor
        .normalize()
        .map_err(|error| format!("normalize browser.interact descriptor: {error:?}"))?;
    let browser_interact_network_scope = AuthorityScope::new("network.read")
        .map_err(|error| format!("browser interaction network scope: {error:?}"))?;
    if AuthorityGrant::new()
        .with_scope(browser_interact_network_scope.clone())
        .with_scope(browser_interact_scope.clone())
        .permits(&browser_interact_descriptor)
        .is_ok()
    {
        return Err("browser interaction was not denied without external-write authority".into());
    }
    let mut browser_interact_authority = AuthorityGrant::new()
        .with_scope(browser_interact_network_scope)
        .with_scope(browser_interact_scope);
    browser_interact_authority.allow_external_write = true;
    browser_interact_authority
        .permits(&browser_interact_descriptor)
        .map_err(|error| format!("browser interaction explicit authority rejected: {error:?}"))?;
    let mut browser_interact = AndroidBrowserInteractAdapter;
    let ambiguous_selector_blocked = browser_interact
        .execute(
            ntd_runtime::ActionId(100),
            &TypedAction::BrowserInteract {
                target: "*".into(),
                operation: "click".into(),
                value: None,
            },
        )
        .is_err();
    if !ambiguous_selector_blocked {
        return Err("ambiguous browser selector did not fail closed".into());
    }

    android_browser_drop_in_memory_session_for_test()?;
    let lost_session_blocked = browser_interact
        .execute(
            ntd_runtime::ActionId(101),
            &TypedAction::BrowserInteract {
                target: "body".into(),
                operation: "click".into(),
                value: None,
            },
        )
        .is_err();
    if !lost_session_blocked {
        return Err("browser interaction did not fail after in-memory session loss".into());
    }

    let browser_resume_action = TypedAction::BrowserObserve {
        target: "session".into(),
    };
    let browser_resume_result = browser_observe
        .execute(ntd_runtime::ActionId(102), &browser_resume_action)
        .map_err(|error| format!("production browser session resume probe: {error}"))?;
    let AdapterResult::Completed {
        output: browser_resume_output,
        ..
    } = browser_resume_result
    else {
        return Err("production browser session resume did not complete".into());
    };
    if verifier.verify(
        &browser_observe_descriptor,
        &browser_resume_action,
        &browser_resume_output,
    ) != ActionVerification::Accept
    {
        return Err("production browser session resume verification failed".into());
    }
    let ActionValue::Text(browser_resume_platform) = &browser_resume_output.value else {
        return Err("browser session resume output was not platform text".into());
    };
    if browser_platform_field(browser_resume_platform, "session") != Some("resumed") {
        return Err("browser session resume did not report resumed state".into());
    }

    let browser_interact_action = TypedAction::BrowserInteract {
        target: "body".into(),
        operation: "click".into(),
        value: None,
    };
    let browser_interact_result = browser_interact
        .execute(ntd_runtime::ActionId(103), &browser_interact_action)
        .map_err(|error| format!("production browser.interact probe: {error}"))?;
    let AdapterResult::Completed {
        output: browser_interact_output,
        ..
    } = browser_interact_result
    else {
        return Err("production browser.interact probe did not complete".into());
    };
    if verifier.verify(
        &browser_interact_descriptor,
        &browser_interact_action,
        &browser_interact_output,
    ) != ActionVerification::Accept
    {
        return Err("production browser.interact receipt verification failed".into());
    }

    let browser_navigate_action = TypedAction::BrowserInteract {
        target: "https://httpbin.org/forms/post".into(),
        operation: "navigate".into(),
        value: None,
    };
    let browser_navigate_result = browser_interact
        .execute(ntd_runtime::ActionId(104), &browser_navigate_action)
        .map_err(|error| format!("production browser.navigate probe: {error}"))?;
    let AdapterResult::Completed {
        output: browser_navigate_output,
        ..
    } = browser_navigate_result
    else {
        return Err("production browser.navigate did not complete".into());
    };
    if verifier.verify(
        &browser_interact_descriptor,
        &browser_navigate_action,
        &browser_navigate_output,
    ) != ActionVerification::Accept
    {
        return Err("production browser.navigate receipt verification failed".into());
    }

    let browser_set_value_action = TypedAction::BrowserInteract {
        target: "input[name='custname']".into(),
        operation: "set_value".into(),
        value: Some("NTD97-form-probe".into()),
    };
    let browser_set_value_result = browser_interact
        .execute(ntd_runtime::ActionId(105), &browser_set_value_action)
        .map_err(|error| format!("production browser.set_value probe: {error}"))?;
    let AdapterResult::Completed {
        output: browser_set_value_output,
        ..
    } = browser_set_value_result
    else {
        return Err("production browser.set_value did not complete".into());
    };
    if verifier.verify(
        &browser_interact_descriptor,
        &browser_set_value_action,
        &browser_set_value_output,
    ) != ActionVerification::Accept
    {
        return Err("production browser.set_value receipt verification failed".into());
    }

    let capability_root_path = Path::new(capability_root);
    let pc_profile_root = capability_root_path.join("pc-pairs");
    if pc_profile_root.exists() {
        fs::remove_dir_all(&pc_profile_root)
            .map_err(|error| format!("clear paired-PC probe profiles: {error}"))?;
    }
    if load_pc_pair_profile(capability_root_path, "workstation").is_ok() {
        return Err("unpaired PC profile did not fail closed".into());
    }

    let mut pc_execute_descriptor = CapabilityDescriptor::new(
        CapabilityId("pc.execute".into()),
        1,
        CapabilityDomain::Pc,
        SideEffectClass::ExternalWrite,
    )
    .map_err(|error| format!("pc.execute descriptor: {error:?}"))?;
    let pc_execute_scope = AuthorityScope::new("pc.execute")
        .map_err(|error| format!("pc.execute scope: {error:?}"))?;
    pc_execute_descriptor
        .required_scopes
        .push(pc_execute_scope.clone());
    pc_execute_descriptor
        .normalize()
        .map_err(|error| format!("normalize pc.execute descriptor: {error:?}"))?;
    if AuthorityGrant::new()
        .with_scope(pc_execute_scope.clone())
        .permits(&pc_execute_descriptor)
        .is_ok()
    {
        return Err("paired-PC execute was not denied without external-write authority".into());
    }
    let mut pc_execute_authority = AuthorityGrant::new().with_scope(pc_execute_scope);
    pc_execute_authority.allow_external_write = true;
    pc_execute_authority
        .permits(&pc_execute_descriptor)
        .map_err(|error| format!("paired-PC execute explicit authority rejected: {error:?}"))?;

    let relative = "m13/probe.txt";
    let write_action = TypedAction::FileWrite {
        path: relative.into(),
        bytes: b"NTD97-M13-PRODUCTION".to_vec(),
    };
    let write_result = file
        .execute(ntd_runtime::ActionId(92), &write_action)
        .map_err(|error| format!("production file.write probe: {error}"))?;
    let AdapterResult::Completed {
        output: write_output,
        rollback_token: Some(rollback_token),
    } = write_result
    else {
        return Err("production file.write probe lacked rollback token".into());
    };
    let write_descriptor = {
        let mut descriptor = CapabilityDescriptor::new(
            CapabilityId("file.write".into()),
            1,
            CapabilityDomain::File,
            SideEffectClass::Reversible,
        )
        .map_err(|error| format!("file.write probe descriptor: {error:?}"))?;
        descriptor.rollback_supported = true;
        descriptor
    };
    if verifier.verify(&write_descriptor, &write_action, &write_output)
        != ActionVerification::Accept
    {
        return Err("production file.write evidence verification failed".into());
    }

    let read_action = TypedAction::FileRead {
        path: relative.into(),
    };
    let read_result = file
        .execute(ntd_runtime::ActionId(93), &read_action)
        .map_err(|error| format!("production file.read probe: {error}"))?;
    let AdapterResult::Completed {
        output: read_output,
        ..
    } = read_result
    else {
        return Err("production file.read probe did not complete".into());
    };
    let read_descriptor = CapabilityDescriptor::new(
        CapabilityId("file.read".into()),
        1,
        CapabilityDomain::File,
        SideEffectClass::ReadOnly,
    )
    .map_err(|error| format!("file.read probe descriptor: {error:?}"))?;
    if verifier.verify(&read_descriptor, &read_action, &read_output) != ActionVerification::Accept {
        return Err("production file.read evidence verification failed".into());
    }
    if read_output.value != ActionValue::Text("NTD97-M13-PRODUCTION".into()) {
        return Err("production file.read returned unexpected content".into());
    }

    file.rollback(ntd_runtime::ActionId(92), &write_action, &rollback_token)
        .map_err(|error| format!("production file.write rollback: {error}"))?;
    let rollback_ok = file.read_text(relative).is_err();
    if !rollback_ok {
        return Err("production file.write rollback did not restore absent state".into());
    }

    let artifact_action = TypedAction::ArtifactDownload {
        url: "https://example.com/".into(),
        path: "m13/download.html".into(),
    };
    let mut artifact_descriptor = CapabilityDescriptor::new(
        CapabilityId("artifact.download".into()),
        1,
        CapabilityDomain::Web,
        SideEffectClass::Reversible,
    )
    .map_err(|error| format!("artifact.download descriptor: {error:?}"))?;
    artifact_descriptor.resumable = true;
    artifact_descriptor.rollback_supported = true;
    let network_scope = AuthorityScope::new("network.read")
        .map_err(|error| format!("artifact network authority scope: {error:?}"))?;
    let file_scope = AuthorityScope::new("file.app_private")
        .map_err(|error| format!("artifact file authority scope: {error:?}"))?;
    artifact_descriptor
        .required_scopes
        .extend([network_scope.clone(), file_scope.clone()]);
    artifact_descriptor
        .normalize()
        .map_err(|error| format!("normalize artifact.download descriptor: {error:?}"))?;
    if AuthorityGrant::new().permits(&artifact_descriptor).is_ok() {
        return Err("artifact download was not denied without explicit scopes".into());
    }
    AuthorityGrant::new()
        .with_scope(network_scope)
        .with_scope(file_scope)
        .permits(&artifact_descriptor)
        .map_err(|error| format!("artifact download explicit authority rejected: {error:?}"))?;

    let mut artifact = AndroidArtifactDownloadAdapter::new(root.clone())?;
    let artifact_result = artifact
        .execute(ntd_runtime::ActionId(96), &artifact_action)
        .map_err(|error| format!("production artifact.download stage probe: {error}"))?;
    let AdapterResult::Suspended {
        resume_token,
        note: _,
    } = artifact_result
    else {
        return Err("production artifact.download did not suspend after staging".into());
    };
    let artifact_resumed = artifact
        .resume(ntd_runtime::ActionId(96), &artifact_action, &resume_token)
        .map_err(|error| format!("production artifact.download resume probe: {error}"))?;
    let AdapterResult::Completed {
        output: artifact_output,
        rollback_token: Some(artifact_rollback),
    } = artifact_resumed
    else {
        return Err("production artifact.download resume did not complete".into());
    };
    if verifier.verify(&artifact_descriptor, &artifact_action, &artifact_output)
        != ActionVerification::Accept
    {
        return Err("production artifact.download evidence verification failed".into());
    }
    let downloaded = fs::read(root.join("m13/download.html"))
        .map_err(|error| format!("read committed artifact download: {error}"))?;
    if downloaded.is_empty() {
        return Err("artifact committed empty content".into());
    }
    let ActionValue::Fields(fields) = &artifact_output.value else {
        return Err("artifact output did not contain field evidence".into());
    };
    let expected_hash = fields
        .get("sha256")
        .ok_or_else(|| "artifact output missing sha256".to_owned())?;
    if digest_hex(&sha256(&downloaded)) != *expected_hash {
        return Err("artifact output hash does not match committed bytes".into());
    }
    artifact
        .rollback(
            ntd_runtime::ActionId(96),
            &artifact_action,
            &artifact_rollback,
        )
        .map_err(|error| format!("production artifact.download rollback: {error}"))?;
    if root.join("m13/download.html").exists() {
        return Err("artifact rollback did not restore absent state".into());
    }

    let grant_read_scope = AuthorityScope::new("file.user_grant.read")
        .map_err(|error| format!("granted file read scope: {error:?}"))?;
    let mut grant_read_descriptor = CapabilityDescriptor::new(
        CapabilityId("file.grant.read".into()),
        1,
        CapabilityDomain::File,
        SideEffectClass::ReadOnly,
    )
    .map_err(|error| format!("file.grant.read descriptor: {error:?}"))?;
    grant_read_descriptor
        .required_scopes
        .push(grant_read_scope.clone());
    grant_read_descriptor
        .normalize()
        .map_err(|error| format!("normalize file.grant.read descriptor: {error:?}"))?;
    if AuthorityGrant::new()
        .permits(&grant_read_descriptor)
        .is_ok()
    {
        return Err("user-granted file read was not denied without runtime scope".into());
    }
    AuthorityGrant::new()
        .with_scope(grant_read_scope)
        .permits(&grant_read_descriptor)
        .map_err(|error| format!("user-granted read runtime scope rejected: {error:?}"))?;
    let grant_read_action = TypedAction::FileRead {
        path: "shared\tprobe.txt".into(),
    };
    let mut user_granted_file = AndroidUserGrantedFileAdapter;
    if user_granted_file
        .execute(ntd_runtime::ActionId(101), &grant_read_action)
        .is_ok()
    {
        return Err("missing persisted SAF grant did not fail closed".into());
    }

    let grant_write_scope = AuthorityScope::new("file.user_grant.write")
        .map_err(|error| format!("granted file write scope: {error:?}"))?;
    let mut grant_write_descriptor = CapabilityDescriptor::new(
        CapabilityId("file.grant.write".into()),
        1,
        CapabilityDomain::File,
        SideEffectClass::ExternalWrite,
    )
    .map_err(|error| format!("file.grant.write descriptor: {error:?}"))?;
    grant_write_descriptor
        .required_scopes
        .push(grant_write_scope.clone());
    grant_write_descriptor
        .normalize()
        .map_err(|error| format!("normalize file.grant.write descriptor: {error:?}"))?;
    if AuthorityGrant::new()
        .with_scope(grant_write_scope.clone())
        .permits(&grant_write_descriptor)
        .is_ok()
    {
        return Err(
            "user-granted file write was not denied without external-write authority".into(),
        );
    }
    let mut grant_write_authority = AuthorityGrant::new().with_scope(grant_write_scope);
    grant_write_authority.allow_external_write = true;
    grant_write_authority
        .permits(&grant_write_descriptor)
        .map_err(|error| format!("user-granted write runtime authority rejected: {error:?}"))?;
    let grant_write_action = TypedAction::FileWrite {
        path: "shared\tprobe.txt".into(),
        bytes: b"NTD97-GRANT-PROBE".to_vec(),
    };
    if user_granted_file
        .execute(ntd_runtime::ActionId(104), &grant_write_action)
        .is_ok()
    {
        return Err("missing persisted SAF write grant did not fail closed".into());
    }

    let upload_relative = "m13/upload.bin";
    let upload_source = b"NTD97-M13-VERIFIED-UPLOAD".to_vec();
    let upload_write = TypedAction::FileWrite {
        path: upload_relative.into(),
        bytes: upload_source.clone(),
    };
    let upload_source_result = file
        .execute(ntd_runtime::ActionId(102), &upload_write)
        .map_err(|error| format!("prepare artifact upload source: {error}"))?;
    let AdapterResult::Completed {
        rollback_token: Some(upload_source_rollback),
        ..
    } = upload_source_result
    else {
        return Err("artifact upload source preparation lacked rollback token".into());
    };

    let upload_action = TypedAction::ArtifactUpload {
        url: "https://httpbin.org/put".into(),
        path: upload_relative.into(),
    };
    let mut upload_descriptor = CapabilityDescriptor::new(
        CapabilityId("artifact.upload".into()),
        1,
        CapabilityDomain::Web,
        SideEffectClass::ExternalWrite,
    )
    .map_err(|error| format!("artifact.upload descriptor: {error:?}"))?;
    upload_descriptor.resumable = true;
    let upload_network_scope = AuthorityScope::new("network.write")
        .map_err(|error| format!("upload network authority scope: {error:?}"))?;
    let upload_file_scope = AuthorityScope::new("file.app_private")
        .map_err(|error| format!("upload file authority scope: {error:?}"))?;
    upload_descriptor
        .required_scopes
        .extend([upload_network_scope.clone(), upload_file_scope.clone()]);
    upload_descriptor
        .normalize()
        .map_err(|error| format!("normalize artifact.upload descriptor: {error:?}"))?;
    if AuthorityGrant::new()
        .with_scope(upload_network_scope.clone())
        .with_scope(upload_file_scope.clone())
        .permits(&upload_descriptor)
        .is_ok()
    {
        return Err("artifact upload was not denied without external-write authority".into());
    }
    let mut upload_authority = AuthorityGrant::new()
        .with_scope(upload_network_scope)
        .with_scope(upload_file_scope);
    upload_authority.allow_external_write = true;
    upload_authority
        .permits(&upload_descriptor)
        .map_err(|error| format!("artifact upload explicit authority rejected: {error:?}"))?;

    let mut upload = AndroidArtifactUploadAdapter::new(root.clone())?;
    let upload_stage = upload
        .execute(ntd_runtime::ActionId(103), &upload_action)
        .map_err(|error| format!("production artifact.upload stage probe: {error}"))?;
    let AdapterResult::Suspended {
        resume_token: upload_resume,
        ..
    } = upload_stage
    else {
        return Err("production artifact.upload did not suspend before network write".into());
    };
    let upload_completed = upload
        .resume(ntd_runtime::ActionId(103), &upload_action, &upload_resume)
        .map_err(|error| format!("production artifact.upload resume probe: {error}"))?;
    let AdapterResult::Completed {
        output: upload_output,
        ..
    } = upload_completed
    else {
        return Err("production artifact.upload resume did not complete".into());
    };
    if verifier.verify(&upload_descriptor, &upload_action, &upload_output)
        != ActionVerification::Accept
    {
        return Err("production artifact.upload evidence verification failed".into());
    }
    let ActionValue::Fields(upload_fields) = &upload_output.value else {
        return Err("artifact upload output did not contain field evidence".into());
    };
    if upload_fields.get("sha256") != Some(&digest_hex(&sha256(&upload_source))) {
        return Err("artifact upload receipt hash does not match source bytes".into());
    }
    file.rollback(
        ntd_runtime::ActionId(102),
        &upload_write,
        &upload_source_rollback,
    )
    .map_err(|error| format!("artifact upload source rollback: {error}"))?;

    let accessibility_scope = AuthorityScope::new("app.accessibility.interact")
        .map_err(|error| format!("accessibility authority scope: {error:?}"))?;
    let mut accessibility_descriptor = CapabilityDescriptor::new(
        CapabilityId("app.action".into()),
        1,
        CapabilityDomain::App,
        SideEffectClass::ExternalWrite,
    )
    .map_err(|error| format!("accessibility app.action descriptor: {error:?}"))?;
    accessibility_descriptor
        .required_scopes
        .push(accessibility_scope.clone());
    accessibility_descriptor
        .normalize()
        .map_err(|error| format!("normalize accessibility app.action descriptor: {error:?}"))?;
    if AuthorityGrant::new()
        .with_scope(accessibility_scope.clone())
        .permits(&accessibility_descriptor)
        .is_ok()
    {
        return Err(
            "accessibility app interaction was not denied without external-write authority".into(),
        );
    }
    let mut accessibility_authority = AuthorityGrant::new().with_scope(accessibility_scope);
    accessibility_authority.allow_external_write = true;
    accessibility_authority
        .permits(&accessibility_descriptor)
        .map_err(|error| format!("accessibility explicit authority rejected: {error:?}"))?;

    let click_view_id = format!("{package_name}:id/ntd_accessibility_probe_button");
    let text_view_id = format!("{package_name}:id/ntd_accessibility_probe_text");
    let mut app_accessibility = AndroidAppActionAdapter;

    let missing_target_action = TypedAction::AppAction {
        app: package_name.to_owned(),
        action: "accessibility.click".into(),
        payload: format!("view_id\t{package_name}:id/ntd_accessibility_missing").into_bytes(),
    };
    if app_accessibility
        .execute(ntd_runtime::ActionId(104), &missing_target_action)
        .is_ok()
    {
        return Err("accessibility missing target did not fail closed".into());
    }

    let click_action = TypedAction::AppAction {
        app: package_name.to_owned(),
        action: "accessibility.click".into(),
        payload: format!("view_id\t{click_view_id}").into_bytes(),
    };
    let click_result = app_accessibility
        .execute(ntd_runtime::ActionId(105), &click_action)
        .map_err(|error| format!("production accessibility click probe: {error}"))?;
    let AdapterResult::Completed {
        output: click_output,
        ..
    } = click_result
    else {
        return Err("production accessibility click did not complete".into());
    };
    if verifier.verify(&accessibility_descriptor, &click_action, &click_output)
        != ActionVerification::Accept
    {
        return Err("production accessibility click evidence verification failed".into());
    }

    let set_text_action = TypedAction::AppAction {
        app: package_name.to_owned(),
        action: "accessibility.set_text".into(),
        payload: format!("view_id\t{text_view_id}\tNTD97-accessibility").into_bytes(),
    };
    let set_text_result = app_accessibility
        .execute(ntd_runtime::ActionId(106), &set_text_action)
        .map_err(|error| format!("production accessibility set_text probe: {error}"))?;
    let AdapterResult::Completed {
        output: set_text_output,
        ..
    } = set_text_result
    else {
        return Err("production accessibility set_text did not complete".into());
    };
    if verifier.verify(
        &accessibility_descriptor,
        &set_text_action,
        &set_text_output,
    ) != ActionVerification::Accept
    {
        return Err("production accessibility set_text evidence verification failed".into());
    }

    let clipboard_scope = AuthorityScope::new("device.clipboard.write")
        .map_err(|error| format!("clipboard authority scope: {error:?}"))?;
    let mut clipboard_descriptor = CapabilityDescriptor::new(
        CapabilityId("device.interact".into()),
        1,
        CapabilityDomain::Device,
        SideEffectClass::ExternalWrite,
    )
    .map_err(|error| format!("device.interact descriptor: {error:?}"))?;
    clipboard_descriptor
        .required_scopes
        .push(clipboard_scope.clone());
    clipboard_descriptor
        .normalize()
        .map_err(|error| format!("normalize device.interact descriptor: {error:?}"))?;
    let clipboard_action = TypedAction::DeviceInteract {
        surface: "clipboard".into(),
        operation: "set_text".into(),
        argument: Some("NTD97-M13-CLIPBOARD".into()),
    };
    let clipboard_authority_blocked = AuthorityGrant::new()
        .permits(&clipboard_descriptor)
        .is_err();
    if !clipboard_authority_blocked {
        return Err("clipboard external write was not denied by default".into());
    }
    let mut clipboard_authority = AuthorityGrant::new().with_scope(clipboard_scope);
    clipboard_authority.allow_external_write = true;
    clipboard_authority
        .permits(&clipboard_descriptor)
        .map_err(|error| format!("clipboard explicit authority rejected: {error:?}"))?;
    let mut clipboard = AndroidDeviceInteractAdapter;
    let clipboard_result = clipboard
        .execute(ntd_runtime::ActionId(94), &clipboard_action)
        .map_err(|error| format!("production clipboard probe: {error}"))?;
    let AdapterResult::Completed {
        output: clipboard_output,
        ..
    } = clipboard_result
    else {
        return Err("production clipboard probe did not complete".into());
    };
    if verifier.verify(&clipboard_descriptor, &clipboard_action, &clipboard_output)
        != ActionVerification::Accept
    {
        return Err("production clipboard evidence verification failed".into());
    }

    let app_scope = AuthorityScope::new("app.launch")
        .map_err(|error| format!("app launch authority scope: {error:?}"))?;
    let mut app_descriptor = CapabilityDescriptor::new(
        CapabilityId("app.action".into()),
        1,
        CapabilityDomain::App,
        SideEffectClass::ExternalWrite,
    )
    .map_err(|error| format!("app.action descriptor: {error:?}"))?;
    app_descriptor.required_scopes.push(app_scope.clone());
    app_descriptor
        .normalize()
        .map_err(|error| format!("normalize app.action descriptor: {error:?}"))?;
    let app_action = TypedAction::AppAction {
        app: package_name.to_owned(),
        action: "launch".into(),
        payload: Vec::new(),
    };
    let app_authority_blocked = AuthorityGrant::new().permits(&app_descriptor).is_err();
    if !app_authority_blocked {
        return Err("app launch external write was not denied by default".into());
    }
    let mut app_authority = AuthorityGrant::new().with_scope(app_scope);
    app_authority.allow_external_write = true;
    app_authority
        .permits(&app_descriptor)
        .map_err(|error| format!("app launch explicit authority rejected: {error:?}"))?;
    let mut app = AndroidAppActionAdapter;
    let app_result = app
        .execute(ntd_runtime::ActionId(95), &app_action)
        .map_err(|error| format!("production app launch probe: {error}"))?;
    let AdapterResult::Completed {
        output: app_output, ..
    } = app_result
    else {
        return Err("production app launch probe did not complete".into());
    };
    if verifier.verify(&app_descriptor, &app_action, &app_output) != ActionVerification::Accept {
        return Err("production app launch evidence verification failed".into());
    }

    Ok(
        "web_fetch=ok\nweb_private_block=ok\nweb_search_boundary=ok\nweb_search_normalized=ok\nbrowser_observe=ok\nbrowser_private_block=ok\nbrowser_interact_authority_block=ok\nbrowser_selector_ambiguity_block=ok\nbrowser_session_loss_block=ok\nbrowser_session_resume=ok\nbrowser_interact=ok\nbrowser_navigate=ok\nbrowser_set_value=ok\nfile_write=ok\nfile_read=ok\nfile_rollback=ok\nstorage_grant_runtime_scope=ok\nstorage_grant_missing_block=ok\nstorage_grant_write_authority_block=ok\nstorage_grant_write_missing_block=ok\npc_pair_missing_block=ok\npc_execute_authority_block=ok\nartifact_download_suspend=ok\nartifact_download_resume=ok\nartifact_download_rollback=ok\nartifact_upload_authority_block=ok\nartifact_upload_suspend=ok\nartifact_upload_resume=ok\nartifact_upload_receipt=ok\napp_accessibility_authority_block=ok\napp_accessibility_missing_target_block=ok\napp_accessibility_click=ok\napp_accessibility_set_text=ok\ndevice_clipboard_authority_block=ok\ndevice_clipboard_write=ok\napp_launch_authority_block=ok\napp_launch=ok\n"
            .into(),
    )
}

#[allow(unsafe_code)]
#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdNativeRuntimeHost_nativeProvisionPcPairProfile(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    capability_root: JString<'_>,
    peer: JString<'_>,
    address: JString<'_>,
    remote_peer_id: JString<'_>,
    remote_verify_key: JString<'_>,
) -> jbyteArray {
    let Some(capability_root) = java_string(&mut env, &capability_root) else {
        return java_bytes(&env, b"ERROR:invalid capability root");
    };
    let Some(peer) = java_string(&mut env, &peer) else {
        return java_bytes(&env, b"ERROR:invalid peer alias");
    };
    let Some(address) = java_string(&mut env, &address) else {
        return java_bytes(&env, b"ERROR:invalid peer address");
    };
    let Some(remote_peer_id) = java_string(&mut env, &remote_peer_id) else {
        return java_bytes(&env, b"ERROR:invalid remote peer id");
    };
    let Some(remote_verify_key) = java_string(&mut env, &remote_verify_key) else {
        return java_bytes(&env, b"ERROR:invalid remote verify key");
    };
    match provision_pc_pair_profile(
        Path::new(&capability_root),
        &peer,
        &address,
        &remote_peer_id,
        &remote_verify_key,
    ) {
        Ok(receipt) => java_bytes(&env, receipt.as_bytes()),
        Err(error) => java_bytes(&env, format!("ERROR:{error}").as_bytes()),
    }
}

#[allow(unsafe_code)]
#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdNativeRuntimeHost_nativeDescribePcPairProfile(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    capability_root: JString<'_>,
    peer: JString<'_>,
) -> jbyteArray {
    let Some(capability_root) = java_string(&mut env, &capability_root) else {
        return java_bytes(&env, b"ERROR:invalid capability root");
    };
    let Some(peer) = java_string(&mut env, &peer) else {
        return java_bytes(&env, b"ERROR:invalid peer alias");
    };
    match describe_pc_pair_profile(Path::new(&capability_root), &peer) {
        Ok(receipt) => java_bytes(&env, receipt.as_bytes()),
        Err(error) => java_bytes(&env, format!("ERROR:{error}").as_bytes()),
    }
}

#[allow(unsafe_code)]
#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdNativeRuntimeHost_nativeRevokePcPairProfile(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    capability_root: JString<'_>,
    peer: JString<'_>,
) -> jbyteArray {
    let Some(capability_root) = java_string(&mut env, &capability_root) else {
        return java_bytes(&env, b"ERROR:invalid capability root");
    };
    let Some(peer) = java_string(&mut env, &peer) else {
        return java_bytes(&env, b"ERROR:invalid peer alias");
    };
    match revoke_pc_pair_profile(Path::new(&capability_root), &peer) {
        Ok(()) => java_bytes(&env, b"OK"),
        Err(error) => java_bytes(&env, format!("ERROR:{error}").as_bytes()),
    }
}

#[allow(unsafe_code)]
#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdNativeRuntimeHost_nativeListPcPairProfiles(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    capability_root: JString<'_>,
) -> jbyteArray {
    let Some(capability_root) = java_string(&mut env, &capability_root) else {
        return java_bytes(&env, b"ERROR:invalid capability root");
    };
    match list_pc_pair_profiles(Path::new(&capability_root)) {
        Ok(peers) => java_bytes(&env, peers.join("\n").as_bytes()),
        Err(error) => java_bytes(&env, format!("ERROR:{error}").as_bytes()),
    }
}

#[allow(unsafe_code)]
#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdNativeRuntimeHost_nativeProductionCapabilityProbe(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    capability_root: JString<'_>,
    package_name: JString<'_>,
) -> jbyteArray {
    let Some(capability_root) = java_string(&mut env, &capability_root) else {
        return java_bytes(&env, b"production_capabilities=failed\n");
    };
    let Some(package_name) = java_string(&mut env, &package_name) else {
        return java_bytes(&env, b"production_capabilities=failed\n");
    };
    match production_capability_probe(&capability_root, &package_name) {
        Ok(result) => java_bytes(&env, result.as_bytes()),
        Err(error) => java_bytes(
            &env,
            format!("production_capabilities=failed\nerror={error}\n").as_bytes(),
        ),
    }
}

fn real_model_probe(
    capsule_path: &str,
    shard_root: &str,
    verify_key: &[u8],
    expected_token_ids: &str,
) -> Result<String, String> {
    let verify_key: [u8; 32] = verify_key
        .try_into()
        .map_err(|_| "verify key must contain exactly 32 bytes".to_owned())?;
    let native_capsule =
        fs::read(capsule_path).map_err(|error| format!("read native capsule: {error}"))?;
    let capsule_hash = sha256(&native_capsule);
    let package = NativePackage {
        asset_id: STORIES260K_ASSET_ID.to_owned(),
        version: STORIES260K_VERSION,
        kind: AssetKind::Intelligence,
        native_capsule,
        capsule_hash,
        capability: None,
    };
    let shard_store = FileTensorShardStore::open(shard_root)
        .map_err(|error| format!("open shards: {error:?}"))?;

    verify_native_package_with_shards(&package, &verify_key, &shard_store)
        .map_err(|error| format!("verify signed package: {error:?}"))?;
    let activation = activate_thin_generative_capsule(&package.native_capsule, &shard_store)
        .map_err(|error| format!("activate package: {error:?}"))?;

    let tokenizer = build_llama_tokenizer(&activation)?;
    let bos = tokenizer
        .bos_token()
        .ok_or_else(|| "real model tokenizer has no BOS token".to_owned())?;

    let resolver = FileBackedTensorResolver::from_activation(shard_store, &activation);
    let generator = GraphGenerator::new(
        activation.graph.clone(),
        CpuReferenceProvider,
        BTreeMap::new(),
        activation.manifest.token_input,
        usize::try_from(activation.manifest.distribution_output)
            .map_err(|_| "distribution output does not fit usize".to_owned())?,
        usize::try_from(activation.manifest.vocabulary_size)
            .map_err(|_| "vocabulary size does not fit usize".to_owned())?,
    )
    .map_err(|error| format!("build graph generator: {error:?}"))?;
    let generated = generator
        .generate_tokens_with_resolver(
            &resolver,
            &[bos],
            GenerationConfig {
                max_new_tokens: STORIES260K_ANDROID_PROBE_TOKENS,
                context_limit: STORIES260K_CONTEXT_LIMIT,
                eos_token: Some(bos),
                distribution: DistributionKind::Logits,
                sampling: SamplingMode::Greedy,
            },
        )
        .map_err(|error| format!("native generation: {error:?}"))?;

    let actual_ids = format_token_ids(&generated.generated_tokens);
    if actual_ids != expected_token_ids.trim() {
        return Err(format!(
            "token divergence: expected={} actual={actual_ids}",
            expected_token_ids.trim()
        ));
    }
    let text = tokenizer
        .decode_after(Some(bos), &generated.generated_tokens, true)
        .map_err(|error| format!("decode generated text: {error:?}"))?;
    let text_hash = sha256(text.as_bytes());

    Ok(format!(
        "signature=ok\nactivation=ok\ngenerated_token_count={}\ngenerated_token_ids={}\ngenerated_text_bytes={}\ngenerated_text_sha256={}\nandroid_real_model=PASS\n",
        generated.generated_tokens.len(),
        actual_ids,
        text.len(),
        digest_hex(&text_hash)
    ))
}

#[allow(unsafe_code)]
#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdNativeRuntimeHost_nativeRealModelProbe(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    capsule_path: JString<'_>,
    shard_root: JString<'_>,
    verify_key: JByteArray<'_>,
    expected_token_ids: JString<'_>,
) -> jbyteArray {
    let Some(capsule_path) = java_string(&mut env, &capsule_path) else {
        return java_bytes(
            &env,
            b"android_real_model=failed\nerror=invalid capsule path\n",
        );
    };
    let Some(shard_root) = java_string(&mut env, &shard_root) else {
        return java_bytes(
            &env,
            b"android_real_model=failed\nerror=invalid shard root\n",
        );
    };
    let Some(expected_token_ids) = java_string(&mut env, &expected_token_ids) else {
        return java_bytes(
            &env,
            b"android_real_model=failed\nerror=invalid expected token ids\n",
        );
    };
    let verify_key = match env.convert_byte_array(&verify_key) {
        Ok(bytes) => bytes,
        Err(_) => {
            return java_bytes(
                &env,
                b"android_real_model=failed\nerror=invalid verify key\n",
            )
        }
    };

    match real_model_probe(&capsule_path, &shard_root, &verify_key, &expected_token_ids) {
        Ok(result) => java_bytes(&env, result.as_bytes()),
        Err(error) => java_bytes(
            &env,
            format!("android_real_model=failed\nerror={error}\n").as_bytes(),
        ),
    }
}

fn java_string(env: &mut JNIEnv<'_>, value: &JString<'_>) -> Option<String> {
    env.get_string(value)
        .ok()
        .map(|text| text.to_string_lossy().into_owned())
}

#[allow(unsafe_code)]
#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdPhysicalEvidenceActivity_nativeValidationWorkload(
    _env: JNIEnv<'_>,
    _class: JClass<'_>,
) -> jlong {
    run_native_validation_workload()
        .ok()
        .and_then(|value| i64::try_from(value).ok())
        .unwrap_or(-1)
}

#[allow(unsafe_code)]
#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdPhysicalEvidenceActivity_nativeRecoveryProbe(
    _env: JNIEnv<'_>,
    _class: JClass<'_>,
) -> jboolean {
    u8::from(run_logical_continuity_soak(1, 5).is_ok())
}

#[allow(clippy::too_many_arguments)]
#[allow(unsafe_code)]
#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdPhysicalEvidenceActivity_nativeEncodeEvidence(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    profile: JString<'_>,
    fingerprint_hash: JString<'_>,
    build_revision: JString<'_>,
    total_ram_bytes: jlong,
    p95_latency_nanos: jlong,
    energy_per_task_microjoules: jlong,
    reliability_permille: jint,
    recovery_permille: jint,
    sovereignty_audit_passed: jboolean,
    sample_count: jint,
    energy_source: JString<'_>,
) -> jbyteArray {
    let Some(profile) = java_string(&mut env, &profile) else {
        return java_bytes(&env, &[]);
    };
    let Some(device_fingerprint) = java_string(&mut env, &fingerprint_hash) else {
        return java_bytes(&env, &[]);
    };
    let Some(build_revision) = java_string(&mut env, &build_revision) else {
        return java_bytes(&env, &[]);
    };
    let Some(energy_source) = java_string(&mut env, &energy_source) else {
        return java_bytes(&env, &[]);
    };
    let Ok(total_ram_bytes) = u64::try_from(total_ram_bytes) else {
        return java_bytes(&env, &[]);
    };
    let Ok(p95_latency_nanos) = u64::try_from(p95_latency_nanos) else {
        return java_bytes(&env, &[]);
    };
    let Ok(energy_per_task_microjoules) = u64::try_from(energy_per_task_microjoules) else {
        return java_bytes(&env, &[]);
    };
    let Ok(reliability_permille) = u16::try_from(reliability_permille) else {
        return java_bytes(&env, &[]);
    };
    let Ok(recovery_permille) = u16::try_from(recovery_permille) else {
        return java_bytes(&env, &[]);
    };
    let Ok(sample_count) = u32::try_from(sample_count) else {
        return java_bytes(&env, &[]);
    };

    let record = PhysicalEvidenceRecord {
        evidence: DeviceEvidence {
            class: EvidenceClass::PhysicalDevice,
            profile,
            device_fingerprint,
            p95_latency_nanos,
            energy_per_task_microjoules,
            reliability_permille,
            recovery_permille,
            sovereignty_audit_passed: sovereignty_audit_passed != 0,
        },
        build_revision,
        total_ram_bytes,
        sample_count,
        energy_source,
    };
    match encode_physical_evidence(&record) {
        Ok(bytes) => java_bytes(&env, &bytes),
        Err(_) => java_bytes(&env, &[]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct AmbiguousExternalRetryAdapter;

    impl CapabilityAdapter for AmbiguousExternalRetryAdapter {
        fn execute(
            &mut self,
            _action_id: ntd_runtime::ActionId,
            _action: &TypedAction,
        ) -> Result<AdapterResult, String> {
            Err("external transport outcome is ambiguous".into())
        }
    }

    struct ResumableExternalRetryAdapter;

    impl CapabilityAdapter for ResumableExternalRetryAdapter {
        fn execute(
            &mut self,
            _action_id: ntd_runtime::ActionId,
            _action: &TypedAction,
        ) -> Result<AdapterResult, String> {
            Ok(AdapterResult::Retryable {
                reason: "retry with durable token".into(),
                resume_token: Some(vec![1, 2, 3]),
            })
        }
    }

    #[test]
    fn ambiguous_external_retry_requires_reconfirmation_but_resumable_token_does_not() {
        let mut browser_registry = CapabilityRegistry::new();
        browser_registry
            .register(
                CapabilityDescriptor::new(
                    CapabilityId("browser.interact".into()),
                    1,
                    CapabilityDomain::Browser,
                    SideEffectClass::ExternalWrite,
                )
                .expect("browser descriptor"),
            )
            .expect("register browser");
        let mut browser_fabric = ActionFabric::new(browser_registry);
        browser_fabric
            .register_adapter(
                CapabilityId("browser.interact".into()),
                AmbiguousExternalRetryAdapter,
            )
            .expect("browser adapter");
        let browser_graph = ntd_runtime::TaskGraph {
            actions: vec![ntd_runtime::ActionNode {
                id: 1,
                capability: CapabilityId("browser.interact".into()),
                side_effect: SideEffectClass::ExternalWrite,
                verification_required: true,
            }],
        };
        let browser_plan = browser_fabric
            .prepare_plan(
                1,
                &browser_graph,
                BTreeMap::from([(
                    1,
                    TypedAction::BrowserInteract {
                        target: "body".into(),
                        operation: "click".into(),
                        value: None,
                    },
                )]),
            )
            .expect("browser plan");
        let mut authority = AuthorityGrant::new();
        authority.allow_external_write = true;
        let report = browser_fabric
            .execute_next(browser_plan, &authority, &mut AndroidProductionVerifier)
            .expect("browser retry report");
        assert_eq!(report.action_status, Some(ActionStatus::Retryable));
        assert!(action_checkpoint_requires_reconfirm(
            &browser_fabric,
            browser_plan
        ));

        let mut upload_registry = CapabilityRegistry::new();
        let mut upload_descriptor = CapabilityDescriptor::new(
            CapabilityId("artifact.upload".into()),
            1,
            CapabilityDomain::Web,
            SideEffectClass::ExternalWrite,
        )
        .expect("upload descriptor");
        upload_descriptor.resumable = true;
        upload_registry
            .register(upload_descriptor)
            .expect("register upload");
        let mut upload_fabric = ActionFabric::new(upload_registry);
        upload_fabric
            .register_adapter(
                CapabilityId("artifact.upload".into()),
                ResumableExternalRetryAdapter,
            )
            .expect("upload adapter");
        let upload_graph = ntd_runtime::TaskGraph {
            actions: vec![ntd_runtime::ActionNode {
                id: 1,
                capability: CapabilityId("artifact.upload".into()),
                side_effect: SideEffectClass::ExternalWrite,
                verification_required: true,
            }],
        };
        let upload_plan = upload_fabric
            .prepare_plan(
                2,
                &upload_graph,
                BTreeMap::from([(
                    1,
                    TypedAction::ArtifactUpload {
                        url: "https://example.com/upload".into(),
                        path: "artifacts/report.bin".into(),
                    },
                )]),
            )
            .expect("upload plan");
        let report = upload_fabric
            .execute_next(upload_plan, &authority, &mut AndroidProductionVerifier)
            .expect("upload retry report");
        assert_eq!(report.action_status, Some(ActionStatus::Retryable));
        assert!(!action_checkpoint_requires_reconfirm(
            &upload_fabric,
            upload_plan
        ));
    }

    #[test]
    fn governed_device_policy_routes_live_evidence() {
        assert_eq!(
            governed_device_surfaces("What is my current battery status?"),
            Some(vec!["battery", "resources"])
        );
        assert_eq!(
            governed_device_surfaces("Is my phone temperature high?"),
            Some(vec!["thermal", "resources"])
        );
        assert_eq!(
            governed_device_surfaces("How much RAM is available?"),
            Some(vec!["memory", "resources"])
        );
        assert_eq!(
            governed_device_surfaces("remember our conversation memory"),
            None
        );
        assert_eq!(governed_device_surfaces("tell me a story"), None);
    }

    #[test]
    fn normalized_web_search_decoder_and_verifier_bind_count_and_digest() {
        fn push_string(out: &mut Vec<u8>, value: &str) {
            let bytes = value.as_bytes();
            out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(bytes);
        }

        let mut encoded = vec![PLATFORM_WEB_PROTOCOL_VERSION, 1];
        encoded.extend_from_slice(&200u32.to_le_bytes());
        push_string(&mut encoded, "https://api.example.com");
        encoded.extend_from_slice(&1u32.to_le_bytes());
        push_string(&mut encoded, "NTD97 result");
        push_string(&mut encoded, "https://example.com/result");
        push_string(&mut encoded, "normalized snippet");

        let decoded = decode_android_web_search_result(&encoded).expect("decode search result");
        assert_eq!(decoded.status, 200);
        assert_eq!(decoded.source, "https://api.example.com");
        assert_eq!(decoded.items.len(), 1);

        let items = vec!["NTD97 result\nhttps://example.com/result\nnormalized snippet".to_owned()];
        let hash = digest_hex(&normalized_search_value_hash(&items).expect("hash"));
        let output = ActionOutput {
            summary: "verified normalized search".into(),
            value: ActionValue::TextList(items.clone()),
            evidence: vec![
                "android-https-search".into(),
                "provider-boundary:runtime-configured".into(),
                "normalization:field-map-v1".into(),
                "status:200".into(),
                "source:https://api.example.com".into(),
                "results:1".into(),
                format!("results-sha256:{hash}"),
                "max-results:5".into(),
            ],
        };
        let descriptor = CapabilityDescriptor::new(
            CapabilityId("web.search".into()),
            1,
            CapabilityDomain::Web,
            SideEffectClass::ReadOnly,
        )
        .expect("descriptor");
        let action = TypedAction::WebSearch {
            query: "ntd97".into(),
            max_results: 5,
        };
        let mut verifier = AndroidProductionVerifier;
        assert_eq!(
            verifier.verify(&descriptor, &action, &output),
            ActionVerification::Accept
        );

        let mut tampered = output;
        let ActionValue::TextList(values) = &mut tampered.value else {
            panic!("text list");
        };
        values[0].push('x');
        assert!(matches!(
            verifier.verify(&descriptor, &action, &tampered),
            ActionVerification::Reject { .. }
        ));
    }

    #[test]
    fn governed_explicit_web_and_browser_commands_materialize_canonical_actions() {
        let search = governed_explicit_action_plan("search web for NTD97 mobile")
            .expect("search plan")
            .expect("search action");
        assert!(matches!(
            search.payloads.get(&1),
            Some(TypedAction::WebSearch {
                query,
                max_results: 5
            }) if query == "NTD97 mobile"
        ));

        let observe = governed_explicit_action_plan("observe browser https://example.com/")
            .expect("observe plan")
            .expect("observe action");
        assert_eq!(
            observe.payloads.get(&1),
            Some(&TypedAction::BrowserObserve {
                target: "https://example.com/".into(),
            })
        );

        let click = governed_explicit_action_plan("browser click body")
            .expect("click plan")
            .expect("click action");
        assert_eq!(
            click.payloads.get(&1),
            Some(&TypedAction::BrowserInteract {
                target: "body".into(),
                operation: "click".into(),
                value: None,
            })
        );
        assert_eq!(
            click.graph.actions[0].side_effect,
            SideEffectClass::ExternalWrite
        );

        let resume = governed_explicit_action_plan("resume browser")
            .expect("resume plan")
            .expect("resume action");
        assert_eq!(
            resume.payloads.get(&1),
            Some(&TypedAction::BrowserObserve {
                target: "session".into(),
            })
        );

        let navigate = governed_explicit_action_plan("browser navigate https://example.com/docs")
            .expect("navigate plan")
            .expect("navigate action");
        assert_eq!(
            navigate.payloads.get(&1),
            Some(&TypedAction::BrowserInteract {
                target: "https://example.com/docs".into(),
                operation: "navigate".into(),
                value: None,
            })
        );

        let set_value =
            governed_explicit_action_plan("browser set input[name='q'] to NTD97 mobile")
                .expect("set_value plan")
                .expect("set_value action");
        assert_eq!(
            set_value.payloads.get(&1),
            Some(&TypedAction::BrowserInteract {
                target: "input[name='q']".into(),
                operation: "set_value".into(),
                value: Some("NTD97 mobile".into()),
            })
        );

        let submit = governed_explicit_action_plan("browser submit form#search")
            .expect("submit plan")
            .expect("submit action");
        assert_eq!(
            submit.payloads.get(&1),
            Some(&TypedAction::BrowserInteract {
                target: "form#search".into(),
                operation: "submit".into(),
                value: None,
            })
        );
    }

    #[test]
    fn browser_interaction_requires_chat_approval_and_action_bound_receipt() {
        let plan = governed_explicit_action_plan("browser click body")
            .expect("click plan")
            .expect("click action");
        let approval = external_write_approval(&plan)
            .expect("approval policy")
            .expect("approval requirement");
        assert_eq!(approval.0, "browser.interact");
        assert!(approval.1.contains("click"));

        let descriptor = CapabilityDescriptor::new(
            CapabilityId("browser.interact".into()),
            1,
            CapabilityDomain::Browser,
            SideEffectClass::ExternalWrite,
        )
        .expect("browser descriptor");
        let action = plan.payloads.get(&1).expect("browser action");
        let digest = digest_hex(&sha256(b"click\nbody\n"));
        let receipt = format!("android-webview:click:1:{digest}");
        let platform = format!(
            "receipt={receipt}\noperation=click\ntarget=body\nurl=https://example.com/\ntitle=Example Domain\ntag=BODY"
        );
        let accepted = ActionOutput {
            summary: "verified browser click".into(),
            value: ActionValue::Fields(BTreeMap::from([
                ("operation".into(), "click".into()),
                ("target".into(), "body".into()),
                ("receipt".into(), receipt.clone()),
                ("platform".into(), platform),
            ])),
            evidence: vec![
                "android-webview-browser".into(),
                "operation:click".into(),
                format!("receipt:{receipt}"),
            ],
        };
        let mut verifier = AndroidProductionVerifier;
        assert_eq!(
            verifier.verify(&descriptor, action, &accepted),
            ActionVerification::Accept
        );

        let mut tampered = accepted.clone();
        let ActionValue::Fields(fields) = &mut tampered.value else {
            panic!("fields");
        };
        fields.insert(
            "receipt".into(),
            format!("android-webview:click:1:{}", "0".repeat(64)),
        );
        assert!(matches!(
            verifier.verify(&descriptor, action, &tampered),
            ActionVerification::Reject { .. }
        ));

        let set_plan = governed_explicit_action_plan("browser set input#q to private-value")
            .expect("set plan")
            .expect("set action");
        let set_action = set_plan.payloads.get(&1).expect("set action");
        let value_hash = digest_hex(&sha256(b"private-value"));
        let set_digest = digest_hex(&sha256(b"set_value\ninput#q\nprivate-value"));
        let set_receipt = format!("android-webview:set_value:2:{set_digest}");
        let set_platform = format!(
            "receipt={set_receipt}\noperation=set_value\ntarget=input#q\nurl=https://example.com/\ntitle=Example\ntag=INPUT\nvalue_sha256={value_hash}"
        );
        let set_output = ActionOutput {
            summary: "verified browser set_value".into(),
            value: ActionValue::Fields(BTreeMap::from([
                ("operation".into(), "set_value".into()),
                ("target".into(), "input#q".into()),
                ("receipt".into(), set_receipt.clone()),
                ("platform".into(), set_platform),
            ])),
            evidence: vec![
                "android-webview-browser".into(),
                "operation:set_value".into(),
                format!("receipt:{set_receipt}"),
            ],
        };
        assert_eq!(
            verifier.verify(&descriptor, set_action, &set_output),
            ActionVerification::Accept
        );
        let mut wrong_hash = set_output;
        let ActionValue::Fields(fields) = &mut wrong_hash.value else {
            panic!("fields");
        };
        fields
            .get_mut("platform")
            .expect("platform")
            .push_str("\nvalue_sha256=00");
        assert!(matches!(
            verifier.verify(&descriptor, set_action, &wrong_hash),
            ActionVerification::Reject { .. }
        ));
    }

    #[test]
    fn paired_pc_commands_require_external_approval_and_authenticated_evidence() {
        let observe = governed_explicit_action_plan("observe pc workstation system")
            .expect("observe plan")
            .expect("observe action");
        assert_eq!(
            observe.payloads.get(&1),
            Some(&TypedAction::PcObserve {
                peer: "workstation".into(),
                surface: "system".into(),
            })
        );
        assert_eq!(
            observe.graph.actions[0].side_effect,
            SideEffectClass::ReadOnly
        );

        let execute = governed_explicit_action_plan("execute pc workstation echo")
            .expect("execute plan")
            .expect("execute action");
        assert_eq!(
            execute.payloads.get(&1),
            Some(&TypedAction::PcExecute {
                peer: "workstation".into(),
                program: "echo".into(),
                args: Vec::new(),
                working_dir: None,
            })
        );
        let approval = external_write_approval(&execute)
            .expect("approval policy")
            .expect("approval required");
        assert_eq!(approval.0, "pc.execute");

        let descriptor = CapabilityDescriptor::new(
            CapabilityId("pc.execute".into()),
            1,
            CapabilityDomain::Pc,
            SideEffectClass::ExternalWrite,
        )
        .expect("descriptor");
        let action = execute.payloads.get(&1).expect("action");
        let accepted = ActionOutput {
            summary: "remote process completed".into(),
            value: ActionValue::Fields(BTreeMap::from([("exit_code".into(), "0".into())])),
            evidence: vec![
                "success=true".into(),
                "pcf97-authenticated".into(),
                "peer:workstation".into(),
                "remote-capability:pc.process.execute".into(),
            ],
        };
        let mut verifier = AndroidProductionVerifier;
        assert_eq!(
            verifier.verify(&descriptor, action, &accepted),
            ActionVerification::Accept
        );
        let mut rejected = accepted;
        rejected
            .evidence
            .retain(|item| item != "pcf97-authenticated");
        assert!(matches!(
            verifier.verify(&descriptor, action, &rejected),
            ActionVerification::Reject { .. }
        ));
    }

    #[test]
    fn paired_pc_profile_is_strictly_pinned_to_alias_and_identity() {
        let root = std::env::temp_dir().join(format!(
            "ntd97-pc-profile-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        let pairs = root.join("pc-pairs");
        fs::create_dir_all(&pairs).expect("pairs");
        let remote = PairedIdentity::from_seed([72; 32]);
        let hex = |bytes: &[u8]| {
            let mut encoded = String::with_capacity(bytes.len() * 2);
            for byte in bytes {
                std::fmt::Write::write_fmt(&mut encoded, format_args!("{byte:02x}"))
                    .expect("write hex");
            }
            encoded
        };
        let profile = format!(
            "NTD97_PC_PAIR_V1\npeer=workstation\naddress=127.0.0.1:45970\nlocal_seed={}\nremote_peer_id={}\nremote_verify_key={}\nEND",
            hex(&[71; 32]),
            hex(&remote.peer_id()),
            hex(&remote.verify_key()),
        );
        fs::write(pairs.join("workstation.pcp97"), profile).expect("write profile");

        let loaded = load_pc_pair_profile(&root, "workstation").expect("load profile");
        assert_eq!(loaded.peer, "workstation");
        assert_eq!(loaded.address, "127.0.0.1:45970".parse().expect("address"));
        assert_eq!(loaded.local_seed, [71; 32]);
        assert_eq!(loaded.remote_peer_id, remote.peer_id());
        assert_eq!(loaded.remote_verify_key, remote.verify_key());
        assert!(load_pc_pair_profile(&root, "other").is_err());

        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn paired_pc_provision_describe_list_and_revoke_keep_seed_private() {
        let root = std::env::temp_dir().join(format!(
            "ntd97-pc-provision-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        let remote = PairedIdentity::from_seed([72; 32]);
        let receipt = provision_pc_pair_profile(
            &root,
            "workstation",
            "127.0.0.1:45970",
            &fixed_hex(&remote.peer_id()),
            &fixed_hex(&remote.verify_key()),
        )
        .expect("provision");
        assert!(receipt.starts_with("NTD97_PC_PAIR_RECEIPT_V1\n"));
        assert!(receipt.contains("peer=workstation\n"));
        assert!(receipt.contains("local_peer_id="));
        assert!(receipt.contains("local_verify_key="));
        assert!(!receipt.contains("local_seed"));
        assert_eq!(
            list_pc_pair_profiles(&root).expect("list"),
            vec!["workstation".to_owned()]
        );
        assert_eq!(
            describe_pc_pair_profile(&root, "workstation").expect("describe"),
            receipt
        );

        let loaded = load_pc_pair_profile(&root, "workstation").expect("load");
        assert_eq!(loaded.remote_peer_id, remote.peer_id());
        assert_eq!(loaded.remote_verify_key, remote.verify_key());
        assert_ne!(loaded.local_seed, [0; 32]);
        assert!(provision_pc_pair_profile(
            &root,
            "workstation",
            "127.0.0.1:45970",
            &fixed_hex(&remote.peer_id()),
            &fixed_hex(&remote.verify_key()),
        )
        .is_err());

        revoke_pc_pair_profile(&root, "workstation").expect("revoke");
        assert!(list_pc_pair_profiles(&root).expect("empty list").is_empty());
        assert!(load_pc_pair_profile(&root, "workstation").is_err());
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn paired_pc_provision_rejects_invalid_address_and_mismatched_identity() {
        let root = std::env::temp_dir().join(format!(
            "ntd97-pc-provision-invalid-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        let remote = PairedIdentity::from_seed([73; 32]);
        assert!(provision_pc_pair_profile(
            &root,
            "workstation",
            "0.0.0.0:45970",
            &fixed_hex(&remote.peer_id()),
            &fixed_hex(&remote.verify_key()),
        )
        .is_err());

        let wrong_peer_id = [0x55; 16];
        assert!(provision_pc_pair_profile(
            &root,
            "workstation",
            "127.0.0.1:45970",
            &fixed_hex(&wrong_peer_id),
            &fixed_hex(&remote.verify_key()),
        )
        .is_err());
        assert!(list_pc_pair_profiles(&root).expect("list").is_empty());
        if root.exists() {
            fs::remove_dir_all(root).expect("cleanup");
        }
    }

    #[test]
    fn accessibility_commands_use_dedicated_scope_and_receipt_binding() {
        let package = "ai.ntd97.mobile";
        let view_id = "ai.ntd97.mobile:id/ntd_accessibility_probe_button";
        let click =
            governed_explicit_action_plan(&format!("accessibility click {package} {view_id}"))
                .expect("click plan")
                .expect("click action");
        assert_eq!(
            click.payloads.get(&1),
            Some(&TypedAction::AppAction {
                app: package.into(),
                action: "accessibility.click".into(),
                payload: format!("view_id\t{view_id}").into_bytes(),
            })
        );
        let click_scopes = app_action_scope_names(&click).expect("click scopes");
        assert_eq!(
            click_scopes,
            std::collections::BTreeSet::from(["app.accessibility.interact"])
        );
        let approval = external_write_approval(&click)
            .expect("approval policy")
            .expect("approval required");
        assert_eq!(approval.0, "app.action");
        assert!(approval.1.contains("accessibility.click"));
        assert!(!approval.1.contains("NTD97-accessibility"));

        let set_text = governed_explicit_action_plan(&format!(
            "accessibility set text {package} ai.ntd97.mobile:id/ntd_accessibility_probe_text to NTD97-accessibility"
        ))
        .expect("set text plan")
        .expect("set text action");
        let set_text_action = set_text.payloads.get(&1).expect("set text payload");
        let TypedAction::AppAction {
            action, payload, ..
        } = set_text_action
        else {
            panic!("expected app action");
        };
        assert_eq!(action, "accessibility.set_text");
        let spec = parse_accessibility_action(action, payload).expect("accessibility spec");
        assert_eq!(spec.selector_kind, "view_id");
        assert_eq!(
            spec.selector_value,
            "ai.ntd97.mobile:id/ntd_accessibility_probe_text"
        );
        assert_eq!(spec.text_bytes, Some("NTD97-accessibility".len()));

        let launch = governed_explicit_action_plan("open app ai.ntd97.mobile")
            .expect("launch plan")
            .expect("launch action");
        assert_eq!(
            app_action_scope_names(&launch).expect("launch scopes"),
            std::collections::BTreeSet::from(["app.launch"])
        );

        let descriptor = CapabilityDescriptor::new(
            CapabilityId("app.action".into()),
            1,
            CapabilityDomain::App,
            SideEffectClass::ExternalWrite,
        )
        .expect("descriptor");
        let expected_receipt = format!(
            "accessibility:{action}:{package}:{}",
            digest_hex(&sha256(payload))
        );
        let output = ActionOutput {
            summary: "verified accessibility".into(),
            value: ActionValue::Fields(BTreeMap::from([
                ("package".into(), package.into()),
                ("operation".into(), action.clone()),
                ("selector_kind".into(), spec.selector_kind.clone()),
                ("selector".into(), spec.selector_value.clone()),
                (
                    "text_bytes".into(),
                    spec.text_bytes.expect("text bytes").to_string(),
                ),
                ("receipt".into(), expected_receipt.clone()),
            ])),
            evidence: vec![
                "android-accessibility-interaction".into(),
                format!("operation:{action}"),
                expected_receipt.clone(),
            ],
        };
        let mut verifier = AndroidProductionVerifier;
        assert_eq!(
            verifier.verify(&descriptor, set_text_action, &output),
            ActionVerification::Accept
        );

        let mut rejected = output;
        rejected.evidence.retain(|item| item != &expected_receipt);
        assert!(matches!(
            verifier.verify(&descriptor, set_text_action, &rejected),
            ActionVerification::Reject { .. }
        ));
    }

    #[test]
    fn governed_storage_and_upload_commands_materialize_canonical_actions() {
        let read = governed_explicit_action_plan("read granted file shared/notes/read.txt")
            .expect("read plan")
            .expect("read action");
        assert_eq!(
            read.payloads.get(&1),
            Some(&TypedAction::FileRead {
                path: "shared\tnotes/read.txt".into(),
            })
        );

        let write =
            governed_explicit_action_plan("write granted file shared/notes/write.txt to sovereign")
                .expect("write plan")
                .expect("write action");
        assert_eq!(
            write.payloads.get(&1),
            Some(&TypedAction::FileWrite {
                path: "shared\tnotes/write.txt".into(),
                bytes: b"sovereign".to_vec(),
            })
        );
        assert_eq!(
            write.graph.actions[0].side_effect,
            SideEffectClass::ExternalWrite
        );
        let write_approval = external_write_approval(&write)
            .expect("write approval")
            .expect("write approval required");
        assert_eq!(write_approval.0, "file.grant.write");

        let explicit_upload = governed_explicit_action_plan(
            "upload artifact artifacts/report.bin to https://example.com/upload",
        )
        .expect("explicit upload plan")
        .expect("explicit upload action");
        assert_eq!(
            explicit_upload.payloads.get(&1),
            Some(&TypedAction::ArtifactUpload {
                url: "https://example.com/upload".into(),
                path: "artifacts/report.bin".into(),
            })
        );
        assert_eq!(
            explicit_upload.graph.actions[0].side_effect,
            SideEffectClass::ExternalWrite
        );

        let upload = match parse_native_action_plan(
            "NTD97_ACTIONS_V1\n1|artifact.upload|https://example.com/upload\tartifacts/report.bin\nEND",
        )
        .expect("upload protocol")
        {
            AssistantPlanDecision::Actions(plan) => plan,
            AssistantPlanDecision::Direct => panic!("expected upload action plan"),
        };
        let upload_approval = external_write_approval(&upload)
            .expect("upload approval")
            .expect("upload approval required");
        assert_eq!(upload_approval.0, "artifact.upload");
        assert!(upload_approval.1.contains("idempotent HTTPS PUT"));
    }

    #[test]
    fn production_verifier_requires_storage_and_upload_receipts() {
        let mut verifier = AndroidProductionVerifier;
        let grant_descriptor = CapabilityDescriptor::new(
            CapabilityId("file.grant.write".into()),
            1,
            CapabilityDomain::File,
            SideEffectClass::ExternalWrite,
        )
        .expect("grant descriptor");
        let grant_action = TypedAction::FileWrite {
            path: "shared\tnotes/out.txt".into(),
            bytes: b"hello".to_vec(),
        };
        let grant_output = ActionOutput {
            summary: "granted write".into(),
            value: ActionValue::Fields(BTreeMap::from([
                ("grant".into(), "shared".into()),
                ("path".into(), "notes/out.txt".into()),
                ("bytes".into(), "5".into()),
                ("receipt".into(), "grant-write:shared:notes/out.txt:5:2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824".into()),
            ])),
            evidence: vec![
                "android-user-granted-file".into(),
                "grant:shared".into(),
                "path:notes/out.txt".into(),
                "operation:write".into(),
                "receipt:grant-write:shared:notes/out.txt:5:2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824".into(),
            ],
        };
        assert_eq!(
            verifier.verify(&grant_descriptor, &grant_action, &grant_output),
            ActionVerification::Accept
        );

        let upload_descriptor = CapabilityDescriptor::new(
            CapabilityId("artifact.upload".into()),
            1,
            CapabilityDomain::Web,
            SideEffectClass::ExternalWrite,
        )
        .expect("upload descriptor");
        let upload_action = TypedAction::ArtifactUpload {
            url: "https://example.com/upload".into(),
            path: "artifacts/report.bin".into(),
        };
        let upload_output = ActionOutput {
            summary: "uploaded".into(),
            value: ActionValue::Fields(BTreeMap::from([
                ("url".into(), "https://example.com/upload".into()),
                ("status".into(), "200".into()),
                (
                    "sha256".into(),
                    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into(),
                ),
                ("bytes".into(), "5".into()),
            ])),
            evidence: vec![
                "android-artifact-upload".into(),
                "transport:https-put".into(),
                "status:200".into(),
                "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into(),
                "bytes:5".into(),
                "url:https://example.com/upload".into(),
            ],
        };
        assert_eq!(
            verifier.verify(&upload_descriptor, &upload_action, &upload_output),
            ActionVerification::Accept
        );

        let missing_hash = ActionOutput {
            summary: "uploaded".into(),
            value: ActionValue::Fields(BTreeMap::from([
                ("url".into(), "https://example.com/upload".into()),
                ("status".into(), "200".into()),
                (
                    "sha256".into(),
                    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into(),
                ),
                ("bytes".into(), "5".into()),
            ])),
            evidence: vec![
                "android-artifact-upload".into(),
                "transport:https-put".into(),
                "status:200".into(),
                "bytes:5".into(),
                "url:https://example.com/upload".into(),
            ],
        };
        assert!(matches!(
            verifier.verify(&upload_descriptor, &upload_action, &missing_hash),
            ActionVerification::Reject { .. }
        ));
    }

    #[test]
    fn external_write_capabilities_require_scope_and_explicit_write_authority() {
        let clipboard_scope = AuthorityScope::new("device.clipboard.write").expect("scope");
        let mut clipboard = CapabilityDescriptor::new(
            CapabilityId("device.interact".into()),
            1,
            CapabilityDomain::Device,
            SideEffectClass::ExternalWrite,
        )
        .expect("descriptor");
        clipboard.required_scopes.push(clipboard_scope.clone());
        clipboard.normalize().expect("normalize");

        assert!(AuthorityGrant::new().permits(&clipboard).is_err());
        let scoped_without_write = AuthorityGrant::new().with_scope(clipboard_scope.clone());
        assert!(scoped_without_write.permits(&clipboard).is_err());
        let mut allowed = AuthorityGrant::new().with_scope(clipboard_scope);
        allowed.allow_external_write = true;
        assert!(allowed.permits(&clipboard).is_ok());

        let app_scope = AuthorityScope::new("app.launch").expect("scope");
        let mut app = CapabilityDescriptor::new(
            CapabilityId("app.action".into()),
            1,
            CapabilityDomain::App,
            SideEffectClass::ExternalWrite,
        )
        .expect("descriptor");
        app.required_scopes.push(app_scope.clone());
        app.normalize().expect("normalize");

        assert!(AuthorityGrant::new().permits(&app).is_err());
        let mut app_allowed = AuthorityGrant::new().with_scope(app_scope);
        app_allowed.allow_external_write = true;
        assert!(app_allowed.permits(&app).is_ok());
    }

    #[test]
    fn production_verifier_requires_device_and_app_receipts() {
        let mut verifier = AndroidProductionVerifier;
        let device_descriptor = CapabilityDescriptor::new(
            CapabilityId("device.interact".into()),
            1,
            CapabilityDomain::Device,
            SideEffectClass::ExternalWrite,
        )
        .expect("device descriptor");
        let device_action = TypedAction::DeviceInteract {
            surface: "clipboard".into(),
            operation: "set_text".into(),
            argument: Some("NTD97".into()),
        };
        let device_receipt = format!(
            "clipboard-set:{}:{}",
            "NTD97".len(),
            digest_hex(&sha256(b"NTD97"))
        );
        let accepted_device = ActionOutput {
            summary: "clipboard written".into(),
            value: ActionValue::None,
            evidence: vec![
                "android-clipboard-write".into(),
                "operation:set_text".into(),
                device_receipt,
            ],
        };
        assert_eq!(
            verifier.verify(&device_descriptor, &device_action, &accepted_device),
            ActionVerification::Accept
        );

        let mismatched_device = ActionOutput {
            summary: "clipboard receipt mismatch".into(),
            value: ActionValue::None,
            evidence: vec![
                "android-clipboard-write".into(),
                "operation:set_text".into(),
                "clipboard-set:5:0000000000000000000000000000000000000000000000000000000000000000"
                    .into(),
            ],
        };
        assert!(matches!(
            verifier.verify(&device_descriptor, &device_action, &mismatched_device),
            ActionVerification::Reject { .. }
        ));

        let app_descriptor = CapabilityDescriptor::new(
            CapabilityId("app.action".into()),
            1,
            CapabilityDomain::App,
            SideEffectClass::ExternalWrite,
        )
        .expect("app descriptor");
        let app_action = TypedAction::AppAction {
            app: "ai.ntd97.mobile".into(),
            action: "launch".into(),
            payload: Vec::new(),
        };
        let rejected_app = ActionOutput {
            summary: "launch claimed".into(),
            value: ActionValue::None,
            evidence: vec!["operation:launch".into()],
        };
        assert!(matches!(
            verifier.verify(&app_descriptor, &app_action, &rejected_app),
            ActionVerification::Reject { .. }
        ));
    }

    #[test]
    fn device_adapter_scopes_verified_fields_to_selected_surface() {
        let snapshot = ResourceSnapshot {
            available_ram_bytes: 123,
            battery_percent: 77,
            charging: true,
            thermal: ThermalState::Nominal,
            latency_budget_ms: 42,
        };
        let mut adapter = AndroidResourceAdapter { snapshot };
        let result = adapter
            .execute(
                ntd_runtime::ActionId(1),
                &TypedAction::DeviceObserve {
                    surface: "battery".into(),
                },
            )
            .expect("battery observation");
        let AdapterResult::Completed { output, .. } = result else {
            panic!("expected completed observation");
        };
        let ActionValue::Fields(fields) = output.value else {
            panic!("expected field evidence");
        };

        assert_eq!(
            fields.get("battery_percent").map(String::as_str),
            Some("77")
        );
        assert_eq!(fields.get("charging").map(String::as_str), Some("true"));
        assert_eq!(fields.get("surface").map(String::as_str), Some("battery"));
        assert!(!fields.contains_key("available_ram_bytes"));
        assert!(!fields.contains_key("thermal"));
        assert_eq!(output.evidence, vec!["android-resource-snapshot"]);
    }
}
