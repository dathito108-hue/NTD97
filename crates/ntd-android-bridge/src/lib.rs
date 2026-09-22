#![deny(unsafe_code)]
#![allow(non_snake_case)]

use std::{
    collections::BTreeMap,
    fs,
    ptr::null_mut,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex, OnceLock,
    },
};

use jni::{
    objects::{GlobalRef, JByteArray, JClass, JObject, JString, JValue},
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
use ntd_runtime::{
    choose_reasoning_budget, decode_conversation_checkpoint, encode_conversation_checkpoint,
    execute_verified_assistant_plan, memory_recall_limit_for_budget, model_inference_signals,
    parse_native_action_plan, run_budgeted_reasoning_cycle, sample_token, ActionFabric,
    ActionOutput, ActionValue, ActionVerification, ActionVerifier, AdapterResult,
    AssistantActionPlan, AssistantPlanDecision, AuthorityGrant, AuthorityScope, CapabilityAdapter,
    CapabilityDescriptor, CapabilityDomain, CapabilityId, CapabilityRegistry, CognitiveContext,
    CognitiveIdentity, CognitiveObservation, CpuReferenceProvider, DistributionKind,
    GenerationConfig, GenerationControl, GraphGenerator, LlamaSpmConfig, LlamaSpmTokenizer,
    NativeChatPromptCompiler, NativeReasoningProbe, ResourceSnapshot, SamplingMode,
    SideEffectClass, SovereignConversationState, TaskStatus, ThermalState, TypedAction,
    NATIVE_ACTION_DIRECT, NATIVE_ACTION_PROTOCOL_V1,
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
const CHAT_STATUS_MISSING: i32 = 0;
const CHAT_STATUS_RUNNING: i32 = 1;
const CHAT_STATUS_COMPLETE: i32 = 2;
const CHAT_STATUS_CANCELLED: i32 = 3;
const CHAT_STATUS_FAILED: i32 = 4;
const ACTION_PLANNER_MAX_NEW_TOKENS: usize = 20;

const WEB_FETCH_MAX_BYTES: usize = 64 * 1024;
const WEB_FETCH_CONNECT_TIMEOUT_MS: i32 = 5_000;
const WEB_FETCH_READ_TIMEOUT_MS: i32 = 5_000;
const WEB_FETCH_MAX_REDIRECTS: i32 = 5;
const WEB_RESPONSE_MAGIC: [u8; 6] = *b"NWR97\0";
const WEB_RESPONSE_VERSION: u16 = 1;

struct NativeChatModel {
    asset_id: String,
    version: u32,
    activation: ThinGenerativeActivation,
    resolver: FileBackedTensorResolver,
    tokenizer: LlamaSpmTokenizer,
    context_limit: usize,
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

struct AndroidPlatformWebBridge {
    vm: JavaVM,
    object: GlobalRef,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AndroidWebResponse {
    status: i32,
    redirects: u32,
    final_url: String,
    content_type: String,
    body: Vec<u8>,
    error: String,
}

impl AndroidPlatformWebBridge {
    fn new(env: &JNIEnv<'_>, object: JObject<'_>) -> Result<Self, String> {
        let vm = env
            .get_java_vm()
            .map_err(|error| format!("get Android JavaVM: {error}"))?;
        let object = env
            .new_global_ref(object)
            .map_err(|error| format!("retain Android web transport: {error}"))?;
        Ok(Self { vm, object })
    }

    fn fetch(&self, url: &str) -> Result<AndroidWebResponse, String> {
        let mut env = self
            .vm
            .attach_current_thread()
            .map_err(|error| format!("attach Android web transport thread: {error}"))?;
        let java_url = env
            .new_string(url)
            .map_err(|error| format!("encode web URL for Android: {error}"))?;
        let java_url_object = JObject::from(java_url);
        let call = env.call_method(
            self.object.as_obj(),
            "fetch",
            "(Ljava/lang/String;IIII)[B",
            &[
                JValue::Object(&java_url_object),
                JValue::Int(
                    i32::try_from(WEB_FETCH_MAX_BYTES)
                        .map_err(|_| "web body limit does not fit jint".to_owned())?,
                ),
                JValue::Int(WEB_FETCH_CONNECT_TIMEOUT_MS),
                JValue::Int(WEB_FETCH_READ_TIMEOUT_MS),
                JValue::Int(WEB_FETCH_MAX_REDIRECTS),
            ],
        );
        let result = match call {
            Ok(value) => value,
            Err(error) => {
                if env.exception_check().unwrap_or(false) {
                    let _ = env.exception_clear();
                }
                return Err(format!("Android web transport call failed: {error}"));
            }
        };
        let object = result
            .l()
            .map_err(|error| format!("Android web transport returned invalid value: {error}"))?;
        let bytes = env
            .convert_byte_array(&JByteArray::from(object))
            .map_err(|error| format!("decode Android web transport frame: {error}"))?;
        decode_android_web_response(&bytes)
    }
}

struct AndroidWebFetchAdapter {
    bridge: Arc<AndroidPlatformWebBridge>,
}

impl CapabilityAdapter for AndroidWebFetchAdapter {
    fn execute(
        &mut self,
        _action_id: ntd_runtime::ActionId,
        action: &TypedAction,
    ) -> Result<AdapterResult, String> {
        let TypedAction::WebFetch { url } = action else {
            return Err("Android web adapter only supports web.fetch".into());
        };
        let response = self.bridge.fetch(url)?;
        if !response.error.is_empty() {
            return Err(response.error);
        }

        let value = if is_textual_content_type(&response.content_type) {
            match String::from_utf8(response.body.clone()) {
                Ok(text) => ActionValue::Text(text),
                Err(_) => ActionValue::Bytes(response.body.clone()),
            }
        } else {
            ActionValue::Bytes(response.body.clone())
        };
        let digest = sha256(&response.body);
        Ok(AdapterResult::Completed {
            output: ActionOutput {
                summary: format!(
                    "HTTP {} {}",
                    response.status,
                    response.final_url
                ),
                value,
                evidence: vec![
                    "transport=android-http-url-connection".into(),
                    format!("http_status={}", response.status),
                    format!("final_url={}", response.final_url),
                    format!("content_type={}", response.content_type),
                    format!("bytes={}", response.body.len()),
                    format!("sha256={}", digest_hex(&digest)),
                    format!("redirects={}", response.redirects),
                ],
            },
            rollback_token: None,
        })
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct AndroidWebVerifier;

impl ActionVerifier for AndroidWebVerifier {
    fn verify(
        &mut self,
        descriptor: &CapabilityDescriptor,
        action: &TypedAction,
        output: &ActionOutput,
    ) -> ActionVerification {
        if descriptor.id.0 != "web.fetch" || !matches!(action, TypedAction::WebFetch { .. }) {
            return ActionVerification::Reject {
                reason: "web fetch capability contract mismatch".into(),
            };
        }

        let evidence = output
            .evidence
            .iter()
            .filter_map(|item| item.split_once('='))
            .collect::<BTreeMap<_, _>>();
        if evidence.get("transport") != Some(&"android-http-url-connection") {
            return ActionVerification::Reject {
                reason: "web fetch lacks trusted Android transport evidence".into(),
            };
        }
        let status = evidence
            .get("http_status")
            .and_then(|value| value.parse::<u16>().ok());
        if !status.is_some_and(|value| (200..300).contains(&value)) {
            return ActionVerification::Reject {
                reason: "web fetch HTTP status is not successful".into(),
            };
        }

        let payload = match &output.value {
            ActionValue::Text(text) => text.as_bytes(),
            ActionValue::Bytes(bytes) => bytes.as_slice(),
            _ => {
                return ActionVerification::Reject {
                    reason: "web fetch output is not a body payload".into(),
                }
            }
        };
        let expected_len = evidence
            .get("bytes")
            .and_then(|value| value.parse::<usize>().ok());
        if expected_len != Some(payload.len()) {
            return ActionVerification::Reject {
                reason: "web fetch byte-count evidence mismatch".into(),
            };
        }
        let digest = sha256(payload);
        if evidence.get("sha256") != Some(&digest_hex(&digest).as_str()) {
            return ActionVerification::Reject {
                reason: "web fetch digest evidence mismatch".into(),
            };
        }
        let final_url = evidence.get("final_url").copied().unwrap_or_default();
        if !(final_url.starts_with("http://") || final_url.starts_with("https://")) {
            return ActionVerification::Reject {
                reason: "web fetch final URL is outside the HTTP boundary".into(),
            };
        }
        ActionVerification::Accept
    }
}

fn decode_android_web_response(bytes: &[u8]) -> Result<AndroidWebResponse, String> {
    let mut cursor = WebResponseCursor::new(bytes);
    if cursor.take(WEB_RESPONSE_MAGIC.len())? != WEB_RESPONSE_MAGIC.as_slice() {
        return Err("Android web response has invalid magic".into());
    }
    if cursor.u16()? != WEB_RESPONSE_VERSION {
        return Err("Android web response has unsupported version".into());
    }

    let status = cursor.i32()?;
    let redirects = cursor.u32()?;
    if redirects > u32::try_from(WEB_FETCH_MAX_REDIRECTS).unwrap_or(u32::MAX) {
        return Err("Android web response redirect count exceeds policy".into());
    }
    let final_url = cursor.string(4096)?;
    let content_type = cursor.string(512)?;
    let body = cursor.bytes(WEB_FETCH_MAX_BYTES)?;
    let error = cursor.string(512)?;
    if !cursor.is_finished() {
        return Err("Android web response has trailing bytes".into());
    }

    if status == 0 {
        if error.trim().is_empty() || !body.is_empty() {
            return Err("Android web transport failure frame is non-canonical".into());
        }
    } else {
        if !(100..=599).contains(&status) || !error.is_empty() {
            return Err("Android web HTTP response frame is non-canonical".into());
        }
        if !(final_url.starts_with("http://") || final_url.starts_with("https://")) {
            return Err("Android web response final URL is invalid".into());
        }
    }

    Ok(AndroidWebResponse {
        status,
        redirects,
        final_url,
        content_type,
        body,
        error,
    })
}

fn is_textual_content_type(content_type: &str) -> bool {
    let media_type = content_type
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    media_type.starts_with("text/")
        || media_type.contains("json")
        || media_type.contains("xml")
        || media_type.contains("javascript")
        || media_type == "application/x-www-form-urlencoded"
}

struct WebResponseCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> WebResponseCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], String> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| "Android web response length overflow".to_owned())?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| "Android web response is truncated".to_owned())?;
        self.offset = end;
        Ok(value)
    }

    fn u16(&mut self) -> Result<u16, String> {
        let value = self.take(2)?;
        Ok(u16::from_le_bytes([value[0], value[1]]))
    }

    fn u32(&mut self) -> Result<u32, String> {
        let value = self.take(4)?;
        Ok(u32::from_le_bytes([
            value[0], value[1], value[2], value[3],
        ]))
    }

    fn i32(&mut self) -> Result<i32, String> {
        Ok(i32::from_le_bytes(self.u32()?.to_le_bytes()))
    }

    fn bytes(&mut self, max_len: usize) -> Result<Vec<u8>, String> {
        let len = usize::try_from(self.u32()?)
            .map_err(|_| "Android web response length does not fit usize".to_owned())?;
        if len > max_len {
            return Err("Android web response field exceeds policy limit".into());
        }
        Ok(self.take(len)?.to_vec())
    }

    fn string(&mut self, max_len: usize) -> Result<String, String> {
        let bytes = self.bytes(max_len)?;
        String::from_utf8(bytes).map_err(|_| "Android web response contains invalid UTF-8".into())
    }

    fn is_finished(&self) -> bool {
        self.offset == self.bytes.len()
    }
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

#[derive(Debug, Default, Clone, Copy)]
struct AndroidGovernedVerifier;

impl ActionVerifier for AndroidGovernedVerifier {
    fn verify(
        &mut self,
        descriptor: &CapabilityDescriptor,
        action: &TypedAction,
        output: &ActionOutput,
    ) -> ActionVerification {
        match descriptor.id.0.as_str() {
            "device.observe" => AndroidResourceVerifier.verify(descriptor, action, output),
            "web.fetch" => AndroidWebVerifier.verify(descriptor, action, output),
            _ => ActionVerification::Reject {
                reason: "Android governed verifier does not recognize capability".into(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeChatSessionStatus {
    Running,
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
    platform_web: Option<Arc<AndroidPlatformWebBridge>>,
    chat_session: Option<NativeChatSession>,
    chat_submit_in_progress: bool,
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
            platform_web: None,
            chat_session: None,
            chat_submit_in_progress: false,
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
    platform_web: Arc<AndroidPlatformWebBridge>,
) -> Result<(), String> {
    if asset_id.trim().is_empty() || version == 0 || context_limit == 0 {
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

    let mut guard = lock_state();
    guard.chat_model = Some(Arc::new(NativeChatModel {
        asset_id: asset_id.to_owned(),
        version,
        activation,
        resolver,
        tokenizer,
        context_limit,
    }));
    guard.platform_web = Some(platform_web);
    Ok(())
}

fn submit_chat(user_message: &str, max_new_tokens: usize) -> Result<u64, String> {
    if max_new_tokens == 0 {
        return Err("max_new_tokens must be greater than zero".into());
    }

    let (model, history, prior_failure, resources, platform_web) = {
        let mut guard = lock_state();
        if guard.chat_submit_in_progress
            || guard
                .chat_session
                .as_ref()
                .is_some_and(|session| session.status == NativeChatSessionStatus::Running)
        {
            return Err("another native chat request is still running".into());
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
            guard.platform_web.clone(),
        )
    };

    let result = submit_chat_reserved(
        &model,
        &history,
        prior_failure,
        resources,
        platform_web,
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

fn execute_android_verified_actions(
    conversation: &mut SovereignConversationState,
    model: &NativeChatModel,
    task_id: u64,
    user_message: &str,
    prompt_limit: usize,
    resources: ResourceSnapshot,
    platform_web: Option<Arc<AndroidPlatformWebBridge>>,
    action_plan: &AssistantActionPlan,
) -> Result<(), String> {
    if action_plan.graph.actions.iter().any(|node| {
        !matches!(node.capability.0.as_str(), "device.observe" | "web.fetch")
    }) {
        conversation
            .record_action_planner_status(task_id, "unsupported")
            .map_err(|error| format!("record unsupported action plan: {error:?}"))?;
        return Err("model requested a capability without an Android production adapter".into());
    }

    conversation
        .install_action_plan(task_id, action_plan)
        .map_err(|error| format!("install model-grounded action plan: {error:?}"))?;

    let needs_device = action_plan
        .graph
        .actions
        .iter()
        .any(|node| node.capability.0 == "device.observe");
    let needs_web = action_plan
        .graph
        .actions
        .iter()
        .any(|node| node.capability.0 == "web.fetch");

    let mut registry = CapabilityRegistry::new();
    if needs_device {
        registry
            .register(
                CapabilityDescriptor::new(
                    CapabilityId("device.observe".into()),
                    1,
                    CapabilityDomain::Device,
                    SideEffectClass::ReadOnly,
                )
                .map_err(|error| format!("build device.observe descriptor: {error:?}"))?,
            )
            .map_err(|error| format!("register device.observe capability: {error:?}"))?;
    }
    if needs_web {
        let mut descriptor = CapabilityDescriptor::new(
            CapabilityId("web.fetch".into()),
            1,
            CapabilityDomain::Web,
            SideEffectClass::ReadOnly,
        )
        .map_err(|error| format!("build web.fetch descriptor: {error:?}"))?;
        descriptor.required_scopes.push(
            AuthorityScope::new("network.read")
                .map_err(|error| format!("build network.read scope: {error:?}"))?,
        );
        registry
            .register(descriptor)
            .map_err(|error| format!("register web.fetch capability: {error:?}"))?;
    }

    let mut fabric = ActionFabric::new(registry);
    if needs_device {
        fabric
            .register_adapter(
                CapabilityId("device.observe".into()),
                AndroidResourceAdapter {
                    snapshot: resources,
                },
            )
            .map_err(|error| format!("register Android resource adapter: {error:?}"))?;
    }
    if needs_web {
        let bridge = platform_web
            .ok_or_else(|| "Android web transport is unavailable".to_owned())?;
        fabric
            .register_adapter(
                CapabilityId("web.fetch".into()),
                AndroidWebFetchAdapter { bridge },
            )
            .map_err(|error| format!("register Android web adapter: {error:?}"))?;
    }

    let task = conversation
        .cognition()
        .state()
        .tasks
        .get(&task_id)
        .cloned()
        .ok_or_else(|| "model-grounded cognitive task disappeared".to_owned())?;
    let mut authority = AuthorityGrant::new();
    if needs_web {
        authority = authority.with_scope(
            AuthorityScope::new("network.read")
                .map_err(|error| format!("grant network.read scope: {error:?}"))?,
        );
    }
    let run = execute_verified_assistant_plan(
        &mut fabric,
        &task,
        action_plan,
        &authority,
        &mut AndroidGovernedVerifier,
        user_message,
    )
    .map_err(|error| format!("execute verified Android action plan: {error:?}"))?;

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
    Ok(())
}

fn submit_chat_reserved(
    model: &Arc<NativeChatModel>,
    history: &[ntd_runtime::ConversationTurn],
    prior_failure: bool,
    resources: ResourceSnapshot,
    platform_web: Option<Arc<AndroidPlatformWebBridge>>,
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
            .is_some_and(|session| session.status == NativeChatSessionStatus::Running)
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

    let planner_outcome = match governed_device_surfaces(user_message) {
        Some(surfaces) => run_constrained_device_planner(model, user_message, &surfaces)?,
        None => run_native_action_planner(model, user_message)?,
    };
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
            execute_android_verified_actions(
                &mut conversation,
                model,
                task_id,
                user_message,
                prompt_limit,
                resources,
                platform_web,
                &action_plan,
            )?;
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
            .is_some_and(|session| session.status == NativeChatSessionStatus::Running)
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
        status: NativeChatSessionStatus::Running,
    });
    Ok(request_id)
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
            session.request_id == request_id && session.status == NativeChatSessionStatus::Running
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

fn next_chat_event(request_id: u64) -> Result<Vec<u8>, String> {
    if let Some(event) = terminal_chat_event(request_id) {
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
    guard.chat_session = Some(NativeChatSession {
        request_id,
        task_id,
        cancel: Arc::new(AtomicBool::new(false)),
        status: NativeChatSessionStatus::Running,
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
    verify_key: JByteArray<'_>,
    context_limit: jint,
    platform_web: JObject<'_>,
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
    let platform_web = match AndroidPlatformWebBridge::new(&env, platform_web) {
        Ok(bridge) => Arc::new(bridge),
        Err(_) => return 0,
    };

    u8::from(
        open_chat_model(
            &asset_id,
            version,
            &capsule_path,
            &shard_root,
            &verify_key,
            context_limit,
            platform_web,
        )
        .is_ok(),
    )
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
    submit_chat(&prompt, max_new_tokens)
        .ok()
        .and_then(|request_id| i64::try_from(request_id).ok())
        .unwrap_or(-1)
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

fn android_web_fetch_probe(url: &str, expected_sha256: &str) -> Result<String, String> {
    if expected_sha256.len() != 64
        || !expected_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err("expected web fixture SHA-256 is invalid".into());
    }
    let bridge = lock_state()
        .platform_web
        .clone()
        .ok_or_else(|| "Android web transport is not installed".to_owned())?;

    let canonical = format!("{NATIVE_ACTION_PROTOCOL_V1}\n1|web.fetch|{url}\nEND");
    let action_plan = match parse_native_action_plan(&canonical)
        .map_err(|error| format!("parse web fetch probe plan: {error:?}"))?
    {
        AssistantPlanDecision::Actions(plan) => plan,
        AssistantPlanDecision::Direct => return Err("web fetch probe became DIRECT".into()),
    };

    let mut descriptor = CapabilityDescriptor::new(
        CapabilityId("web.fetch".into()),
        1,
        CapabilityDomain::Web,
        SideEffectClass::ReadOnly,
    )
    .map_err(|error| format!("build web.fetch descriptor: {error:?}"))?;
    descriptor.required_scopes.push(
        AuthorityScope::new("network.read")
            .map_err(|error| format!("build network.read scope: {error:?}"))?,
    );
    let mut registry = CapabilityRegistry::new();
    registry
        .register(descriptor)
        .map_err(|error| format!("register web.fetch probe capability: {error:?}"))?;
    let mut fabric = ActionFabric::new(registry);
    fabric
        .register_adapter(
            CapabilityId("web.fetch".into()),
            AndroidWebFetchAdapter { bridge },
        )
        .map_err(|error| format!("register web.fetch probe adapter: {error:?}"))?;
    let plan_id = fabric
        .prepare_plan(1, &action_plan.graph, action_plan.payloads)
        .map_err(|error| format!("prepare web.fetch probe plan: {error:?}"))?;
    let authority = AuthorityGrant::new().with_scope(
        AuthorityScope::new("network.read")
            .map_err(|error| format!("grant network.read probe scope: {error:?}"))?,
    );
    let report = fabric
        .execute_next(plan_id, &authority, &mut AndroidWebVerifier)
        .map_err(|error| format!("execute web.fetch probe: {error:?}"))?;
    if report.plan_status != ntd_runtime::ActionPlanStatus::Completed
        || report.action_status != Some(ntd_runtime::ActionStatus::Committed)
    {
        return Err(format!(
            "web.fetch probe did not commit: plan={:?} action={:?}",
            report.plan_status, report.action_status
        ));
    }

    let plan = fabric
        .state()
        .plans
        .get(&plan_id.0)
        .ok_or_else(|| "web.fetch probe plan disappeared".to_owned())?;
    let output = plan
        .actions
        .first()
        .and_then(|action| action.output.as_ref())
        .ok_or_else(|| "web.fetch probe has no verified output".to_owned())?;
    let evidence = output
        .evidence
        .iter()
        .filter_map(|item| item.split_once('='))
        .collect::<BTreeMap<_, _>>();
    if evidence.get("sha256") != Some(&expected_sha256) {
        return Err(format!(
            "web.fetch probe digest mismatch: expected={expected_sha256} actual={}",
            evidence.get("sha256").copied().unwrap_or("missing")
        ));
    }

    Ok(format!(
        "web_fetch=PASS\nweb_fetch_status={}\nweb_fetch_bytes={}\nweb_fetch_sha256={}\nweb_fetch_redirects={}\n",
        evidence.get("http_status").copied().unwrap_or("missing"),
        evidence.get("bytes").copied().unwrap_or("missing"),
        evidence.get("sha256").copied().unwrap_or("missing"),
        evidence.get("redirects").copied().unwrap_or("missing"),
    ))
}

#[allow(unsafe_code)]
#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdNativeRuntimeHost_nativeWebFetchProbe(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    url: JString<'_>,
    expected_sha256: JString<'_>,
) -> jbyteArray {
    let Some(url) = java_string(&mut env, &url) else {
        return java_bytes(&env, b"web_fetch=failed\nerror=invalid URL\n");
    };
    let Some(expected_sha256) = java_string(&mut env, &expected_sha256) else {
        return java_bytes(
            &env,
            b"web_fetch=failed\nerror=invalid expected digest\n",
        );
    };

    match android_web_fetch_probe(&url, &expected_sha256) {
        Ok(result) => java_bytes(&env, result.as_bytes()),
        Err(error) => java_bytes(
            &env,
            format!("web_fetch=failed\nerror={error}\n").as_bytes(),
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
