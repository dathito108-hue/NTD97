#![deny(unsafe_code)]
#![allow(non_snake_case)]

use std::{
    collections::BTreeMap,
    fs,
    ptr::null_mut,
    sync::{Mutex, OnceLock},
};

use jni::{
    objects::{JByteArray, JClass, JString},
    sys::{jboolean, jbyteArray, jint, jlong},
    JNIEnv,
};
use ntd_assimilation::{
    activate_thin_generative_capsule, verify_native_package_with_shards, AssetKind,
    FileBackedTensorResolver, FileTensorShardStore, NativePackage,
};
use ntd_capsule::{sha256, NativeTokenizerDescriptor, NativeTokenizerModel};
use ntd_mobile_shell::{
    decode_mobile_continuity_bundle, encode_mobile_continuity_bundle,
    restore_mobile_continuity_bundle, AvatarController, AvatarSurface, LipSyncState,
    MobileContinuityBundle, MobileContinuityState, WakeReason,
};
use ntd_runtime::{
    CpuReferenceProvider, DistributionKind, GenerationConfig, GraphGenerator, LlamaSpmConfig,
    LlamaSpmTokenizer, ResourceSnapshot, SamplingMode, ThermalState,
};
use ntd_validation::{
    encode_physical_evidence, run_logical_continuity_soak, run_native_validation_workload,
    DeviceEvidence, EvidenceClass, PhysicalEvidenceRecord,
};

const BRIDGE_PROTOCOL_VERSION: u8 = 1;
const REAL_MODEL_VERIFY_KEY: [u8; 32] = [
    0xe1, 0xb7, 0x1a, 0xbf, 0xd3, 0x23, 0x28, 0x04, 0x26, 0x1e, 0x42, 0x3f, 0x36, 0x55, 0x6f,
    0x6b, 0x41, 0x85, 0xbe, 0xd4, 0x1f, 0xdf, 0xd0, 0x0d, 0x76, 0x9c, 0xe1, 0x5a, 0x39, 0x4f,
    0x43, 0xce,
];
const REAL_MODEL_OUTPUT_SHA256: [u8; 32] = [
    0x59, 0x4a, 0x91, 0x1e, 0xbb, 0x2e, 0xcf, 0xeb, 0x60, 0x89, 0x19, 0xbf, 0x15, 0x78, 0x87,
    0xe8, 0x2d, 0x00, 0x90, 0x50, 0x7f, 0xa1, 0x87, 0xd4, 0x5b, 0x2b, 0x7e, 0x23, 0xe5, 0xe8,
    0xf5, 0x83,
];
const REAL_MODEL_CONTEXT: usize = 128;
const REAL_MODEL_TEXT_BYTES: usize = 322;

struct NativeState {
    bundle: Option<MobileContinuityBundle>,
    resources: ResourceSnapshot,
    input_peak_milli: u16,
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

fn real_model_tokenizer(
    descriptor: &NativeTokenizerDescriptor,
) -> Result<LlamaSpmTokenizer, String> {
    let NativeTokenizerModel::LlamaSpm {
        score_bits,
        token_types,
        add_space_prefix,
        add_bos_token,
        add_eos_token,
    } = &descriptor.model
    else {
        return Err("real model tokenizer is not native LLaMA SPM".into());
    };

    LlamaSpmTokenizer::new(
        descriptor.tokens.clone(),
        score_bits.clone(),
        token_types.clone(),
        LlamaSpmConfig {
            bos_token: descriptor.bos_token,
            eos_token: descriptor.eos_token,
            unknown_token: descriptor.unknown_token,
            add_space_prefix: *add_space_prefix,
            add_bos_token: *add_bos_token,
            add_eos_token: *add_eos_token,
        },
    )
    .map_err(|error| format!("native tokenizer: {error:?}"))
}

fn real_model_probe(capsule_path: &str, shard_root: &str) -> Result<Vec<u8>, String> {
    let native_capsule =
        fs::read(capsule_path).map_err(|error| format!("read NCC97 capsule: {error}"))?;
    let capsule_hash = sha256(&native_capsule);
    let package = NativePackage {
        asset_id: "model.ntd97.stories260k".into(),
        version: 1,
        kind: AssetKind::Intelligence,
        native_capsule,
        capsule_hash,
        capability: None,
    };
    let shard_store =
        FileTensorShardStore::open(shard_root).map_err(|error| format!("open shards: {error:?}"))?;

    verify_native_package_with_shards(&package, &REAL_MODEL_VERIFY_KEY, &shard_store)
        .map_err(|error| format!("verify signed native package: {error:?}"))?;
    let activation = activate_thin_generative_capsule(&package.native_capsule, &shard_store)
        .map_err(|error| format!("activate native package: {error:?}"))?;
    if activation.manifest.vocabulary_size != 512 {
        return Err(format!(
            "real model vocabulary drift: {}",
            activation.manifest.vocabulary_size
        ));
    }

    let tokenizer = real_model_tokenizer(&activation.tokenizer)?;
    let bos = tokenizer
        .bos_token()
        .ok_or_else(|| "real model is missing BOS".to_owned())?;
    if bos != 1 {
        return Err(format!("real model BOS drift: {bos}"));
    }

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
    .map_err(|error| format!("build native generator: {error:?}"))?;

    let generated = generator
        .generate_tokens_with_resolver(
            &resolver,
            &[bos],
            GenerationConfig {
                max_new_tokens: REAL_MODEL_CONTEXT,
                context_limit: REAL_MODEL_CONTEXT,
                eos_token: Some(bos),
                distribution: DistributionKind::Logits,
                sampling: SamplingMode::Greedy,
            },
        )
        .map_err(|error| format!("native generation: {error:?}"))?;
    if generated.generated_tokens.len() != REAL_MODEL_CONTEXT {
        return Err(format!(
            "generated token count drift: expected {}, got {}",
            REAL_MODEL_CONTEXT,
            generated.generated_tokens.len()
        ));
    }

    let text = tokenizer
        .decode_after(Some(bos), &generated.generated_tokens, true)
        .map_err(|error| format!("native decode: {error:?}"))?;
    let output_hash = sha256(text.as_bytes());
    if text.len() != REAL_MODEL_TEXT_BYTES || output_hash != REAL_MODEL_OUTPUT_SHA256 {
        return Err(format!(
            "generated output drift: bytes={} sha256={}",
            text.len(),
            digest_hex(&output_hash)
        ));
    }

    Ok(format!(
        "real_model_package=ok\nreal_model_generation=ok\ngenerated_tokens={}\ngenerated_bytes={}\ngenerated_sha256={}\ncapsule_sha256={}\n",
        generated.generated_tokens.len(),
        text.len(),
        digest_hex(&output_hash),
        digest_hex(&capsule_hash),
    )
    .into_bytes())
}

fn digest_hex(digest: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("write digest");
    }
    output
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

#[allow(unsafe_code)]
#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdNativeRuntimeHost_nativeProbeRealModel(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    capsule_path: JString<'_>,
    shard_root: JString<'_>,
) -> jbyteArray {
    let Some(capsule_path) = java_string(&mut env, &capsule_path) else {
        return java_bytes(&env, b"real_model_generation=failed\nreason=invalid capsule path\n");
    };
    let Some(shard_root) = java_string(&mut env, &shard_root) else {
        return java_bytes(&env, b"real_model_generation=failed\nreason=invalid shard root\n");
    };

    let result = match real_model_probe(&capsule_path, &shard_root) {
        Ok(result) => result,
        Err(error) => format!("real_model_generation=failed\nreason={error}\n").into_bytes(),
    };
    java_bytes(&env, &result)
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
