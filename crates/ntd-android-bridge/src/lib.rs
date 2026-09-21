#![forbid(unsafe_code)]
#![allow(non_snake_case)]

use std::{
    ptr::null_mut,
    sync::{Mutex, OnceLock},
};

use jni::{
    objects::{JByteArray, JClass, JString},
    sys::{jboolean, jbyteArray, jint, jlong},
    JNIEnv,
};
use ntd_mobile_shell::{
    decode_mobile_continuity_bundle, encode_mobile_continuity_bundle,
    restore_mobile_continuity_bundle, AvatarController, AvatarSurface, LipSyncState,
    MobileContinuityBundle, MobileContinuityState, WakeReason,
};
use ntd_runtime::{ResourceSnapshot, ThermalState};

const BRIDGE_PROTOCOL_VERSION: u8 = 1;

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
    state().lock().unwrap_or_else(|poisoned| poisoned.into_inner())
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
    if current.state != MobileContinuityState::WaitingApproval
        || current.pending_approval.is_none()
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

#[no_mangle]
pub extern "system" fn Java_ai_ntd97_mobile_NtdNativeRuntimeHost_nativePullSpeakerPcm(
    env: JNIEnv<'_>,
    _class: JClass<'_>,
    _max_samples: jint,
    _sample_rate_hz: jint,
) -> jbyteArray {
    java_bytes(&env, &[])
}
