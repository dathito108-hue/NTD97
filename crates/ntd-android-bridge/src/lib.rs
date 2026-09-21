#![forbid(unsafe_code)]

use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use jni::{
    objects::{GlobalRef, JClass, JObject, JString, JValue},
    sys::{jboolean, jint, jlong, jstring},
    JNIEnv, JavaVM,
};
use ntd_core::{CapabilityId, SideEffectClass};
use ntd_platform::{
    CheckpointStore, ContinuityPhase, ContinuitySupervisor, EmbodimentController,
    PlatformConstraints, WakeReason,
};
use ntd_runtime::{
    ActionFabric, ActionOutput, ActionPlanStatus, ActionValue, ActionVerification, ActionVerifier,
    AdapterResult, AuthorityGrant, AuthorityScope, CapabilityAdapter, CapabilityDescriptor,
    CapabilityDomain, CapabilityRegistry, CognitiveIdentity, CognitiveRuntime, ResourceSnapshot,
    ThermalState, TypedAction,
};
use serde_json::{json, Value};

const BRIDGE_VERSION: i32 = 1;

struct FileCheckpointStore {
    path: PathBuf,
}

impl FileCheckpointStore {
    fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

impl CheckpointStore for FileCheckpointStore {
    fn commit(&mut self, bytes: &[u8]) -> Result<(), String> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        let temp = self.path.with_extension("tmp");
        fs::write(&temp, bytes).map_err(|error| error.to_string())?;
        fs::rename(&temp, &self.path).map_err(|error| error.to_string())
    }

    fn load(&mut self) -> Result<Option<Vec<u8>>, String> {
        if !self.path.exists() {
            return Ok(None);
        }
        fs::read(&self.path)
            .map(Some)
            .map_err(|error| error.to_string())
    }
}

struct JavaCapabilityAdapter {
    vm: Arc<JavaVM>,
    host: GlobalRef,
}

impl JavaCapabilityAdapter {
    fn call_host(
        &self,
        method: &str,
        signature: &str,
        action_id: ntd_runtime::ActionId,
        action: &TypedAction,
        token: Option<&[u8]>,
    ) -> Result<String, String> {
        let mut env = self
            .vm
            .attach_current_thread()
            .map_err(|error| error.to_string())?;
        let request = action_request_json(action);
        let request_string = env
            .new_string(request)
            .map_err(|error| error.to_string())?;
        let request_object = JObject::from(request_string);

        let token_string = env
            .new_string(token.map(encode_hex).unwrap_or_default())
            .map_err(|error| error.to_string())?;
        let token_object = JObject::from(token_string);

        let args = match token {
            Some(_) => vec![
                JValue::Long(action_id.0 as jlong),
                JValue::Object(&request_object),
                JValue::Object(&token_object),
            ],
            None => vec![
                JValue::Long(action_id.0 as jlong),
                JValue::Object(&request_object),
            ],
        };

        let value = env
            .call_method(self.host.as_obj(), method, signature, &args)
            .map_err(|error| error.to_string())?;
        let object = value.l().map_err(|error| error.to_string())?;
        let string = JString::from(object);
        env.get_string(&string)
            .map(|value| value.into())
            .map_err(|error| error.to_string())
    }
}

impl CapabilityAdapter for JavaCapabilityAdapter {
    fn execute(
        &mut self,
        action_id: ntd_runtime::ActionId,
        action: &TypedAction,
    ) -> Result<AdapterResult, String> {
        let response = self.call_host(
            "executeAction",
            "(JLjava/lang/String;)Ljava/lang/String;",
            action_id,
            action,
            None,
        )?;
        parse_host_response(&response)
    }

    fn resume(
        &mut self,
        action_id: ntd_runtime::ActionId,
        action: &TypedAction,
        resume_token: &[u8],
    ) -> Result<AdapterResult, String> {
        let response = self.call_host(
            "resumeAction",
            "(JLjava/lang/String;Ljava/lang/String;)Ljava/lang/String;",
            action_id,
            action,
            Some(resume_token),
        )?;
        parse_host_response(&response)
    }

    fn rollback(
        &mut self,
        action_id: ntd_runtime::ActionId,
        action: &TypedAction,
        rollback_token: &[u8],
    ) -> Result<(), String> {
        let response = self.call_host(
            "rollbackAction",
            "(JLjava/lang/String;Ljava/lang/String;)Ljava/lang/String;",
            action_id,
            action,
            Some(rollback_token),
        )?;
        let value: Value = serde_json::from_str(&response).map_err(|error| error.to_string())?;
        if value
            .get("status")
            .and_then(Value::as_str)
            .is_some_and(|status| status == "rolled_back")
        {
            Ok(())
        } else {
            Err(value
                .get("reason")
                .and_then(Value::as_str)
                .unwrap_or("Android rollback was not verified")
                .to_owned())
        }
    }
}

struct PlatformVerifier;

impl ActionVerifier for PlatformVerifier {
    fn verify(
        &mut self,
        descriptor: &CapabilityDescriptor,
        _action: &TypedAction,
        output: &ActionOutput,
    ) -> ActionVerification {
        if output.summary.trim().is_empty() {
            return ActionVerification::Reject {
                reason: "Android capability returned an empty result".into(),
            };
        }

        if descriptor.side_effect != SideEffectClass::ReadOnly && output.evidence.is_empty() {
            return ActionVerification::Retry {
                reason: "side effect requires observable Android evidence".into(),
            };
        }

        ActionVerification::Accept
    }
}

fn registry() -> Result<CapabilityRegistry, String> {
    let mut registry = CapabilityRegistry::new();

    register(
        &mut registry,
        "web.search",
        CapabilityDomain::Web,
        SideEffectClass::ReadOnly,
        "network.read",
        true,
        false,
    )?;
    register(
        &mut registry,
        "web.fetch",
        CapabilityDomain::Web,
        SideEffectClass::ReadOnly,
        "network.read",
        true,
        false,
    )?;
    register(
        &mut registry,
        "browser.observe",
        CapabilityDomain::Browser,
        SideEffectClass::ReadOnly,
        "browser.observe",
        true,
        false,
    )?;
    register(
        &mut registry,
        "browser.interact",
        CapabilityDomain::Browser,
        SideEffectClass::ExternalWrite,
        "browser.interact",
        true,
        true,
    )?;
    register(
        &mut registry,
        "file.read",
        CapabilityDomain::File,
        SideEffectClass::ReadOnly,
        "files.read",
        false,
        false,
    )?;
    register(
        &mut registry,
        "file.write",
        CapabilityDomain::File,
        SideEffectClass::ExternalWrite,
        "files.write",
        false,
        true,
    )?;
    register(
        &mut registry,
        "device.observe",
        CapabilityDomain::Device,
        SideEffectClass::ReadOnly,
        "device.observe",
        false,
        false,
    )?;
    register(
        &mut registry,
        "device.interact",
        CapabilityDomain::Device,
        SideEffectClass::ExternalWrite,
        "device.interact",
        false,
        true,
    )?;
    register(
        &mut registry,
        "app.action",
        CapabilityDomain::App,
        SideEffectClass::ExternalWrite,
        "android.interact",
        false,
        true,
    )?;

    Ok(registry)
}

fn register(
    registry: &mut CapabilityRegistry,
    id: &str,
    domain: CapabilityDomain,
    side_effect: SideEffectClass,
    scope: &str,
    resumable: bool,
    rollback_supported: bool,
) -> Result<(), String> {
    let mut descriptor =
        CapabilityDescriptor::new(CapabilityId(id.into()), 1, domain, side_effect)
            .map_err(|error| format!("{error:?}"))?;
    descriptor.required_scopes =
        vec![AuthorityScope::new(scope).map_err(|error| format!("{error:?}"))?];
    descriptor.resumable = resumable;
    descriptor.rollback_supported = rollback_supported;
    registry
        .register(descriptor)
        .map_err(|error| format!("{error:?}"))
}

fn create_state(path: &str, identity_hex: &str, root_hex: &str) -> Result<Value, String> {
    if Path::new(path).exists() {
        return Ok(json!({"status":"exists"}));
    }

    let identity = parse_fixed_hex::<16>(identity_hex)?;
    let root = parse_fixed_hex::<32>(root_hex)?;
    let cognition = CognitiveRuntime::new(CognitiveIdentity(identity));
    let actions = ActionFabric::new(registry()?);
    let mut supervisor = ContinuitySupervisor::new(root, cognition, actions);
    let mut store = FileCheckpointStore::new(path);
    let generation = supervisor
        .checkpoint_to(&mut store)
        .map_err(|error| format!("{error:?}"))?;

    Ok(json!({
        "status":"created",
        "generation":generation,
    }))
}

fn restore(path: &str) -> Result<ContinuitySupervisor, String> {
    let mut store = FileCheckpointStore::new(path);
    ContinuitySupervisor::restore(registry()?, &mut store)
        .map_err(|error| format!("{error:?}"))
}

fn persist(path: &str, supervisor: &mut ContinuitySupervisor) -> Result<u64, String> {
    let mut store = FileCheckpointStore::new(path);
    supervisor
        .checkpoint_to(&mut store)
        .map_err(|error| format!("{error:?}"))
}

fn attach_host(
    env: &mut JNIEnv<'_>,
    supervisor: &mut ContinuitySupervisor,
    host: JObject<'_>,
) -> Result<(), String> {
    let vm = Arc::new(env.get_java_vm().map_err(|error| error.to_string())?);
    let global = env
        .new_global_ref(host)
        .map_err(|error| error.to_string())?;
    let capability_ids = supervisor
        .actions()
        .registry()
        .descriptors()
        .map(|descriptor| descriptor.id.clone())
        .collect::<Vec<_>>();

    for capability in capability_ids {
        supervisor
            .actions_mut()
            .register_adapter(
                capability,
                JavaCapabilityAdapter {
                    vm: Arc::clone(&vm),
                    host: global.clone(),
                },
            )
            .map_err(|error| format!("{error:?}"))?;
    }

    Ok(())
}

fn parse_host_response(response: &str) -> Result<AdapterResult, String> {
    let value: Value = serde_json::from_str(response).map_err(|error| error.to_string())?;
    let status = value
        .get("status")
        .and_then(Value::as_str)
        .ok_or_else(|| "Android capability response missing status".to_owned())?;

    match status {
        "completed" => {
            let summary = value
                .get("summary")
                .and_then(Value::as_str)
                .unwrap_or("completed")
                .to_owned();
            let text = value
                .get("value")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let evidence = value
                .get("evidence")
                .and_then(Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let rollback_token = value
                .get("rollbackTokenHex")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(decode_hex)
                .transpose()?;

            Ok(AdapterResult::Completed {
                output: ActionOutput {
                    summary,
                    value: ActionValue::Text(text),
                    evidence,
                },
                rollback_token,
            })
        }
        "suspended" => Ok(AdapterResult::Suspended {
            resume_token: decode_hex(
                value
                    .get("resumeTokenHex")
                    .and_then(Value::as_str)
                    .ok_or_else(|| "suspended response missing resume token".to_owned())?,
            )?,
            note: value
                .get("note")
                .and_then(Value::as_str)
                .unwrap_or("Android capability suspended")
                .to_owned(),
        }),
        "retryable" => Ok(AdapterResult::Retryable {
            reason: value
                .get("reason")
                .and_then(Value::as_str)
                .unwrap_or("Android capability requested retry")
                .to_owned(),
            resume_token: value
                .get("resumeTokenHex")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(decode_hex)
                .transpose()?,
        }),
        "failed" => Err(value
            .get("reason")
            .and_then(Value::as_str)
            .unwrap_or("Android capability failed")
            .to_owned()),
        other => Err(format!("unknown Android capability status: {other}")),
    }
}

fn action_request_json(action: &TypedAction) -> String {
    let value = match action {
        TypedAction::WebSearch { query, max_results } => json!({
            "type":"web_search",
            "query":query,
            "maxResults":max_results,
        }),
        TypedAction::WebFetch { url } => json!({
            "type":"web_fetch",
            "url":url,
        }),
        TypedAction::BrowserObserve { target } => json!({
            "type":"browser_observe",
            "target":target,
        }),
        TypedAction::BrowserInteract {
            target,
            operation,
            value,
        } => json!({
            "type":"browser_interact",
            "target":target,
            "operation":operation,
            "value":value,
        }),
        TypedAction::FileRead { path } => json!({
            "type":"file_read",
            "path":path,
        }),
        TypedAction::FileWrite { path, bytes } => json!({
            "type":"file_write",
            "path":path,
            "bytesHex":encode_hex(bytes),
        }),
        TypedAction::DeviceObserve { surface } => json!({
            "type":"device_observe",
            "surface":surface,
        }),
        TypedAction::DeviceInteract {
            surface,
            operation,
            argument,
        } => json!({
            "type":"device_interact",
            "surface":surface,
            "operation":operation,
            "argument":argument,
        }),
        TypedAction::AppAction {
            app,
            action,
            payload,
        } => json!({
            "type":"app_action",
            "app":app,
            "action":action,
            "payloadHex":encode_hex(payload),
        }),
        TypedAction::Custom { type_name, payload } => json!({
            "type":"custom",
            "typeName":type_name,
            "payloadHex":encode_hex(payload),
        }),
    };
    value.to_string()
}

fn wake_reason(value: jint) -> Result<WakeReason, String> {
    match value {
        1 => Ok(WakeReason::UserLaunch),
        2 => Ok(WakeReason::ForegroundService),
        3 => Ok(WakeReason::Scheduled),
        4 => Ok(WakeReason::Retry),
        5 => Ok(WakeReason::Boot),
        6 => Ok(WakeReason::Connectivity),
        7 => Ok(WakeReason::Approval),
        8 => Ok(WakeReason::Notification),
        _ => Err("invalid wake reason".into()),
    }
}

fn thermal_state(value: jint) -> Result<ThermalState, String> {
    match value {
        0 => Ok(ThermalState::Nominal),
        1 => Ok(ThermalState::Warm),
        2 => Ok(ThermalState::Hot),
        3 => Ok(ThermalState::Critical),
        _ => Err("invalid thermal state".into()),
    }
}

fn progress_permille(supervisor: &ContinuitySupervisor) -> u16 {
    let Some(plan) = supervisor
        .actions()
        .state()
        .plans
        .values()
        .find(|plan| {
            !matches!(
                plan.status,
                ActionPlanStatus::Completed
                    | ActionPlanStatus::Failed
                    | ActionPlanStatus::RolledBack
            )
        })
    else {
        return if supervisor.platform().phase == ContinuityPhase::Completed {
            1000
        } else {
            0
        };
    };

    if plan.actions.is_empty() {
        return 0;
    }
    let numerator = plan.cursor.saturating_mul(1000);
    u16::try_from(numerator / plan.actions.len())
        .unwrap_or(1000)
        .min(1000)
}

fn authority_from(
    scopes_csv: &str,
    allow_write: bool,
    allow_irreversible: bool,
) -> Result<AuthorityGrant, String> {
    let mut grant = AuthorityGrant::new();
    for scope in scopes_csv
        .split(',')
        .map(str::trim)
        .filter(|scope| !scope.is_empty())
    {
        grant = grant.with_scope(
            AuthorityScope::new(scope).map_err(|error| format!("{error:?}"))?,
        );
    }
    grant.allow_external_write = allow_write;
    grant.allow_irreversible = allow_irreversible;
    Ok(grant)
}

fn pending_approval_json(supervisor: &ContinuitySupervisor) -> Value {
    match &supervisor.platform().pending_approval {
        Some(approval) => json!({
            "id":approval.id,
            "planId":approval.plan_id.0,
            "actionId":approval.action_id.0,
            "requiredScopes":approval.required_scopes,
            "externalWrite":approval.external_write,
            "irreversible":approval.irreversible,
        }),
        None => Value::Null,
    }
}

fn snapshot_json(supervisor: &ContinuitySupervisor) -> Value {
    json!({
        "phase":format!("{:?}", supervisor.platform().phase),
        "tick":supervisor.platform().logical_tick,
        "generation":supervisor.platform().checkpoint_generation,
        "bootCount":supervisor.platform().boot_count,
        "pendingApproval":pending_approval_json(supervisor),
        "nextPlanId":supervisor.actions().state().plans.values().find(|plan| {
            !matches!(
                plan.status,
                ActionPlanStatus::Completed
                    | ActionPlanStatus::Failed
                    | ActionPlanStatus::RolledBack
            )
        }).map(|plan| plan.id.0),
    })
}

fn frame_json(
    supervisor: &ContinuitySupervisor,
    battery_percent: u8,
    charging: bool,
    thermal: ThermalState,
    visible: bool,
) -> Value {
    let resources = ResourceSnapshot {
        available_ram_bytes: 0,
        battery_percent,
        charging,
        thermal,
        latency_budget_ms: 100,
    };
    let mut controller = EmbodimentController::new(resources, visible);
    let frame = controller.sync_continuity(
        supervisor.platform().phase,
        resources,
        visible,
        progress_permille(supervisor),
    );

    json!({
        "mode":format!("{:?}", frame.mode),
        "expression":format!("{:?}", frame.expression),
        "gesture":format!("{:?}", frame.gesture),
        "voice":format!("{:?}", frame.voice),
        "viseme":format!("{:?}", frame.viseme),
        "lipIntensity":frame.lip_intensity,
        "gazeX":frame.gaze.x_milli,
        "gazeY":frame.gaze.y_milli,
        "gazeZ":frame.gaze.z_milli,
        "targetFps":frame.render.target_fps,
        "renderQuality":format!("{:?}", frame.render.quality),
        "shadows":frame.render.shadows,
        "secondaryMotion":frame.render.secondary_motion,
        "progress":frame.progress_permille,
    })
}

fn parse_fixed_hex<const N: usize>(value: &str) -> Result<[u8; N], String> {
    let decoded = decode_hex(value)?;
    if decoded.len() != N {
        return Err(format!("expected {N} bytes"));
    }
    let mut out = [0u8; N];
    out.copy_from_slice(&decoded);
    Ok(out)
}

fn decode_hex(value: &str) -> Result<Vec<u8>, String> {
    if value.len() % 2 != 0 {
        return Err("hex string has odd length".into());
    }

    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for pair in bytes.chunks_exact(2) {
        let high = hex_nibble(pair[0])?;
        let low = hex_nibble(pair[1])?;
        out.push((high << 4) | low);
    }
    Ok(out)
}

fn hex_nibble(value: u8) -> Result<u8, String> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        b'A'..=b'F' => Ok(value - b'A' + 10),
        _ => Err("invalid hex digit".into()),
    }
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        out.push(HEX[usize::from(byte >> 4)] as char);
        out.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    out
}

fn get_string(env: &mut JNIEnv<'_>, value: JString<'_>) -> Result<String, String> {
    env.get_string(&value)
        .map(|value| value.into())
        .map_err(|error| error.to_string())
}

fn java_response(env: &mut JNIEnv<'_>, result: Result<Value, String>) -> jstring {
    let value = match result {
        Ok(value) => value,
        Err(error) => json!({"status":"error","reason":error}),
    };
    env.new_string(value.to_string())
        .map(|value| value.into_raw())
        .unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub extern "system" fn Java_com_ntd97_app_NativeBridge_nativeVersion(
    _env: JNIEnv<'_>,
    _class: JClass<'_>,
) -> jint {
    BRIDGE_VERSION
}

#[no_mangle]
pub extern "system" fn Java_com_ntd97_app_NativeBridge_nativeCreateState(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    path: JString<'_>,
    identity_hex: JString<'_>,
    root_hex: JString<'_>,
) -> jstring {
    let result = (|| {
        let path = get_string(&mut env, path)?;
        let identity = get_string(&mut env, identity_hex)?;
        let root = get_string(&mut env, root_hex)?;
        create_state(&path, &identity, &root)
    })();
    java_response(&mut env, result)
}

#[no_mangle]
pub extern "system" fn Java_com_ntd97_app_NativeBridge_nativeWake(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    path: JString<'_>,
    reason: jint,
    user_visible: jboolean,
    foreground_allowed: jboolean,
    background_allowed: jboolean,
    network_available: jboolean,
    charging: jboolean,
    battery_percent: jint,
    thermal_critical: jboolean,
) -> jstring {
    let result = (|| {
        let path = get_string(&mut env, path)?;
        let mut supervisor = restore(&path)?;
        let disposition = supervisor.on_wake(
            wake_reason(reason)?,
            PlatformConstraints {
                user_visible: user_visible != 0,
                foreground_allowed: foreground_allowed != 0,
                background_execution_allowed: background_allowed != 0,
                network_available: network_available != 0,
                charging: charging != 0,
                battery_percent: battery_percent.clamp(0, 100) as u8,
                thermal_critical: thermal_critical != 0,
            },
        );
        let generation = persist(&path, &mut supervisor)?;
        Ok(json!({
            "status":"ok",
            "disposition":format!("{disposition:?}"),
            "generation":generation,
            "snapshot":snapshot_json(&supervisor),
        }))
    })();
    java_response(&mut env, result)
}

#[no_mangle]
pub extern "system" fn Java_com_ntd97_app_NativeBridge_nativeFrame(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    path: JString<'_>,
    battery_percent: jint,
    charging: jboolean,
    thermal: jint,
    visible: jboolean,
) -> jstring {
    let result = (|| {
        let path = get_string(&mut env, path)?;
        let supervisor = restore(&path)?;
        Ok(json!({
            "status":"ok",
            "frame":frame_json(
                &supervisor,
                battery_percent.clamp(0, 100) as u8,
                charging != 0,
                thermal_state(thermal)?,
                visible != 0,
            ),
        }))
    })();
    java_response(&mut env, result)
}

#[no_mangle]
pub extern "system" fn Java_com_ntd97_app_NativeBridge_nativeSnapshot(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    path: JString<'_>,
) -> jstring {
    let result = (|| {
        let path = get_string(&mut env, path)?;
        let supervisor = restore(&path)?;
        Ok(json!({
            "status":"ok",
            "snapshot":snapshot_json(&supervisor),
        }))
    })();
    java_response(&mut env, result)
}

#[no_mangle]
pub extern "system" fn Java_com_ntd97_app_NativeBridge_nativeNextPlanId(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    path: JString<'_>,
) -> jlong {
    let Ok(path) = get_string(&mut env, path) else {
        return -1;
    };
    let Ok(supervisor) = restore(&path) else {
        return -1;
    };
    supervisor
        .actions()
        .state()
        .plans
        .values()
        .find(|plan| {
            !matches!(
                plan.status,
                ActionPlanStatus::Completed
                    | ActionPlanStatus::Failed
                    | ActionPlanStatus::RolledBack
            )
        })
        .map(|plan| plan.id.0 as jlong)
        .unwrap_or(-1)
}

#[no_mangle]
pub extern "system" fn Java_com_ntd97_app_NativeBridge_nativeRunOne(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    path: JString<'_>,
    host: JObject<'_>,
    plan_id: jlong,
    scopes_csv: JString<'_>,
    allow_write: jboolean,
    allow_irreversible: jboolean,
) -> jstring {
    let result = (|| {
        if plan_id <= 0 {
            return Err("plan id must be positive".into());
        }
        let path = get_string(&mut env, path)?;
        let scopes = get_string(&mut env, scopes_csv)?;
        let mut supervisor = restore(&path)?;
        attach_host(&mut env, &mut supervisor, host)?;
        let authority = authority_from(
            &scopes,
            allow_write != 0,
            allow_irreversible != 0,
        )?;
        let mut store = FileCheckpointStore::new(&path);
        let report = supervisor
            .resume_plan_once_durable(
                ntd_runtime::ActionPlanId(plan_id as u64),
                &authority,
                &mut PlatformVerifier,
                &mut store,
            )
            .map_err(|error| format!("{error:?}"))?;

        Ok(json!({
            "status":"ok",
            "planStatus":format!("{:?}", report.plan_status),
            "actionStatus":report.action_status.map(|status| format!("{status:?}")),
            "cursor":report.cursor,
            "summary":report.summary,
            "snapshot":snapshot_json(&supervisor),
        }))
    })();
    java_response(&mut env, result)
}

#[no_mangle]
pub extern "system" fn Java_com_ntd97_app_NativeBridge_nativeAcknowledgeApproval(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    path: JString<'_>,
    approval_id: jlong,
) -> jstring {
    let result = (|| {
        if approval_id <= 0 {
            return Err("approval id must be positive".into());
        }
        let path = get_string(&mut env, path)?;
        let mut supervisor = restore(&path)?;
        supervisor
            .acknowledge_approval(approval_id as u64)
            .map_err(|error| format!("{error:?}"))?;
        let generation = persist(&path, &mut supervisor)?;
        Ok(json!({
            "status":"ok",
            "generation":generation,
            "snapshot":snapshot_json(&supervisor),
        }))
    })();
    java_response(&mut env, result)
}

#[no_mangle]
pub extern "system" fn Java_com_ntd97_app_NativeBridge_nativeSuspend(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    path: JString<'_>,
) -> jstring {
    let result = (|| {
        let path = get_string(&mut env, path)?;
        let mut supervisor = restore(&path)?;
        let mut store = FileCheckpointStore::new(&path);
        let generation = supervisor
            .mark_suspended_by_os(&mut store)
            .map_err(|error| format!("{error:?}"))?;
        Ok(json!({
            "status":"ok",
            "generation":generation,
            "snapshot":snapshot_json(&supervisor),
        }))
    })();
    java_response(&mut env, result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ntd_platform::{AssistantMode, LipViseme};

    #[test]
    fn hex_round_trip_is_stable() {
        let bytes = [0, 1, 15, 16, 127, 255];
        let encoded = encode_hex(&bytes);
        assert_eq!(decode_hex(&encoded).expect("decode"), bytes);
    }

    #[test]
    fn platform_verifier_requires_evidence_for_side_effects() {
        let descriptor = CapabilityDescriptor::new(
            CapabilityId("app.action".into()),
            1,
            CapabilityDomain::App,
            SideEffectClass::ExternalWrite,
        )
        .expect("descriptor");
        let decision = PlatformVerifier.verify(
            &descriptor,
            &TypedAction::AppAction {
                app: "example".into(),
                action: "launch".into(),
                payload: Vec::new(),
            },
            &ActionOutput::text("launched", "ok"),
        );
        assert!(matches!(decision, ActionVerification::Retry { .. }));
    }

    #[test]
    fn assistant_mode_debug_names_are_stable_enough_for_bridge_payload() {
        assert_eq!(format!("{:?}", AssistantMode::Acting), "Acting");
        assert_eq!(format!("{:?}", LipViseme::Closed), "Closed");
    }
}
