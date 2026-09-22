#![forbid(unsafe_code)]

use std::sync::Arc;

use jni::{
    objects::{JByteArray, JObject, JString, JValue},
    JavaVM,
};
use ntd_capsule::sha256;
use ntd_runtime::{
    ActionId, ActionOutput, ActionValue, ActionVerification, ActionVerifier, AdapterResult,
    CapabilityAdapter, CapabilityDescriptor, TypedAction,
};

const RECEIPT_MAGIC: [u8; 6] = *b"APR97\0";
const RECEIPT_VERSION: u8 = 1;
const RECEIPT_COMPLETED: u8 = 1;
const RECEIPT_RETRYABLE: u8 = 2;
const VALUE_NONE: u8 = 0;
const VALUE_TEXT: u8 = 1;
const VALUE_BYTES: u8 = 2;
const MAX_RECEIPT_BYTES: usize = 2 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
struct PlatformReceipt {
    status: u8,
    action_id: u64,
    summary: String,
    value: ActionValue,
    evidence: Vec<String>,
    rollback_token: Option<Vec<u8>>,
}

pub struct AndroidPlatformAdapter {
    vm: Arc<JavaVM>,
}

impl AndroidPlatformAdapter {
    pub fn new(vm: Arc<JavaVM>) -> Self {
        Self { vm }
    }

    fn call_web_fetch(&self, action_id: ActionId, url: &str) -> Result<PlatformReceipt, String> {
        let mut env = self
            .vm
            .attach_current_thread()
            .map_err(|error| format!("attach Android platform thread: {error}"))?;
        let url = env
            .new_string(url)
            .map_err(|error| format!("allocate web URL: {error}"))?;
        let url_obj = JObject::from(url);
        let value = env
            .call_static_method(
                "ai/ntd97/mobile/NtdAndroidCapabilityHost",
                "webFetch",
                "(JLjava/lang/String;)[B",
                &[
                    JValue::Long(action_id_to_jlong(action_id)?),
                    JValue::Object(&url_obj),
                ],
            )
            .map_err(|error| format!("Android web.fetch failed: {error}"))?
            .l()
            .map_err(|error| format!("Android web.fetch result type: {error}"))?;
        decode_java_receipt(&mut env, action_id, value)
    }

    fn call_file_read(&self, action_id: ActionId, path: &str) -> Result<PlatformReceipt, String> {
        let mut env = self
            .vm
            .attach_current_thread()
            .map_err(|error| format!("attach Android platform thread: {error}"))?;
        let path = env
            .new_string(path)
            .map_err(|error| format!("allocate file path: {error}"))?;
        let path_obj = JObject::from(path);
        let value = env
            .call_static_method(
                "ai/ntd97/mobile/NtdAndroidCapabilityHost",
                "fileRead",
                "(JLjava/lang/String;)[B",
                &[
                    JValue::Long(action_id_to_jlong(action_id)?),
                    JValue::Object(&path_obj),
                ],
            )
            .map_err(|error| format!("Android file.read failed: {error}"))?
            .l()
            .map_err(|error| format!("Android file.read result type: {error}"))?;
        decode_java_receipt(&mut env, action_id, value)
    }

    fn call_file_write(
        &self,
        action_id: ActionId,
        path: &str,
        bytes: &[u8],
    ) -> Result<PlatformReceipt, String> {
        let mut env = self
            .vm
            .attach_current_thread()
            .map_err(|error| format!("attach Android platform thread: {error}"))?;
        let path = env
            .new_string(path)
            .map_err(|error| format!("allocate file path: {error}"))?;
        let payload = env
            .byte_array_from_slice(bytes)
            .map_err(|error| format!("allocate file payload: {error}"))?;
        let path_obj = JObject::from(path);
        let payload_obj = JObject::from(payload);
        let value = env
            .call_static_method(
                "ai/ntd97/mobile/NtdAndroidCapabilityHost",
                "fileWrite",
                "(JLjava/lang/String;[B)[B",
                &[
                    JValue::Long(action_id_to_jlong(action_id)?),
                    JValue::Object(&path_obj),
                    JValue::Object(&payload_obj),
                ],
            )
            .map_err(|error| format!("Android file.write failed: {error}"))?
            .l()
            .map_err(|error| format!("Android file.write result type: {error}"))?;
        decode_java_receipt(&mut env, action_id, value)
    }

    fn call_file_rollback(
        &self,
        action_id: ActionId,
        path: &str,
        rollback_token: &[u8],
    ) -> Result<(), String> {
        let mut env = self
            .vm
            .attach_current_thread()
            .map_err(|error| format!("attach Android platform thread: {error}"))?;
        let path = env
            .new_string(path)
            .map_err(|error| format!("allocate rollback path: {error}"))?;
        let token = env
            .byte_array_from_slice(rollback_token)
            .map_err(|error| format!("allocate rollback token: {error}"))?;
        let path_obj = JObject::from(path);
        let token_obj = JObject::from(token);
        let ok = env
            .call_static_method(
                "ai/ntd97/mobile/NtdAndroidCapabilityHost",
                "rollbackFileWrite",
                "(JLjava/lang/String;[B)Z",
                &[
                    JValue::Long(action_id_to_jlong(action_id)?),
                    JValue::Object(&path_obj),
                    JValue::Object(&token_obj),
                ],
            )
            .map_err(|error| format!("Android file rollback failed: {error}"))?
            .z()
            .map_err(|error| format!("Android file rollback result type: {error}"))?;
        if ok {
            Ok(())
        } else {
            Err("Android file rollback was rejected".into())
        }
    }
}

impl CapabilityAdapter for AndroidPlatformAdapter {
    fn execute(
        &mut self,
        action_id: ActionId,
        action: &TypedAction,
    ) -> Result<AdapterResult, String> {
        let receipt = match action {
            TypedAction::WebFetch { url } => self.call_web_fetch(action_id, url)?,
            TypedAction::FileRead { path } => self.call_file_read(action_id, path)?,
            TypedAction::FileWrite { path, bytes } => {
                self.call_file_write(action_id, path, bytes)?
            }
            _ => return Err("Android platform adapter does not support this action".into()),
        };
        receipt_to_adapter_result(receipt)
    }

    fn rollback(
        &mut self,
        action_id: ActionId,
        action: &TypedAction,
        rollback_token: &[u8],
    ) -> Result<(), String> {
        match action {
            TypedAction::FileWrite { path, .. } => {
                self.call_file_rollback(action_id, path, rollback_token)
            }
            _ => Err("Android platform adapter has no rollback for this action".into()),
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct AndroidPlatformVerifier;

impl ActionVerifier for AndroidPlatformVerifier {
    fn verify(
        &mut self,
        descriptor: &CapabilityDescriptor,
        action: &TypedAction,
        output: &ActionOutput,
    ) -> ActionVerification {
        let expected_prefix = match action {
            TypedAction::WebFetch { .. } => "android-http",
            TypedAction::FileRead { .. } => "android-app-file-read",
            TypedAction::FileWrite { .. } => "android-app-file-write",
            _ => {
                return ActionVerification::Reject {
                    reason: "unsupported Android platform action".into(),
                };
            }
        };

        if output.summary.trim().is_empty()
            || !output
                .evidence
                .iter()
                .any(|item| item == expected_prefix)
        {
            return ActionVerification::Reject {
                reason: "platform output lacks trusted Android evidence".into(),
            };
        }

        let payload = match (action, &output.value) {
            (TypedAction::WebFetch { .. } | TypedAction::FileRead { .. }, ActionValue::Bytes(bytes)) => {
                bytes.as_slice()
            }
            (TypedAction::FileWrite { bytes, .. }, ActionValue::None) => bytes.as_slice(),
            _ => {
                return ActionVerification::Reject {
                    reason: "platform output has unexpected value type".into(),
                };
            }
        };
        let digest = digest_hex(&sha256(payload));
        let expected_digest = format!("sha256={digest}");
        if !output
            .evidence
            .iter()
            .any(|item| item == &expected_digest)
        {
            return ActionVerification::Reject {
                reason: "platform output digest evidence mismatch".into(),
            };
        }

        if descriptor.verification_required {
            ActionVerification::Accept
        } else {
            ActionVerification::Reject {
                reason: "production platform capability must require verification".into(),
            }
        }
    }
}

fn receipt_to_adapter_result(receipt: PlatformReceipt) -> Result<AdapterResult, String> {
    match receipt.status {
        RECEIPT_COMPLETED => Ok(AdapterResult::Completed {
            output: ActionOutput {
                summary: receipt.summary,
                value: receipt.value,
                evidence: receipt.evidence,
            },
            rollback_token: receipt.rollback_token,
        }),
        RECEIPT_RETRYABLE => Ok(AdapterResult::Retryable {
            reason: receipt.summary,
            resume_token: None,
        }),
        _ => Err("invalid Android platform receipt status".into()),
    }
}

fn decode_java_receipt(
    env: &mut jni::JNIEnv<'_>,
    expected_action_id: ActionId,
    value: JObject<'_>,
) -> Result<PlatformReceipt, String> {
    if value.is_null() {
        return Err("Android platform returned a null receipt".into());
    }
    let array = JByteArray::from(value);
    let bytes = env
        .convert_byte_array(&array)
        .map_err(|error| format!("copy Android platform receipt: {error}"))?;
    if bytes.len() > MAX_RECEIPT_BYTES {
        return Err("Android platform receipt exceeds limit".into());
    }
    let receipt = decode_receipt(&bytes)?;
    if receipt.action_id != expected_action_id.0 {
        return Err("Android platform receipt action id mismatch".into());
    }
    Ok(receipt)
}

fn decode_receipt(bytes: &[u8]) -> Result<PlatformReceipt, String> {
    let mut cursor = Cursor::new(bytes);
    if cursor.take(6)? != RECEIPT_MAGIC {
        return Err("invalid Android platform receipt magic".into());
    }
    if cursor.u8()? != RECEIPT_VERSION {
        return Err("unsupported Android platform receipt version".into());
    }
    let status = cursor.u8()?;
    if !matches!(status, RECEIPT_COMPLETED | RECEIPT_RETRYABLE) {
        return Err("invalid Android platform receipt status".into());
    }
    let action_id = cursor.u64()?;
    let summary = cursor.string()?;
    let value_kind = cursor.u8()?;
    let value_bytes = cursor.bytes()?;
    let value = match value_kind {
        VALUE_NONE if value_bytes.is_empty() => ActionValue::None,
        VALUE_TEXT => ActionValue::Text(
            String::from_utf8(value_bytes).map_err(|_| "invalid receipt text value")?,
        ),
        VALUE_BYTES => ActionValue::Bytes(value_bytes),
        _ => return Err("invalid Android platform receipt value".into()),
    };
    let evidence_count = usize::from(cursor.u16()?);
    let mut evidence = Vec::with_capacity(evidence_count);
    for _ in 0..evidence_count {
        evidence.push(cursor.string()?);
    }
    let rollback = cursor.bytes()?;
    if !cursor.is_finished() {
        return Err("non-canonical Android platform receipt".into());
    }
    Ok(PlatformReceipt {
        status,
        action_id,
        summary,
        value,
        evidence,
        rollback_token: (!rollback.is_empty()).then_some(rollback),
    })
}

fn action_id_to_jlong(action_id: ActionId) -> Result<i64, String> {
    i64::try_from(action_id.0).map_err(|_| "action id does not fit JNI long".into())
}

fn digest_hex(digest: &[u8; 32]) -> String {
    use std::fmt::Write as _;

    let mut out = String::with_capacity(64);
    for byte in digest {
        write!(&mut out, "{byte:02x}").expect("write digest");
    }
    out
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], String> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or_else(|| "Android platform receipt overflow".to_owned())?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| "truncated Android platform receipt".to_owned())?;
        self.offset = end;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, String> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> Result<u32, String> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
        ]))
    }

    fn u64(&mut self) -> Result<u64, String> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn bytes(&mut self) -> Result<Vec<u8>, String> {
        let length = usize::try_from(self.u32()?)
            .map_err(|_| "Android platform receipt length overflow".to_owned())?;
        Ok(self.take(length)?.to_vec())
    }

    fn string(&mut self) -> Result<String, String> {
        String::from_utf8(self.bytes()?).map_err(|_| "invalid Android platform receipt UTF-8".into())
    }

    fn is_finished(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_u16(out: &mut Vec<u8>, value: u16) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u32(out: &mut Vec<u8>, value: u32) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(out: &mut Vec<u8>, value: u64) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn push_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
        push_u32(out, u32::try_from(bytes.len()).expect("len"));
        out.extend_from_slice(bytes);
    }

    #[test]
    fn receipt_codec_binds_action_and_verified_payload() {
        let body = b"hello";
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&RECEIPT_MAGIC);
        bytes.push(RECEIPT_VERSION);
        bytes.push(RECEIPT_COMPLETED);
        push_u64(&mut bytes, 9);
        push_bytes(&mut bytes, b"fetched");
        bytes.push(VALUE_BYTES);
        push_bytes(&mut bytes, body);
        push_u16(&mut bytes, 2);
        push_bytes(&mut bytes, b"android-http");
        push_bytes(
            &mut bytes,
            format!("sha256={}", digest_hex(&sha256(body))).as_bytes(),
        );
        push_bytes(&mut bytes, &[]);

        let decoded = decode_receipt(&bytes).expect("decode");
        assert_eq!(decoded.action_id, 9);
        assert_eq!(decoded.value, ActionValue::Bytes(body.to_vec()));
        let descriptor = CapabilityDescriptor::new(
            ntd_runtime::CapabilityId("web.fetch".into()),
            1,
            ntd_runtime::CapabilityDomain::Web,
            ntd_runtime::SideEffectClass::ReadOnly,
        )
        .expect("descriptor");
        assert_eq!(
            AndroidPlatformVerifier.verify(
                &descriptor,
                &TypedAction::WebFetch {
                    url: "https://example.invalid".into(),
                },
                &ActionOutput {
                    summary: decoded.summary,
                    value: decoded.value,
                    evidence: decoded.evidence,
                },
            ),
            ActionVerification::Accept
        );
    }

    #[test]
    fn receipt_codec_rejects_trailing_bytes() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&RECEIPT_MAGIC);
        bytes.push(RECEIPT_VERSION);
        bytes.push(RECEIPT_RETRYABLE);
        push_u64(&mut bytes, 1);
        push_bytes(&mut bytes, b"offline");
        bytes.push(VALUE_NONE);
        push_bytes(&mut bytes, &[]);
        push_u16(&mut bytes, 0);
        push_bytes(&mut bytes, &[]);
        bytes.push(99);

        assert!(decode_receipt(&bytes).is_err());
    }
}
