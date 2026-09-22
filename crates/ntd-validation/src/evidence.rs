#![forbid(unsafe_code)]

use std::collections::BTreeSet;

use ntd_capsule::sha256;

use crate::ValidationError;

const EVIDENCE_MAGIC: [u8; 6] = *b"NDE97\0";
const EVIDENCE_MAJOR: u16 = 0;
const EVIDENCE_MINOR: u16 = 1;
const EVIDENCE_DIGEST_LEN: usize = 32;
const M13_EVIDENCE_MAGIC: [u8; 6] = *b"M13E97";
const M13_EVIDENCE_MAJOR: u16 = 0;
const M13_EVIDENCE_MINOR: u16 = 1;

pub const M13_GATE_WEB_SEARCH: u16 = 1 << 0;
pub const M13_GATE_BROWSER: u16 = 1 << 1;
pub const M13_GATE_SAF_READ: u16 = 1 << 2;
pub const M13_GATE_SAF_WRITE: u16 = 1 << 3;
pub const M13_GATE_DEVICE_CLIPBOARD: u16 = 1 << 4;
pub const M13_GATE_APP_ACCESSIBILITY: u16 = 1 << 5;
pub const M13_GATE_PC_OBSERVE: u16 = 1 << 6;
pub const M13_GATE_PC_WRITE: u16 = 1 << 7;
pub const M13_GATE_UPLOAD_RESTORE: u16 = 1 << 8;
pub const M13_GATE_MIXED_SEQUENCE: u16 = 1 << 9;
pub const M13_REQUIRED_GATE_MASK: u16 = M13_GATE_WEB_SEARCH
    | M13_GATE_BROWSER
    | M13_GATE_SAF_READ
    | M13_GATE_SAF_WRITE
    | M13_GATE_DEVICE_CLIPBOARD
    | M13_GATE_APP_ACCESSIBILITY
    | M13_GATE_PC_OBSERVE
    | M13_GATE_PC_WRITE
    | M13_GATE_UPLOAD_RESTORE
    | M13_GATE_MIXED_SEQUENCE;
pub const M13_MIN_VERIFIED_ACTIONS: u32 = 10;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct M13HardwareEvidenceRecord {
    pub build_revision: String,
    pub device_fingerprint: String,
    pub gate_mask: u16,
    pub verified_action_count: u32,
    pub process_death_reconfirmed: bool,
    pub sovereignty_audit_passed: bool,
    pub run_nonce_sha256: [u8; 32],
}

impl M13HardwareEvidenceRecord {
    pub fn validate(&self) -> bool {
        is_git_revision(&self.build_revision)
            && is_sha256_label(&self.device_fingerprint)
            && self.gate_mask != 0
            && self.gate_mask & !M13_REQUIRED_GATE_MASK == 0
            && self.verified_action_count > 0
            && self.run_nonce_sha256.iter().any(|byte| *byte != 0)
    }

    pub fn accepted_for_revision(&self, expected_revision: &str) -> bool {
        self.validate()
            && self.build_revision == expected_revision
            && self.gate_mask == M13_REQUIRED_GATE_MASK
            && self.verified_action_count >= M13_MIN_VERIFIED_ACTIONS
            && self.process_death_reconfirmed
            && self.sovereignty_audit_passed
    }
}

pub fn encode_m13_hardware_evidence(
    record: &M13HardwareEvidenceRecord,
) -> Result<Vec<u8>, ValidationError> {
    if !record.validate() {
        return Err(ValidationError::EvidenceCodec);
    }
    let mut out = Vec::new();
    out.extend_from_slice(&M13_EVIDENCE_MAGIC);
    push_u16(&mut out, M13_EVIDENCE_MAJOR);
    push_u16(&mut out, M13_EVIDENCE_MINOR);
    out.push(u8::from(record.process_death_reconfirmed));
    out.push(u8::from(record.sovereignty_audit_passed));
    push_u16(&mut out, record.gate_mask);
    push_u32(&mut out, record.verified_action_count);
    out.extend_from_slice(&record.run_nonce_sha256);
    push_string(&mut out, &record.build_revision)?;
    push_string(&mut out, &record.device_fingerprint)?;
    let digest = sha256(&out);
    out.extend_from_slice(&digest);
    Ok(out)
}

pub fn decode_m13_hardware_evidence(
    bytes: &[u8],
) -> Result<M13HardwareEvidenceRecord, ValidationError> {
    if bytes.len() <= EVIDENCE_DIGEST_LEN {
        return Err(ValidationError::EvidenceCodec);
    }
    let payload_len = bytes
        .len()
        .checked_sub(EVIDENCE_DIGEST_LEN)
        .ok_or(ValidationError::EvidenceCodec)?;
    let (payload, digest) = bytes.split_at(payload_len);
    if sha256(payload).as_slice() != digest {
        return Err(ValidationError::EvidenceCodec);
    }

    let mut cursor = Cursor::new(payload);
    if cursor.take(6)? != M13_EVIDENCE_MAGIC
        || cursor.u16()? != M13_EVIDENCE_MAJOR
        || cursor.u16()? > M13_EVIDENCE_MINOR
    {
        return Err(ValidationError::EvidenceCodec);
    }
    let process_death_reconfirmed = decode_bool(cursor.u8()?)?;
    let sovereignty_audit_passed = decode_bool(cursor.u8()?)?;
    let gate_mask = cursor.u16()?;
    let verified_action_count = cursor.u32()?;
    let mut run_nonce_sha256 = [0u8; 32];
    run_nonce_sha256.copy_from_slice(cursor.take(32)?);
    let build_revision = cursor.string()?;
    let device_fingerprint = cursor.string()?;
    if !cursor.is_finished() {
        return Err(ValidationError::EvidenceCodec);
    }
    let record = M13HardwareEvidenceRecord {
        build_revision,
        device_fingerprint,
        gate_mask,
        verified_action_count,
        process_death_reconfirmed,
        sovereignty_audit_passed,
        run_nonce_sha256,
    };
    if !record.validate() {
        return Err(ValidationError::EvidenceCodec);
    }
    Ok(record)
}

fn decode_bool(value: u8) -> Result<bool, ValidationError> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(ValidationError::EvidenceCodec),
    }
}

fn is_sha256_label(value: &str) -> bool {
    value.len() == 71
        && value.starts_with("sha256:")
        && value[7..].bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EvidenceClass {
    CiSurrogate = 1,
    PhysicalDevice = 2,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceEvidence {
    pub class: EvidenceClass,
    pub profile: String,
    pub device_fingerprint: String,
    pub p95_latency_nanos: u64,
    pub energy_per_task_microjoules: u64,
    pub reliability_permille: u16,
    pub recovery_permille: u16,
    pub sovereignty_audit_passed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalEvidenceRecord {
    pub evidence: DeviceEvidence,
    pub build_revision: String,
    pub total_ram_bytes: u64,
    pub sample_count: u32,
    pub energy_source: String,
}

impl PhysicalEvidenceRecord {
    pub fn validate(&self) -> bool {
        self.evidence.class == EvidenceClass::PhysicalDevice
            && !self.evidence.profile.trim().is_empty()
            && !self.evidence.device_fingerprint.trim().is_empty()
            && self.evidence.p95_latency_nanos > 0
            && self.evidence.energy_per_task_microjoules > 0
            && self.evidence.reliability_permille <= 1000
            && self.evidence.recovery_permille <= 1000
            && !self.build_revision.trim().is_empty()
            && self.total_ram_bytes > 0
            && self.sample_count > 0
            && !self.energy_source.trim().is_empty()
    }
}

pub fn encode_physical_evidence(
    record: &PhysicalEvidenceRecord,
) -> Result<Vec<u8>, ValidationError> {
    if !record.validate() {
        return Err(ValidationError::EvidenceCodec);
    }

    let mut out = Vec::new();
    out.extend_from_slice(&EVIDENCE_MAGIC);
    push_u16(&mut out, EVIDENCE_MAJOR);
    push_u16(&mut out, EVIDENCE_MINOR);
    out.push(record.evidence.class as u8);
    out.push(u8::from(record.evidence.sovereignty_audit_passed));
    push_u16(&mut out, record.evidence.reliability_permille);
    push_u16(&mut out, record.evidence.recovery_permille);
    push_u32(&mut out, record.sample_count);
    push_u64(&mut out, record.evidence.p95_latency_nanos);
    push_u64(&mut out, record.evidence.energy_per_task_microjoules);
    push_u64(&mut out, record.total_ram_bytes);
    push_string(&mut out, &record.evidence.profile)?;
    push_string(&mut out, &record.evidence.device_fingerprint)?;
    push_string(&mut out, &record.build_revision)?;
    push_string(&mut out, &record.energy_source)?;
    let digest = sha256(&out);
    out.extend_from_slice(&digest);
    Ok(out)
}

pub fn decode_physical_evidence(bytes: &[u8]) -> Result<PhysicalEvidenceRecord, ValidationError> {
    if bytes.len() <= EVIDENCE_DIGEST_LEN {
        return Err(ValidationError::EvidenceCodec);
    }
    let payload_len = bytes
        .len()
        .checked_sub(EVIDENCE_DIGEST_LEN)
        .ok_or(ValidationError::EvidenceCodec)?;
    let (payload, digest) = bytes.split_at(payload_len);
    if sha256(payload).as_slice() != digest {
        return Err(ValidationError::EvidenceCodec);
    }

    let mut cursor = Cursor::new(payload);
    if cursor.take(6)? != EVIDENCE_MAGIC
        || cursor.u16()? != EVIDENCE_MAJOR
        || cursor.u16()? > EVIDENCE_MINOR
    {
        return Err(ValidationError::EvidenceCodec);
    }
    if cursor.u8()? != EvidenceClass::PhysicalDevice as u8 {
        return Err(ValidationError::EvidenceCodec);
    }
    let sovereignty_audit_passed = match cursor.u8()? {
        0 => false,
        1 => true,
        _ => return Err(ValidationError::EvidenceCodec),
    };
    let reliability_permille = cursor.u16()?;
    let recovery_permille = cursor.u16()?;
    let sample_count = cursor.u32()?;
    let p95_latency_nanos = cursor.u64()?;
    let energy_per_task_microjoules = cursor.u64()?;
    let total_ram_bytes = cursor.u64()?;
    let profile = cursor.string()?;
    let device_fingerprint = cursor.string()?;
    let build_revision = cursor.string()?;
    let energy_source = cursor.string()?;
    if !cursor.is_finished() {
        return Err(ValidationError::EvidenceCodec);
    }

    let record = PhysicalEvidenceRecord {
        evidence: DeviceEvidence {
            class: EvidenceClass::PhysicalDevice,
            profile,
            device_fingerprint,
            p95_latency_nanos,
            energy_per_task_microjoules,
            reliability_permille,
            recovery_permille,
            sovereignty_audit_passed,
        },
        build_revision,
        total_ram_bytes,
        sample_count,
        energy_source,
    };
    if !record.validate() {
        return Err(ValidationError::EvidenceCodec);
    }
    Ok(record)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationTargets {
    pub required_physical_profiles: Vec<String>,
    pub max_p95_latency_nanos: u64,
    pub max_energy_per_task_microjoules: u64,
    pub min_reliability_permille: u16,
    pub min_recovery_permille: u16,
}

impl ValidationTargets {
    pub fn m10_reference() -> Self {
        Self {
            required_physical_profiles: vec![
                "mobile-4gb".into(),
                "mobile-8gb".into(),
                "mobile-12gb".into(),
            ],
            max_p95_latency_nanos: 2_000_000_000,
            max_energy_per_task_microjoules: 5_000_000,
            min_reliability_permille: 990,
            min_recovery_permille: 990,
        }
    }

    pub fn validate(&self) -> bool {
        !self.required_physical_profiles.is_empty()
            && self.max_p95_latency_nanos > 0
            && self.max_energy_per_task_microjoules > 0
            && self.min_reliability_permille <= 1000
            && self.min_recovery_permille <= 1000
            && self
                .required_physical_profiles
                .iter()
                .all(|profile| !profile.trim().is_empty())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceGateFailure {
    InvalidTargets,
    InvalidExpectedRevision,
    MissingProfile(String),
    DuplicateProfile(String),
    InvalidRecord(String),
    BuildRevisionMismatch {
        profile: String,
        expected: String,
        actual: String,
    },
    DuplicateFingerprint(String),
    LatencyExceeded {
        profile: String,
        actual: u64,
        maximum: u64,
    },
    EnergyExceeded {
        profile: String,
        actual: u64,
        maximum: u64,
    },
    ReliabilityBelowTarget {
        profile: String,
        actual: u16,
        minimum: u16,
    },
    RecoveryBelowTarget {
        profile: String,
        actual: u16,
        minimum: u16,
    },
    SovereigntyAuditMissing(String),
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EvidenceGateReport {
    pub accepted: bool,
    pub accepted_profiles: Vec<String>,
    pub failures: Vec<EvidenceGateFailure>,
}

pub fn evaluate_physical_records(
    records: &[PhysicalEvidenceRecord],
    targets: &ValidationTargets,
    expected_revision: &str,
) -> EvidenceGateReport {
    let mut report = EvidenceGateReport::default();
    if !targets.validate() {
        report.failures.push(EvidenceGateFailure::InvalidTargets);
        return report;
    }
    if !is_git_revision(expected_revision) {
        report
            .failures
            .push(EvidenceGateFailure::InvalidExpectedRevision);
        return report;
    }

    let mut fingerprints = BTreeSet::new();
    for required in &targets.required_physical_profiles {
        let matching = records
            .iter()
            .filter(|record| record.evidence.profile == *required)
            .collect::<Vec<_>>();
        if matching.is_empty() {
            report
                .failures
                .push(EvidenceGateFailure::MissingProfile(required.clone()));
            continue;
        }
        if matching.len() != 1 {
            report
                .failures
                .push(EvidenceGateFailure::DuplicateProfile(required.clone()));
            continue;
        }

        let failure_count_before = report.failures.len();
        let record = matching[0];
        let evidence = &record.evidence;
        if !record.validate() {
            report
                .failures
                .push(EvidenceGateFailure::InvalidRecord(required.clone()));
            continue;
        }
        if record.build_revision != expected_revision {
            report
                .failures
                .push(EvidenceGateFailure::BuildRevisionMismatch {
                    profile: required.clone(),
                    expected: expected_revision.to_owned(),
                    actual: record.build_revision.clone(),
                });
        }
        if !fingerprints.insert(evidence.device_fingerprint.clone()) {
            report
                .failures
                .push(EvidenceGateFailure::DuplicateFingerprint(
                    evidence.device_fingerprint.clone(),
                ));
        }
        if evidence.p95_latency_nanos > targets.max_p95_latency_nanos {
            report.failures.push(EvidenceGateFailure::LatencyExceeded {
                profile: required.clone(),
                actual: evidence.p95_latency_nanos,
                maximum: targets.max_p95_latency_nanos,
            });
        }
        if evidence.energy_per_task_microjoules > targets.max_energy_per_task_microjoules {
            report.failures.push(EvidenceGateFailure::EnergyExceeded {
                profile: required.clone(),
                actual: evidence.energy_per_task_microjoules,
                maximum: targets.max_energy_per_task_microjoules,
            });
        }
        if evidence.reliability_permille < targets.min_reliability_permille {
            report
                .failures
                .push(EvidenceGateFailure::ReliabilityBelowTarget {
                    profile: required.clone(),
                    actual: evidence.reliability_permille,
                    minimum: targets.min_reliability_permille,
                });
        }
        if evidence.recovery_permille < targets.min_recovery_permille {
            report
                .failures
                .push(EvidenceGateFailure::RecoveryBelowTarget {
                    profile: required.clone(),
                    actual: evidence.recovery_permille,
                    minimum: targets.min_recovery_permille,
                });
        }
        if !evidence.sovereignty_audit_passed {
            report
                .failures
                .push(EvidenceGateFailure::SovereigntyAuditMissing(
                    required.clone(),
                ));
        }

        if report.failures.len() == failure_count_before {
            report.accepted_profiles.push(required.clone());
        }
    }

    report.accepted = report.failures.is_empty()
        && report.accepted_profiles.len() == targets.required_physical_profiles.len();
    report
}

fn is_git_revision(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EvidenceMatrix {
    entries: Vec<DeviceEvidence>,
}

impl EvidenceMatrix {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, evidence: DeviceEvidence) {
        self.entries.push(evidence);
    }

    pub fn entries(&self) -> &[DeviceEvidence] {
        &self.entries
    }

    pub fn meets_targets(&self, targets: &ValidationTargets) -> bool {
        if !targets.validate() {
            return false;
        }
        let mut fingerprints = BTreeSet::new();
        for required in &targets.required_physical_profiles {
            let Some(evidence) = self.entries.iter().find(|entry| {
                entry.class == EvidenceClass::PhysicalDevice
                    && entry.profile == *required
                    && !entry.device_fingerprint.trim().is_empty()
                    && entry.p95_latency_nanos > 0
                    && entry.p95_latency_nanos <= targets.max_p95_latency_nanos
                    && entry.energy_per_task_microjoules > 0
                    && entry.energy_per_task_microjoules <= targets.max_energy_per_task_microjoules
                    && entry.reliability_permille >= targets.min_reliability_permille
                    && entry.recovery_permille >= targets.min_recovery_permille
                    && entry.sovereignty_audit_passed
            }) else {
                return false;
            };
            if !fingerprints.insert(evidence.device_fingerprint.clone()) {
                return false;
            }
        }
        true
    }
}

fn push_string(out: &mut Vec<u8>, value: &str) -> Result<(), ValidationError> {
    let bytes = value.as_bytes();
    push_u32(
        out,
        u32::try_from(bytes.len()).map_err(|_| ValidationError::EvidenceCodec)?,
    );
    out.extend_from_slice(bytes);
    Ok(())
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

    fn take(&mut self, len: usize) -> Result<&'a [u8], ValidationError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(ValidationError::EvidenceCodec)?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or(ValidationError::EvidenceCodec)?;
        self.offset = end;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8, ValidationError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, ValidationError> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> Result<u32, ValidationError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn u64(&mut self) -> Result<u64, ValidationError> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn string(&mut self) -> Result<String, ValidationError> {
        let len = usize::try_from(self.u32()?).map_err(|_| ValidationError::EvidenceCodec)?;
        String::from_utf8(self.take(len)?.to_vec()).map_err(|_| ValidationError::EvidenceCodec)
    }

    fn is_finished(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> PhysicalEvidenceRecord {
        PhysicalEvidenceRecord {
            evidence: DeviceEvidence {
                class: EvidenceClass::PhysicalDevice,
                profile: "mobile-8gb".into(),
                device_fingerprint: "sha256:fixture".into(),
                p95_latency_nanos: 42_000_000,
                energy_per_task_microjoules: 120_000,
                reliability_permille: 1000,
                recovery_permille: 1000,
                sovereignty_audit_passed: true,
            },
            build_revision: "0123456789abcdef0123456789abcdef01234567".into(),
            total_ram_bytes: 8 * 1024 * 1024 * 1024,
            sample_count: 64,
            energy_source: "energy-counter".into(),
        }
    }

    fn m13_fixture() -> M13HardwareEvidenceRecord {
        M13HardwareEvidenceRecord {
            build_revision: "0123456789abcdef0123456789abcdef01234567".into(),
            device_fingerprint:
                "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into(),
            gate_mask: M13_REQUIRED_GATE_MASK,
            verified_action_count: M13_MIN_VERIFIED_ACTIONS,
            process_death_reconfirmed: true,
            sovereignty_audit_passed: true,
            run_nonce_sha256: [0x97; 32],
        }
    }

    #[test]
    fn m13_hardware_evidence_round_trips_and_requires_all_gates() {
        let record = m13_fixture();
        let encoded = encode_m13_hardware_evidence(&record).expect("encode");
        let decoded = decode_m13_hardware_evidence(&encoded).expect("decode");
        assert_eq!(decoded, record);
        assert!(decoded.accepted_for_revision(&record.build_revision));

        let mut missing = record.clone();
        missing.gate_mask &= !M13_GATE_PC_WRITE;
        assert!(!missing.accepted_for_revision(&record.build_revision));
    }

    #[test]
    fn m13_hardware_evidence_rejects_tamper_and_wrong_revision() {
        let record = m13_fixture();
        let mut encoded = encode_m13_hardware_evidence(&record).expect("encode");
        encoded[20] ^= 1;
        assert_eq!(
            decode_m13_hardware_evidence(&encoded),
            Err(ValidationError::EvidenceCodec)
        );
        assert!(!record.accepted_for_revision("fedcba9876543210fedcba9876543210fedcba98"));
    }

    #[test]
    fn physical_evidence_round_trips_with_integrity_digest() {
        let record = fixture();
        let encoded = encode_physical_evidence(&record).expect("encode");
        assert_eq!(decode_physical_evidence(&encoded).expect("decode"), record);
    }

    #[test]
    fn m10_gate_requires_same_revision_and_distinct_devices() {
        let revision = "0123456789abcdef0123456789abcdef01234567";
        let mut records = Vec::new();
        for (index, profile) in ["mobile-4gb", "mobile-8gb", "mobile-12gb"]
            .into_iter()
            .enumerate()
        {
            let mut record = fixture();
            record.evidence.profile = profile.into();
            record.evidence.device_fingerprint = format!("sha256:device-{index}");
            record.build_revision = revision.into();
            records.push(record);
        }

        let report =
            evaluate_physical_records(&records, &ValidationTargets::m10_reference(), revision);
        assert!(report.accepted);
        assert!(report.failures.is_empty());
        assert_eq!(report.accepted_profiles.len(), 3);
    }

    #[test]
    fn m10_gate_rejects_mixed_revision_and_duplicate_device() {
        let revision = "0123456789abcdef0123456789abcdef01234567";
        let mut records = Vec::new();
        for profile in ["mobile-4gb", "mobile-8gb", "mobile-12gb"] {
            let mut record = fixture();
            record.evidence.profile = profile.into();
            record.evidence.device_fingerprint = "sha256:same-device".into();
            record.build_revision = revision.into();
            records.push(record);
        }
        records[1].build_revision = "fedcba9876543210fedcba9876543210fedcba98".into();

        let report =
            evaluate_physical_records(&records, &ValidationTargets::m10_reference(), revision);
        assert!(!report.accepted);
        assert!(report
            .failures
            .iter()
            .any(|failure| matches!(failure, EvidenceGateFailure::BuildRevisionMismatch { .. })));
        assert!(report
            .failures
            .iter()
            .any(|failure| matches!(failure, EvidenceGateFailure::DuplicateFingerprint(_))));
    }

    #[test]
    fn physical_evidence_rejects_tampering() {
        let mut encoded = encode_physical_evidence(&fixture()).expect("encode");
        encoded[20] ^= 1;
        assert_eq!(
            decode_physical_evidence(&encoded),
            Err(ValidationError::EvidenceCodec)
        );
    }
}
