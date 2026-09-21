#![forbid(unsafe_code)]

use std::collections::BTreeSet;

use ntd_capsule::sha256;

use crate::ValidationError;

const EVIDENCE_MAGIC: [u8; 6] = *b"NDE97\0";
const EVIDENCE_MAJOR: u16 = 0;
const EVIDENCE_MINOR: u16 = 1;
const EVIDENCE_DIGEST_LEN: usize = 32;

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
            report
                .failures
                .push(EvidenceGateFailure::LatencyExceeded {
                    profile: required.clone(),
                    actual: evidence.p95_latency_nanos,
                    maximum: targets.max_p95_latency_nanos,
                });
        }
        if evidence.energy_per_task_microjoules > targets.max_energy_per_task_microjoules {
            report
                .failures
                .push(EvidenceGateFailure::EnergyExceeded {
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

        if !report.failures.iter().any(|failure| match failure {
            EvidenceGateFailure::BuildRevisionMismatch { profile, .. }
            | EvidenceGateFailure::LatencyExceeded { profile, .. }
            | EvidenceGateFailure::EnergyExceeded { profile, .. }
            | EvidenceGateFailure::ReliabilityBelowTarget { profile, .. }
            | EvidenceGateFailure::RecoveryBelowTarget { profile, .. } => profile == required,
            EvidenceGateFailure::InvalidRecord(profile)
            | EvidenceGateFailure::DuplicateProfile(profile)
            | EvidenceGateFailure::MissingProfile(profile)
            | EvidenceGateFailure::SovereigntyAuditMissing(profile) => profile == required,
            EvidenceGateFailure::DuplicateFingerprint(_) => false,
            EvidenceGateFailure::InvalidTargets | EvidenceGateFailure::InvalidExpectedRevision => {
                false
            }
        }) {
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
        records[1].build_revision =
            "fedcba9876543210fedcba9876543210fedcba98".into();

        let report =
            evaluate_physical_records(&records, &ValidationTargets::m10_reference(), revision);
        assert!(!report.accepted);
        assert!(report.failures.iter().any(|failure| matches!(
            failure,
            EvidenceGateFailure::BuildRevisionMismatch { .. }
        )));
        assert!(report.failures.iter().any(|failure| matches!(
            failure,
            EvidenceGateFailure::DuplicateFingerprint(_)
        )));
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
