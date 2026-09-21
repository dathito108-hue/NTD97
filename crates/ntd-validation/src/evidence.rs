#![forbid(unsafe_code)]

use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceClass {
    CiSurrogate,
    PhysicalDevice,
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
pub struct ValidationTargets {
    pub required_physical_profiles: Vec<String>,
    pub max_p95_latency_nanos: u64,
    pub max_energy_per_task_microjoules: u64,
    pub min_reliability_permille: u16,
    pub min_recovery_permille: u16,
}

impl ValidationTargets {
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
                    && entry.p95_latency_nanos <= targets.max_p95_latency_nanos
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
