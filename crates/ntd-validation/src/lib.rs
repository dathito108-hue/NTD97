#![forbid(unsafe_code)]

mod evidence;
mod matrix;
mod reuse;
mod score;
mod soak;
mod trace;
mod workload;

pub use evidence::{
    decode_m13_hardware_evidence, decode_physical_evidence, encode_m13_hardware_evidence,
    encode_physical_evidence, evaluate_physical_records, DeviceEvidence, EvidenceClass,
    EvidenceGateFailure, EvidenceGateReport, EvidenceMatrix, M13HardwareEvidenceRecord,
    PhysicalEvidenceRecord, ValidationTargets, M13_GATE_APP_ACCESSIBILITY, M13_GATE_BROWSER,
    M13_GATE_DEVICE_CLIPBOARD, M13_GATE_MIXED_SEQUENCE, M13_GATE_PC_OBSERVE, M13_GATE_PC_WRITE,
    M13_GATE_SAF_READ, M13_GATE_SAF_WRITE, M13_GATE_UPLOAD_RESTORE, M13_GATE_WEB_SEARCH,
    M13_MIN_VERIFIED_ACTIONS, M13_REQUIRED_GATE_MASK,
};
pub use matrix::{
    evaluate_device_profile, representative_device_profiles, DeviceMatrixReport,
    RepresentativeDevice,
};
pub use reuse::{analyze_prefix_reuse, verify_paged_round_trip, PagingReport, PrefixReuseReport};
pub use score::GeneralAgentScorecard;
pub use soak::{run_logical_continuity_soak, ContinuitySoakReport};
pub use trace::{EnergySampler, NoEnergySampler, TraceRecorder, TraceSpan};
pub use workload::run_native_validation_workload;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    InvalidConfiguration,
    MobileCompute(String),
    Continuity(String),
    PagingMismatch,
    EvidenceCodec,
    RuntimeWorkload(String),
}
