#![forbid(unsafe_code)]

mod evidence;
mod matrix;
mod reuse;
mod score;
mod soak;
mod trace;

pub use evidence::{DeviceEvidence, EvidenceClass, EvidenceMatrix, ValidationTargets};
pub use matrix::{
    evaluate_device_profile, representative_device_profiles, DeviceMatrixReport,
    RepresentativeDevice,
};
pub use reuse::{analyze_prefix_reuse, verify_paged_round_trip, PagingReport, PrefixReuseReport};
pub use score::GeneralAgentScorecard;
pub use soak::{run_logical_continuity_soak, ContinuitySoakReport};
pub use trace::{EnergySampler, NoEnergySampler, TraceRecorder, TraceSpan};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    InvalidConfiguration,
    MobileCompute(String),
    Continuity(String),
    PagingMismatch,
}
