#![forbid(unsafe_code)]

mod bundle;
mod continuity;
mod embodiment;

pub use bundle::{
    decode_continuity_bundle, encode_continuity_bundle, BundleError, ContinuityBundle,
    CNT97_HEADER_LEN, CNT97_MAGIC, CNT97_MAJOR, CNT97_MINOR,
};
pub use continuity::{
    ApprovalRequest, CheckpointStore, ContinuityError, ContinuityPhase, ContinuitySupervisor,
    InMemoryCheckpointStore, PlatformConstraints, PlatformContinuityState, RetryBackoff,
    WakeDisposition, WakeReason,
};
pub use embodiment::{
    AssistantExpression, AssistantGesture, AssistantInput, AssistantMode, EmbodimentController,
    EmbodimentFrame, GazeTarget, LipViseme, RenderBudget, RenderQuality, VoiceMode,
};
