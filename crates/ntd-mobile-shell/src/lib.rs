#![forbid(unsafe_code)]

mod avatar;
mod continuity;
mod media;
mod scheduler;

pub use avatar::{
    AvatarController, AvatarExpression, AvatarFrame, AvatarGesture, AvatarMode, AvatarRenderProfile,
    AvatarSurface, GazeTarget, LipSyncState,
};
pub use continuity::{
    build_mobile_continuity_bundle, decode_mobile_continuity_bundle,
    encode_mobile_continuity_bundle, restore_mobile_continuity_bundle, MobileContinuityBundle,
    MobileContinuityError, MobileContinuityState, PendingApproval, RestoredMobileSession,
    RetryBackoff, WakeReason, MCS97_HEADER_LEN, MCS97_MAGIC, MCS97_MAJOR, MCS97_MINOR,
};

pub use media::{MediaEvent, VoiceFrame, VoiceInputState, VoiceOutputState, VoiceStateMachine};
pub use scheduler::{
    choose_platform_directive, PlatformDirective, WakeContext, WakePolicyError,
};
