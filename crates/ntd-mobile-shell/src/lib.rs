#![forbid(unsafe_code)]

mod avatar;
mod continuity;

pub use avatar::{
    AvatarController, AvatarExpression, AvatarFrame, AvatarGesture, AvatarMode, AvatarRenderProfile,
    AvatarSurface, GazeTarget, LipSyncState,
};
pub use continuity::{
    decode_mobile_continuity_bundle, encode_mobile_continuity_bundle, MobileContinuityBundle,
    MobileContinuityError, MobileContinuityState, PendingApproval, RestoredMobileSession,
    RetryBackoff, WakeReason, MCS97_HEADER_LEN, MCS97_MAGIC, MCS97_MAJOR, MCS97_MINOR,
};
