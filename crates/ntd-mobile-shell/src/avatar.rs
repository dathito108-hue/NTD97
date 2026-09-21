#![forbid(unsafe_code)]

use ntd_runtime::{ActionPlanState, ActionPlanStatus, ResourceSnapshot, ThermalState};

use crate::MobileContinuityState;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AvatarMode {
    Idle = 1,
    Active = 2,
    Thinking = 3,
    Acting = 4,
    WaitingApproval = 5,
    Sleeping = 6,
    Recovering = 7,
    Completed = 8,
    Error = 9,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AvatarExpression {
    Neutral = 1,
    Focused = 2,
    Attentive = 3,
    Confirming = 4,
    Concerned = 5,
    Positive = 6,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AvatarGesture {
    None = 0,
    Acknowledge = 1,
    Present = 2,
    Working = 3,
    AskApproval = 4,
    Success = 5,
    Warning = 6,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AvatarSurface {
    InApp = 1,
    Overlay = 2,
    Bubble = 3,
    VoiceOnly = 4,
    NotificationOnly = 5,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GazeTarget {
    pub x_milli: i16,
    pub y_milli: i16,
    pub z_milli: i16,
}

impl GazeTarget {
    pub const CENTER: Self = Self {
        x_milli: 0,
        y_milli: 0,
        z_milli: 1000,
    };

    pub fn clamped(self) -> Self {
        Self {
            x_milli: self.x_milli.clamp(-1000, 1000),
            y_milli: self.y_milli.clamp(-1000, 1000),
            z_milli: self.z_milli.clamp(-1000, 1000),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LipSyncState {
    pub amplitude_milli: u16,
    pub viseme: u8,
    pub speaking: bool,
}

impl LipSyncState {
    pub const SILENT: Self = Self {
        amplitude_milli: 0,
        viseme: 0,
        speaking: false,
    };

    pub fn normalized(self) -> Self {
        Self {
            amplitude_milli: self.amplitude_milli.min(1000),
            viseme: self.viseme.min(31),
            speaking: self.speaking,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AvatarRenderProfile {
    pub target_fps: u16,
    pub geometry_level: u8,
    pub effects_level: u8,
    pub update_hz: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AvatarFrame {
    pub mode: AvatarMode,
    pub expression: AvatarExpression,
    pub gesture: AvatarGesture,
    pub gaze: GazeTarget,
    pub lip_sync: LipSyncState,
    pub surface: AvatarSurface,
    pub render: AvatarRenderProfile,
    pub task_id: Option<u64>,
    pub action_cursor: Option<usize>,
    pub action_total: Option<usize>,
    pub status_text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AvatarController {
    surface: AvatarSurface,
    gaze: GazeTarget,
    lip_sync: LipSyncState,
}

impl AvatarController {
    pub fn new(surface: AvatarSurface) -> Self {
        Self {
            surface,
            gaze: GazeTarget::CENTER,
            lip_sync: LipSyncState::SILENT,
        }
    }

    pub fn set_surface(&mut self, surface: AvatarSurface) {
        self.surface = surface;
    }

    pub fn set_gaze(&mut self, gaze: GazeTarget) {
        self.gaze = gaze.clamped();
    }

    pub fn set_lip_sync(&mut self, lip_sync: LipSyncState) {
        self.lip_sync = lip_sync.normalized();
    }

    pub fn frame(
        &self,
        continuity: MobileContinuityState,
        action_plan: Option<&ActionPlanState>,
        resources: ResourceSnapshot,
    ) -> AvatarFrame {
        let render = render_profile(resources);
        let (mode, expression, gesture, status_text) =
            derive_pose(continuity, action_plan.map(|plan| plan.status));

        AvatarFrame {
            mode,
            expression,
            gesture,
            gaze: self.gaze,
            lip_sync: self.lip_sync,
            surface: self.surface,
            render,
            task_id: action_plan.map(|plan| plan.task_id),
            action_cursor: action_plan.map(|plan| plan.cursor),
            action_total: action_plan.map(|plan| plan.actions.len()),
            status_text,
        }
    }
}

fn derive_pose(
    continuity: MobileContinuityState,
    plan_status: Option<ActionPlanStatus>,
) -> (AvatarMode, AvatarExpression, AvatarGesture, String) {
    if let Some(status) = plan_status {
        match status {
            ActionPlanStatus::Ready => {
                return (
                    AvatarMode::Acting,
                    AvatarExpression::Focused,
                    AvatarGesture::Working,
                    "Executing task".into(),
                )
            }
            ActionPlanStatus::Suspended => {
                return (
                    AvatarMode::Recovering,
                    AvatarExpression::Attentive,
                    AvatarGesture::Acknowledge,
                    "Task paused".into(),
                )
            }
            ActionPlanStatus::Completed => {
                return (
                    AvatarMode::Completed,
                    AvatarExpression::Positive,
                    AvatarGesture::Success,
                    "Task completed".into(),
                )
            }
            ActionPlanStatus::Failed => {
                return (
                    AvatarMode::Error,
                    AvatarExpression::Concerned,
                    AvatarGesture::Warning,
                    "Task needs recovery".into(),
                )
            }
            ActionPlanStatus::RolledBack => {
                return (
                    AvatarMode::Recovering,
                    AvatarExpression::Concerned,
                    AvatarGesture::Acknowledge,
                    "Changes rolled back".into(),
                )
            }
        }
    }

    match continuity {
        MobileContinuityState::Interactive => (
            AvatarMode::Active,
            AvatarExpression::Attentive,
            AvatarGesture::None,
            "Ready".into(),
        ),
        MobileContinuityState::ActiveExecution => (
            AvatarMode::Thinking,
            AvatarExpression::Focused,
            AvatarGesture::Working,
            "Thinking".into(),
        ),
        MobileContinuityState::Checkpointed
        | MobileContinuityState::SuspendedByOs
        | MobileContinuityState::WaitingCondition => (
            AvatarMode::Sleeping,
            AvatarExpression::Neutral,
            AvatarGesture::None,
            "Standing by".into(),
        ),
        MobileContinuityState::WaitingApproval => (
            AvatarMode::WaitingApproval,
            AvatarExpression::Confirming,
            AvatarGesture::AskApproval,
            "Approval required".into(),
        ),
        MobileContinuityState::Reconstructing | MobileContinuityState::VerifyingResume => (
            AvatarMode::Recovering,
            AvatarExpression::Focused,
            AvatarGesture::Acknowledge,
            "Restoring task".into(),
        ),
        MobileContinuityState::Completed => (
            AvatarMode::Completed,
            AvatarExpression::Positive,
            AvatarGesture::Success,
            "Completed".into(),
        ),
        MobileContinuityState::FailedRecoverable => (
            AvatarMode::Error,
            AvatarExpression::Concerned,
            AvatarGesture::Warning,
            "Recovery available".into(),
        ),
    }
}

fn render_profile(resources: ResourceSnapshot) -> AvatarRenderProfile {
    let resources = resources.normalized();

    if resources.thermal == ThermalState::Critical || resources.battery_percent <= 10 {
        AvatarRenderProfile {
            target_fps: 5,
            geometry_level: 1,
            effects_level: 0,
            update_hz: 2,
        }
    } else if resources.thermal == ThermalState::Hot || resources.battery_percent <= 20 {
        AvatarRenderProfile {
            target_fps: 15,
            geometry_level: 1,
            effects_level: 1,
            update_hz: 5,
        }
    } else if resources.thermal == ThermalState::Warm || resources.battery_percent <= 40 {
        AvatarRenderProfile {
            target_fps: 30,
            geometry_level: 2,
            effects_level: 1,
            update_hz: 15,
        }
    } else {
        AvatarRenderProfile {
            target_fps: 60,
            geometry_level: 3,
            effects_level: 2,
            update_hz: 30,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ntd_runtime::ThermalState;

    fn snapshot(battery_percent: u8, thermal: ThermalState) -> ResourceSnapshot {
        ResourceSnapshot {
            available_ram_bytes: 4 * 1024 * 1024 * 1024,
            battery_percent,
            charging: false,
            thermal,
            latency_budget_ms: 100,
        }
    }

    #[test]
    fn critical_resources_reduce_render_cost() {
        let controller = AvatarController::new(AvatarSurface::InApp);
        let frame = controller.frame(
            MobileContinuityState::Interactive,
            None,
            snapshot(5, ThermalState::Critical),
        );

        assert_eq!(frame.render.target_fps, 5);
        assert_eq!(frame.render.effects_level, 0);
    }

    #[test]
    fn waiting_approval_drives_explicit_avatar_state() {
        let controller = AvatarController::new(AvatarSurface::Overlay);
        let frame = controller.frame(
            MobileContinuityState::WaitingApproval,
            None,
            snapshot(80, ThermalState::Nominal),
        );

        assert_eq!(frame.mode, AvatarMode::WaitingApproval);
        assert_eq!(frame.gesture, AvatarGesture::AskApproval);
        assert_eq!(frame.status_text, "Approval required");
    }

    #[test]
    fn gaze_and_lip_sync_are_normalized() {
        let mut controller = AvatarController::new(AvatarSurface::Bubble);
        controller.set_gaze(GazeTarget {
            x_milli: 2000,
            y_milli: -2000,
            z_milli: 1500,
        });
        controller.set_lip_sync(LipSyncState {
            amplitude_milli: 2000,
            viseme: 99,
            speaking: true,
        });

        let frame = controller.frame(
            MobileContinuityState::Interactive,
            None,
            snapshot(90, ThermalState::Nominal),
        );

        assert_eq!(frame.gaze.x_milli, 1000);
        assert_eq!(frame.gaze.y_milli, -1000);
        assert_eq!(frame.lip_sync.amplitude_milli, 1000);
        assert_eq!(frame.lip_sync.viseme, 31);
    }
}
