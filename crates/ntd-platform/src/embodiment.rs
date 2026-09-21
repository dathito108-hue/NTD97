#![forbid(unsafe_code)]

use ntd_runtime::{ResourceSnapshot, ThermalState};

use crate::ContinuityPhase;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AssistantMode {
    Sleeping = 1,
    Idle = 2,
    Listening = 3,
    Thinking = 4,
    Acting = 5,
    WaitingApproval = 6,
    Reconstructing = 7,
    Success = 8,
    Error = 9,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AssistantExpression {
    Neutral = 1,
    Listening = 2,
    Focused = 3,
    Concerned = 4,
    Joy = 5,
    Alert = 6,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AssistantGesture {
    None = 1,
    Wave = 2,
    Working = 3,
    Confirm = 4,
    Point = 5,
    Warning = 6,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum VoiceMode {
    Silent = 1,
    Listening = 2,
    Speaking = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum LipViseme {
    Closed = 1,
    Open = 2,
    Wide = 3,
    Rounded = 4,
    Teeth = 5,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GazeTarget {
    pub x_milli: i16,
    pub y_milli: i16,
    pub z_milli: i16,
}

impl GazeTarget {
    pub const FORWARD: Self = Self {
        x_milli: 0,
        y_milli: 0,
        z_milli: 1000,
    };

    pub fn normalized(self) -> Self {
        Self {
            x_milli: self.x_milli.clamp(-1000, 1000),
            y_milli: self.y_milli.clamp(-1000, 1000),
            z_milli: self.z_milli.clamp(-1000, 1000),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderQuality {
    Suspended,
    Minimal,
    Balanced,
    Full,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderBudget {
    pub target_fps: u8,
    pub quality: RenderQuality,
    pub shadows: bool,
    pub secondary_motion: bool,
}

impl RenderBudget {
    pub fn for_resources(resources: ResourceSnapshot, visible: bool) -> Self {
        if !visible || resources.thermal == ThermalState::Critical {
            return Self {
                target_fps: 0,
                quality: RenderQuality::Suspended,
                shadows: false,
                secondary_motion: false,
            };
        }

        if resources.thermal == ThermalState::Hot
            || (resources.battery_percent <= 10 && !resources.charging)
        {
            return Self {
                target_fps: 15,
                quality: RenderQuality::Minimal,
                shadows: false,
                secondary_motion: false,
            };
        }

        if resources.thermal == ThermalState::Warm
            || (resources.battery_percent <= 25 && !resources.charging)
        {
            return Self {
                target_fps: 30,
                quality: RenderQuality::Balanced,
                shadows: false,
                secondary_motion: true,
            };
        }

        Self {
            target_fps: 60,
            quality: RenderQuality::Full,
            shadows: true,
            secondary_motion: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbodimentFrame {
    pub mode: AssistantMode,
    pub expression: AssistantExpression,
    pub gesture: AssistantGesture,
    pub voice: VoiceMode,
    pub viseme: LipViseme,
    pub lip_intensity: u16,
    pub gaze: GazeTarget,
    pub render: RenderBudget,
    pub progress_permille: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssistantInput {
    Tap,
    LongPress,
    Approve,
    Cancel,
    Gaze(GazeTarget),
    VoiceListening(bool),
    SpeechFrame {
        viseme: LipViseme,
        intensity: u16,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbodimentController {
    frame: EmbodimentFrame,
}

impl EmbodimentController {
    pub fn new(resources: ResourceSnapshot, visible: bool) -> Self {
        Self {
            frame: EmbodimentFrame {
                mode: AssistantMode::Idle,
                expression: AssistantExpression::Neutral,
                gesture: AssistantGesture::None,
                voice: VoiceMode::Silent,
                viseme: LipViseme::Closed,
                lip_intensity: 0,
                gaze: GazeTarget::FORWARD,
                render: RenderBudget::for_resources(resources, visible),
                progress_permille: 0,
            },
        }
    }

    pub fn frame(&self) -> EmbodimentFrame {
        self.frame
    }

    pub fn sync_continuity(
        &mut self,
        phase: ContinuityPhase,
        resources: ResourceSnapshot,
        visible: bool,
        progress_permille: u16,
    ) -> EmbodimentFrame {
        self.frame.render = RenderBudget::for_resources(resources, visible);
        self.frame.progress_permille = progress_permille.min(1000);

        let (mode, expression, gesture) = match phase {
            ContinuityPhase::Interactive => (
                AssistantMode::Idle,
                AssistantExpression::Neutral,
                AssistantGesture::None,
            ),
            ContinuityPhase::ActiveExecution => (
                AssistantMode::Acting,
                AssistantExpression::Focused,
                AssistantGesture::Working,
            ),
            ContinuityPhase::Checkpointed | ContinuityPhase::SuspendedByOs => (
                AssistantMode::Sleeping,
                AssistantExpression::Neutral,
                AssistantGesture::None,
            ),
            ContinuityPhase::WaitingCondition => (
                AssistantMode::Thinking,
                AssistantExpression::Focused,
                AssistantGesture::None,
            ),
            ContinuityPhase::WaitingApproval => (
                AssistantMode::WaitingApproval,
                AssistantExpression::Alert,
                AssistantGesture::Point,
            ),
            ContinuityPhase::Reconstructing | ContinuityPhase::VerifyingResume => (
                AssistantMode::Reconstructing,
                AssistantExpression::Focused,
                AssistantGesture::Working,
            ),
            ContinuityPhase::Completed => (
                AssistantMode::Success,
                AssistantExpression::Joy,
                AssistantGesture::Confirm,
            ),
            ContinuityPhase::FailedRecoverable => (
                AssistantMode::Error,
                AssistantExpression::Concerned,
                AssistantGesture::Warning,
            ),
        };

        self.frame.mode = mode;
        self.frame.expression = expression;
        self.frame.gesture = gesture;
        self.frame
    }

    pub fn apply_input(&mut self, input: AssistantInput) -> EmbodimentFrame {
        match input {
            AssistantInput::Tap => {
                self.frame.gesture = AssistantGesture::Wave;
            }
            AssistantInput::LongPress => {
                self.frame.mode = AssistantMode::Listening;
                self.frame.expression = AssistantExpression::Listening;
                self.frame.voice = VoiceMode::Listening;
            }
            AssistantInput::Approve => {
                self.frame.gesture = AssistantGesture::Confirm;
                self.frame.expression = AssistantExpression::Focused;
            }
            AssistantInput::Cancel => {
                self.frame.gesture = AssistantGesture::Warning;
                self.frame.expression = AssistantExpression::Concerned;
            }
            AssistantInput::Gaze(target) => {
                self.frame.gaze = target.normalized();
            }
            AssistantInput::VoiceListening(listening) => {
                self.frame.voice = if listening {
                    VoiceMode::Listening
                } else {
                    VoiceMode::Silent
                };
                if listening {
                    self.frame.mode = AssistantMode::Listening;
                    self.frame.expression = AssistantExpression::Listening;
                }
            }
            AssistantInput::SpeechFrame { viseme, intensity } => {
                self.frame.voice = VoiceMode::Speaking;
                self.frame.viseme = viseme;
                self.frame.lip_intensity = intensity.min(1000);
            }
        }

        self.frame
    }

    pub fn end_speech(&mut self) -> EmbodimentFrame {
        self.frame.voice = VoiceMode::Silent;
        self.frame.viseme = LipViseme::Closed;
        self.frame.lip_intensity = 0;
        self.frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resources(thermal: ThermalState, battery_percent: u8) -> ResourceSnapshot {
        ResourceSnapshot {
            available_ram_bytes: 4 * 1024 * 1024 * 1024,
            battery_percent,
            charging: false,
            thermal,
            latency_budget_ms: 100,
        }
    }

    #[test]
    fn thermal_pressure_can_suspend_render_without_changing_agent_state() {
        let mut controller =
            EmbodimentController::new(resources(ThermalState::Nominal, 80), true);
        let frame = controller.sync_continuity(
            ContinuityPhase::ActiveExecution,
            resources(ThermalState::Critical, 80),
            true,
            400,
        );

        assert_eq!(frame.mode, AssistantMode::Acting);
        assert_eq!(frame.render.quality, RenderQuality::Suspended);
        assert_eq!(frame.render.target_fps, 0);
    }

    #[test]
    fn speech_frame_drives_bounded_lip_state() {
        let mut controller =
            EmbodimentController::new(resources(ThermalState::Nominal, 80), true);
        let frame = controller.apply_input(AssistantInput::SpeechFrame {
            viseme: LipViseme::Wide,
            intensity: 2000,
        });
        assert_eq!(frame.voice, VoiceMode::Speaking);
        assert_eq!(frame.lip_intensity, 1000);
    }
}
