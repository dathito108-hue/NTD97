#![forbid(unsafe_code)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum VoiceInputState {
    Idle = 1,
    Listening = 2,
    Capturing = 3,
    Decoding = 4,
    Failed = 5,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum VoiceOutputState {
    Silent = 1,
    Preparing = 2,
    Speaking = 3,
    Paused = 4,
    Failed = 5,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoiceFrame {
    pub input: VoiceInputState,
    pub output: VoiceOutputState,
    pub transcript: Option<String>,
    pub spoken_text: Option<String>,
    pub amplitude_milli: u16,
    pub viseme: u8,
}

impl Default for VoiceFrame {
    fn default() -> Self {
        Self {
            input: VoiceInputState::Idle,
            output: VoiceOutputState::Silent,
            transcript: None,
            spoken_text: None,
            amplitude_milli: 0,
            viseme: 0,
        }
    }
}

impl VoiceFrame {
    pub fn normalized(mut self) -> Self {
        self.amplitude_milli = self.amplitude_milli.min(1000);
        self.viseme = self.viseme.min(31);

        if self.output != VoiceOutputState::Speaking {
            self.amplitude_milli = 0;
            self.viseme = 0;
        }

        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediaEvent {
    VoiceInputStarted,
    VoiceInputStopped,
    TranscriptReady(String),
    SpeechStarted(String),
    SpeechProgress { amplitude_milli: u16, viseme: u8 },
    SpeechFinished,
    Failure(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VoiceStateMachine {
    frame: VoiceFrame,
}

impl VoiceStateMachine {
    pub fn frame(&self) -> &VoiceFrame {
        &self.frame
    }

    pub fn apply(&mut self, event: MediaEvent) {
        match event {
            MediaEvent::VoiceInputStarted => {
                self.frame.input = VoiceInputState::Listening;
                self.frame.transcript = None;
            }
            MediaEvent::VoiceInputStopped => {
                self.frame.input = VoiceInputState::Decoding;
            }
            MediaEvent::TranscriptReady(transcript) => {
                self.frame.input = VoiceInputState::Idle;
                self.frame.transcript = Some(transcript);
            }
            MediaEvent::SpeechStarted(text) => {
                self.frame.output = VoiceOutputState::Speaking;
                self.frame.spoken_text = Some(text);
            }
            MediaEvent::SpeechProgress {
                amplitude_milli,
                viseme,
            } => {
                self.frame.output = VoiceOutputState::Speaking;
                self.frame.amplitude_milli = amplitude_milli.min(1000);
                self.frame.viseme = viseme.min(31);
            }
            MediaEvent::SpeechFinished => {
                self.frame.output = VoiceOutputState::Silent;
                self.frame.amplitude_milli = 0;
                self.frame.viseme = 0;
            }
            MediaEvent::Failure(_) => {
                self.frame.input = VoiceInputState::Failed;
                self.frame.output = VoiceOutputState::Failed;
                self.frame.amplitude_milli = 0;
                self.frame.viseme = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speech_progress_drives_lip_sync_bounds() {
        let mut voice = VoiceStateMachine::default();
        voice.apply(MediaEvent::SpeechStarted("hello".into()));
        voice.apply(MediaEvent::SpeechProgress {
            amplitude_milli: 5_000,
            viseme: 200,
        });

        assert_eq!(voice.frame().output, VoiceOutputState::Speaking);
        assert_eq!(voice.frame().amplitude_milli, 1000);
        assert_eq!(voice.frame().viseme, 31);

        voice.apply(MediaEvent::SpeechFinished);
        assert_eq!(voice.frame().amplitude_milli, 0);
    }
}
