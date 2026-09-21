package ai.ntd97.mobile;

public interface NtdRuntimeHost {
    final class ResumeResult {
        public final boolean restored;
        public final boolean hasEligibleWork;
        public final boolean requiresForeground;
        public final String status;
        public final Approval approval;

        public ResumeResult(
                boolean restored,
                boolean hasEligibleWork,
                boolean requiresForeground,
                String status,
                Approval approval) {
            this.restored = restored;
            this.hasEligibleWork = hasEligibleWork;
            this.requiresForeground = requiresForeground;
            this.status = status;
            this.approval = approval;
        }
    }

    final class Approval {
        public final String capability;
        public final String rationale;

        public Approval(String capability, String rationale) {
            this.capability = capability;
            this.rationale = rationale;
        }
    }

    ResumeResult restoreAndVerify(byte[] mcs97, String wakeReason);

    byte[] checkpoint();

    boolean resolveApproval(boolean approved);

    default void acceptMicrophonePcm(short[] samples, int sampleRateHz) {}

    default short[] pullSpeakerPcm(int maxSamples, int sampleRateHz) {
        return null;
    }

    AvatarState avatarState();

    final class AvatarState {
        public final int mode;
        public final int expression;
        public final int gesture;
        public final int gazeX;
        public final int gazeY;
        public final int gazeZ;
        public final int lipAmplitude;
        public final int viseme;
        public final boolean speaking;
        public final int targetFps;
        public final String status;

        public AvatarState(
                int mode,
                int expression,
                int gesture,
                int gazeX,
                int gazeY,
                int gazeZ,
                int lipAmplitude,
                int viseme,
                boolean speaking,
                int targetFps,
                String status) {
            this.mode = mode;
            this.expression = expression;
            this.gesture = gesture;
            this.gazeX = gazeX;
            this.gazeY = gazeY;
            this.gazeZ = gazeZ;
            this.lipAmplitude = lipAmplitude;
            this.viseme = viseme;
            this.speaking = speaking;
            this.targetFps = targetFps;
            this.status = status;
        }

        public static AvatarState idle() {
            return new AvatarState(
                    1, 1, 0,
                    0, 0, 1000,
                    0, 0, false,
                    30,
                    "Ready");
        }
    }
}
