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

    final class ChatEvent {
        public static final int TOKEN = 1;
        public static final int COMPLETE = 2;
        public static final int CANCELLED = 3;
        public static final int ERROR = 4;
        public static final int APPROVAL_REQUIRED = 5;
        public static final int ACTION_CHECKPOINTED = 6;

        public final int kind;
        public final int tokenId;
        public final String text;

        public ChatEvent(int kind, int tokenId, String text) {
            this.kind = kind;
            this.tokenId = tokenId;
            this.text = text;
        }
    }

    ResumeResult restoreAndVerify(byte[] mcs97, String wakeReason);

    byte[] checkpoint();

    boolean resolveApproval(boolean approved);

    default boolean chatReady() {
        return false;
    }

    default long submitChat(String prompt, int maxNewTokens) {
        return -1L;
    }

    default ChatEvent nextChatEvent(long requestId) {
        return new ChatEvent(ChatEvent.ERROR, -1, "Native chat unavailable");
    }

    default boolean cancelChat(long requestId) {
        return false;
    }

    default boolean resolveChatApproval(long requestId, boolean approved) {
        return false;
    }

    default int chatStatus(long requestId) {
        return 0;
    }

    default String chatLastError() {
        return "";
    }

    default int chatReasoningBudget(long requestId) {
        return 0;
    }

    default int chatRecalledMemoryItems(long requestId) {
        return 0;
    }

    default int chatMemoryRecordCount(long requestId) {
        return 0;
    }

    default int chatReasoningIterations(long requestId) {
        return 0;
    }

    default int chatActionPlannerStatus(long requestId) {
        return 0;
    }

    default int chatActionCount(long requestId) {
        return 0;
    }

    default int chatVerifiedActionCount(long requestId) {
        return 0;
    }

    default boolean chatVerifiedSynthesisReady(long requestId) {
        return false;
    }

    default byte[] chatCheckpoint() {
        return new byte[0];
    }

    default long restoreChatCheckpoint(byte[] checkpoint) {
        return -1L;
    }

    default String chatTranscript() {
        return "";
    }

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
