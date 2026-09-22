package ai.ntd97.mobile;

import android.content.Context;

import java.io.File;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

final class NtdNativeRuntimeHost implements NtdRuntimeHost {
    private static final int PROTOCOL_VERSION = 1;
    private static boolean loadAttempted;
    private static boolean loaded;

    private final Context context;
    private boolean chatModelReady;

    private NtdNativeRuntimeHost(Context context) {
        this.context = context.getApplicationContext();
        NtdDeviceAppPlatform.initialize(this.context);
    }

    static NtdNativeRuntimeHost create(Context context) {
        if (!ensureLoaded()) {
            return null;
        }
        return new NtdNativeRuntimeHost(context);
    }

    static byte[] runRealModelProbe(
            String capsulePath,
            String shardRoot,
            byte[] verifyKey,
            String expectedTokenIds) {
        if (!ensureLoaded()) {
            return "android_real_model=failed\nerror=native library unavailable\n"
                    .getBytes(StandardCharsets.UTF_8);
        }
        byte[] result = nativeRealModelProbe(
                capsulePath,
                shardRoot,
                verifyKey,
                expectedTokenIds);
        return result == null ? new byte[0] : result;
    }

    static byte[] runProductionCapabilityProbe(Context context) {
        NtdDeviceAppPlatform.initialize(context);
        if (!ensureLoaded()) {
            return "production_capabilities=failed\nerror=native library unavailable\n"
                    .getBytes(StandardCharsets.UTF_8);
        }
        File root = new File(context.getFilesDir(), "ntd97-capability-probe");
        byte[] result = nativeProductionCapabilityProbe(
                root.getAbsolutePath(),
                context.getPackageName());
        return result == null ? new byte[0] : result;
    }

    static String provisionPcPair(
            Context context,
            String peer,
            String address,
            String remotePeerId,
            String remoteVerifyKey) {
        if (!ensureLoaded()) {
            return "ERROR:native library unavailable";
        }
        byte[] result = nativeProvisionPcPairProfile(
                pcCapabilityRoot(context).getAbsolutePath(),
                peer == null ? "" : peer.trim(),
                address == null ? "" : address.trim(),
                remotePeerId == null ? "" : remotePeerId.trim(),
                remoteVerifyKey == null ? "" : remoteVerifyKey.trim());
        return utf8Result(result);
    }

    static String describePcPair(Context context, String peer) {
        if (!ensureLoaded()) {
            return "ERROR:native library unavailable";
        }
        byte[] result = nativeDescribePcPairProfile(
                pcCapabilityRoot(context).getAbsolutePath(),
                peer == null ? "" : peer.trim());
        return utf8Result(result);
    }

    static String revokePcPair(Context context, String peer) {
        if (!ensureLoaded()) {
            return "ERROR:native library unavailable";
        }
        byte[] result = nativeRevokePcPairProfile(
                pcCapabilityRoot(context).getAbsolutePath(),
                peer == null ? "" : peer.trim());
        return utf8Result(result);
    }

    static String listPcPairs(Context context) {
        if (!ensureLoaded()) {
            return "ERROR:native library unavailable";
        }
        return utf8Result(nativeListPcPairProfiles(pcCapabilityRoot(context).getAbsolutePath()));
    }

    private static File pcCapabilityRoot(Context context) {
        return new File(context.getFilesDir(), "ntd97-capability-files");
    }

    private static String utf8Result(byte[] result) {
        return result == null ? "ERROR:empty native result" : new String(result, StandardCharsets.UTF_8);
    }

    private static synchronized boolean ensureLoaded() {
        if (!loadAttempted) {
            loadAttempted = true;
            try {
                System.loadLibrary("ntd97_android");
                loaded = true;
            } catch (UnsatisfiedLinkError error) {
                loaded = false;
            }
        }
        return loaded;
    }

    @Override
    public ResumeResult restoreAndVerify(byte[] mcs97, String wakeReason) {
        if (mcs97 == null || mcs97.length == 0) {
            return new ResumeResult(false, false, false, "No native checkpoint", null);
        }

        pushResourceSnapshot();
        byte[] encoded = nativeRestoreAndVerify(
                mcs97,
                wakeReason == null ? "scheduled" : wakeReason);
        return decodeRestoreResult(encoded);
    }

    @Override
    public byte[] checkpoint() {
        byte[] checkpoint = nativeCheckpoint();
        return checkpoint == null ? new byte[0] : checkpoint;
    }

    @Override
    public boolean resolveApproval(boolean approved) {
        return nativeResolveApproval(approved);
    }

    @Override
    public synchronized boolean chatReady() {
        if (chatModelReady) {
            return true;
        }

        try {
            NtdNativeModelStore.ModelFiles model =
                    NtdNativeModelStore.prepareBundledModel(context);
            if (model == null) {
                return false;
            }
            chatModelReady = nativeOpenChatModel(
                    "model.ntd97.stories260k",
                    1,
                    model.capsule.getAbsolutePath(),
                    model.shardRoot.getAbsolutePath(),
                    new File(context.getFilesDir(), "ntd97-capability-files").getAbsolutePath(),
                    model.verifyKey,
                    128);
            return chatModelReady;
        } catch (java.io.IOException error) {
            chatModelReady = false;
            return false;
        }
    }

    @Override
    public long submitChat(String prompt, int maxNewTokens) {
        if (!chatReady() || prompt == null || prompt.trim().isEmpty()) {
            return -1L;
        }
        return nativeSubmitChat(prompt, Math.max(1, maxNewTokens));
    }

    @Override
    public ChatEvent nextChatEvent(long requestId) {
        return decodeChatEvent(nativeNextChatEvent(requestId));
    }

    @Override
    public boolean cancelChat(long requestId) {
        return nativeCancelChat(requestId);
    }

    @Override
    public boolean resolveChatApproval(long requestId, boolean approved) {
        return nativeResolveChatApproval(requestId, approved);
    }

    @Override
    public int chatStatus(long requestId) {
        return nativeChatStatus(requestId);
    }

    @Override
    public String chatLastError() {
        byte[] encoded = nativeChatLastError();
        return encoded == null ? "" : new String(encoded, StandardCharsets.UTF_8);
    }

    @Override
    public int chatReasoningBudget(long requestId) {
        return nativeChatReasoningBudget(requestId);
    }

    @Override
    public int chatRecalledMemoryItems(long requestId) {
        return nativeChatRecalledMemoryItems(requestId);
    }

    @Override
    public int chatMemoryRecordCount(long requestId) {
        return nativeChatMemoryRecordCount(requestId);
    }

    @Override
    public int chatReasoningIterations(long requestId) {
        return nativeChatReasoningIterations(requestId);
    }

    @Override
    public int chatActionPlannerStatus(long requestId) {
        return nativeChatActionPlannerStatus(requestId);
    }

    @Override
    public int chatActionCount(long requestId) {
        return nativeChatActionCount(requestId);
    }

    @Override
    public int chatVerifiedActionCount(long requestId) {
        return nativeChatVerifiedActionCount(requestId);
    }

    @Override
    public boolean chatVerifiedSynthesisReady(long requestId) {
        return nativeChatVerifiedSynthesisReady(requestId);
    }

    @Override
    public byte[] chatCheckpoint() {
        byte[] checkpoint = nativeChatCheckpoint();
        return checkpoint == null ? new byte[0] : checkpoint;
    }

    @Override
    public long restoreChatCheckpoint(byte[] checkpoint) {
        if (checkpoint == null || checkpoint.length == 0) {
            return 0L;
        }
        return nativeRestoreChatCheckpoint(checkpoint);
    }

    @Override
    public String chatTranscript() {
        byte[] transcript = nativeChatTranscript();
        return transcript == null
                ? ""
                : new String(transcript, StandardCharsets.UTF_8);
    }

    @Override
    public void acceptMicrophonePcm(short[] samples, int sampleRateHz) {
        if (samples == null || samples.length == 0) {
            return;
        }

        ByteBuffer pcm = ByteBuffer
                .allocate(samples.length * 2)
                .order(ByteOrder.LITTLE_ENDIAN);
        for (short sample : samples) {
            pcm.putShort(sample);
        }
        nativeAcceptMicrophonePcm(pcm.array(), sampleRateHz);
    }

    @Override
    public short[] pullSpeakerPcm(int maxSamples, int sampleRateHz) {
        byte[] pcm = nativePullSpeakerPcm(Math.max(0, maxSamples), sampleRateHz);
        if (pcm == null || pcm.length < 2) {
            return new short[0];
        }

        int sampleCount = Math.min(maxSamples, pcm.length / 2);
        short[] samples = new short[Math.max(0, sampleCount)];
        ByteBuffer buffer = ByteBuffer.wrap(pcm).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < samples.length; i++) {
            samples[i] = buffer.getShort();
        }
        return samples;
    }

    @Override
    public AvatarState avatarState() {
        pushResourceSnapshot();
        return decodeAvatarState(nativeAvatarState());
    }

    private void pushResourceSnapshot() {
        NtdDeviceStateSampler.Snapshot snapshot = NtdDeviceStateSampler.sample(context);
        nativeUpdateResources(
                snapshot.availableRamBytes,
                snapshot.batteryPercent,
                snapshot.charging,
                snapshot.thermalState,
                snapshot.latencyBudgetMs);
    }

    private static ChatEvent decodeChatEvent(byte[] encoded) {
        try {
            if (encoded == null || encoded.length < 10) {
                throw new IllegalArgumentException("empty chat event");
            }
            ByteBuffer buffer = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
            int version = Byte.toUnsignedInt(buffer.get());
            if (version != 1) {
                throw new IllegalArgumentException("unsupported chat event protocol");
            }
            int kind = Byte.toUnsignedInt(buffer.get());
            int tokenId = buffer.getInt();
            String text = readString(buffer);
            return new ChatEvent(kind, tokenId, text);
        } catch (RuntimeException error) {
            return new ChatEvent(ChatEvent.ERROR, -1, "Native chat protocol error");
        }
    }

    private static ResumeResult decodeRestoreResult(byte[] encoded) {
        try {
            ByteBuffer buffer = protocolBuffer(encoded);
            boolean restored = readBoolean(buffer);
            boolean eligible = readBoolean(buffer);
            boolean foreground = readBoolean(buffer);
            String status = readString(buffer);
            Approval approval = null;
            if (readBoolean(buffer)) {
                approval = new Approval(readString(buffer), readString(buffer));
            }
            return new ResumeResult(restored, eligible, foreground, status, approval);
        } catch (RuntimeException error) {
            return new ResumeResult(
                    false,
                    false,
                    false,
                    "Native runtime bridge protocol error",
                    null);
        }
    }

    private static AvatarState decodeAvatarState(byte[] encoded) {
        try {
            ByteBuffer buffer = protocolBuffer(encoded);
            int mode = Byte.toUnsignedInt(buffer.get());
            int expression = Byte.toUnsignedInt(buffer.get());
            int gesture = Byte.toUnsignedInt(buffer.get());
            int gazeX = buffer.getShort();
            int gazeY = buffer.getShort();
            int gazeZ = buffer.getShort();
            int lipAmplitude = Short.toUnsignedInt(buffer.getShort());
            int viseme = Byte.toUnsignedInt(buffer.get());
            boolean speaking = readBoolean(buffer);
            int targetFps = Short.toUnsignedInt(buffer.getShort());
            String status = readString(buffer);

            return new AvatarState(
                    mode,
                    expression,
                    gesture,
                    gazeX,
                    gazeY,
                    gazeZ,
                    lipAmplitude,
                    viseme,
                    speaking,
                    targetFps,
                    status);
        } catch (RuntimeException error) {
            return AvatarState.idle();
        }
    }

    private static ByteBuffer protocolBuffer(byte[] encoded) {
        if (encoded == null || encoded.length == 0) {
            throw new IllegalArgumentException("empty bridge response");
        }

        ByteBuffer buffer = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
        int version = Byte.toUnsignedInt(buffer.get());
        if (version != PROTOCOL_VERSION) {
            throw new IllegalArgumentException("unsupported bridge protocol");
        }
        return buffer;
    }

    private static boolean readBoolean(ByteBuffer buffer) {
        int value = Byte.toUnsignedInt(buffer.get());
        if (value != 0 && value != 1) {
            throw new IllegalArgumentException("invalid bridge boolean");
        }
        return value == 1;
    }

    private static String readString(ByteBuffer buffer) {
        int length = buffer.getInt();
        if (length < 0 || length > buffer.remaining()) {
            throw new IllegalArgumentException("invalid bridge string");
        }
        byte[] bytes = new byte[length];
        buffer.get(bytes);
        return new String(bytes, StandardCharsets.UTF_8);
    }

    private static native boolean nativeOpenChatModel(
            String assetId,
            int version,
            String capsulePath,
            String shardRoot,
            String capabilityRoot,
            byte[] verifyKey,
            int contextLimit);

    private static native long nativeSubmitChat(String prompt, int maxNewTokens);

    private static native byte[] nativeNextChatEvent(long requestId);

    private static native boolean nativeCancelChat(long requestId);

    private static native boolean nativeResolveChatApproval(long requestId, boolean approved);

    private static native int nativeChatStatus(long requestId);

    private static native byte[] nativeChatLastError();

    private static native int nativeChatReasoningBudget(long requestId);

    private static native int nativeChatRecalledMemoryItems(long requestId);

    private static native int nativeChatMemoryRecordCount(long requestId);

    private static native int nativeChatReasoningIterations(long requestId);

    private static native int nativeChatActionPlannerStatus(long requestId);

    private static native int nativeChatActionCount(long requestId);

    private static native int nativeChatVerifiedActionCount(long requestId);

    private static native boolean nativeChatVerifiedSynthesisReady(long requestId);

    private static native byte[] nativeChatCheckpoint();

    private static native long nativeRestoreChatCheckpoint(byte[] checkpoint);

    private static native byte[] nativeChatTranscript();

    private static native byte[] nativeProvisionPcPairProfile(
            String capabilityRoot,
            String peer,
            String address,
            String remotePeerId,
            String remoteVerifyKey);

    private static native byte[] nativeDescribePcPairProfile(
            String capabilityRoot,
            String peer);

    private static native byte[] nativeRevokePcPairProfile(
            String capabilityRoot,
            String peer);

    private static native byte[] nativeListPcPairProfiles(String capabilityRoot);

    private static native byte[] nativeProductionCapabilityProbe(
            String capabilityRoot,
            String packageName);

    private static native byte[] nativeRealModelProbe(
            String capsulePath,
            String shardRoot,
            byte[] verifyKey,
            String expectedTokenIds);

    private static native byte[] nativeRestoreAndVerify(byte[] mcs97, String wakeReason);

    private static native byte[] nativeCheckpoint();

    private static native boolean nativeResolveApproval(boolean approved);

    private static native byte[] nativeAvatarState();

    private static native void nativeUpdateResources(
            long availableRamBytes,
            int batteryPercent,
            boolean charging,
            int thermalState,
            int latencyBudgetMs);

    private static native void nativeAcceptMicrophonePcm(byte[] pcmLittleEndian, int sampleRateHz);

    private static native byte[] nativePullSpeakerPcm(int maxSamples, int sampleRateHz);
}
