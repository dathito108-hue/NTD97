package ai.ntd97.mobile;

import android.app.Activity;
import android.content.res.AssetManager;
import android.os.Bundle;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

public final class NtdRealModelProbeActivity extends Activity {
    private static final String ASSET_ROOT = "ntd97-real-model";
    private static final String RUNTIME_ROOT = "ntd97-real-model";
    private static final String RESULT_FILE = "ntd97-real-model-probe.txt";

    private static final int MAX_PRODUCTION_FOCUS_ATTEMPTS = 24;
    private static final long PRODUCTION_FOCUS_RETRY_MS = 250L;
    private static final long PRODUCTION_FOCUS_SETTLE_MS = 750L;

    private String pendingResult;
    private boolean productionProbeStarted;
    private boolean productionCapabilityProbeStarted;
    private int productionFocusAttempts;
    private String externalApprovalDiagnostic = "not-run";
    private String uploadContinuityDiagnostic = "not-run";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        String result;
        try {
            File runtimeRoot = new File(getFilesDir(), RUNTIME_ROOT);
            deleteTree(runtimeRoot);
            copyAssetTree(getAssets(), ASSET_ROOT, runtimeRoot);

            File capsule = new File(runtimeRoot, "model.ncc97");
            File shardRoot = new File(runtimeRoot, "ntp97-shards");
            byte[] verifyKey = readAllBytes(new File(runtimeRoot, "verify-key.bin"));
            String expectedTokenIds = new String(
                    readAllBytes(new File(runtimeRoot, "expected-token-ids.txt")),
                    StandardCharsets.UTF_8).trim();

            byte[] nativeResult = NtdNativeRuntimeHost.runRealModelProbe(
                    capsule.getAbsolutePath(),
                    shardRoot.getAbsolutePath(),
                    verifyKey,
                    expectedTokenIds);
            result = new String(nativeResult, StandardCharsets.UTF_8);
            if (result.isEmpty()) {
                result = "android_real_model=failed\nerror=empty native result\n";
            }
            pendingResult = result;
        } catch (Exception error) {
            String message = error.getMessage();
            if (message == null || message.isEmpty()) {
                message = error.getClass().getSimpleName();
            }
            result = "android_real_model=failed\nerror="
                    + message.replace('\n', ' ').replace('\r', ' ')
                    + "\n";
            writeResult(result);
            finish();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (productionProbeStarted || pendingResult == null) {
            return;
        }

        productionProbeStarted = true;
        getWindow().getDecorView().postDelayed(this::runProductionProbeAndFinish, 300L);
    }

    private void runProductionProbeAndFinish() {
        String result = pendingResult;
        try {
            result = result + runChatApiProbe();
        } catch (Exception error) {
            String message = error.getMessage();
            if (message == null || message.isEmpty()) {
                message = error.getClass().getSimpleName();
            }
            result = result
                    + "chat_submit=failed\n"
                    + "chat_external_approval=failed\n"
                    + "chat_external_approval_detail=foreground-probe:"
                    + message.replace('\n', ' ').replace('\r', ' ')
                    + "\n"
                    + "chat_upload_continuity=failed\n"
                    + "chat_upload_continuity_detail=foreground-probe\n";
        }

        pendingResult = result;
        scheduleProductionCapabilityProbe();
    }

    private void scheduleProductionCapabilityProbe() {
        if (productionCapabilityProbeStarted || pendingResult == null || isFinishing()) {
            return;
        }

        if (hasWindowFocus()) {
            productionCapabilityProbeStarted = true;
            getWindow().getDecorView().postDelayed(
                    this::runProductionCapabilityProbeAndFinish,
                    PRODUCTION_FOCUS_SETTLE_MS);
            return;
        }

        productionFocusAttempts++;
        if (productionFocusAttempts > MAX_PRODUCTION_FOCUS_ATTEMPTS) {
            writeResult(
                    pendingResult
                            + "production_capabilities=failed\n"
                            + "error=foreground focus unavailable after clipboard approval\n");
            finish();
            return;
        }
        getWindow().getDecorView().postDelayed(
                this::scheduleProductionCapabilityProbe,
                PRODUCTION_FOCUS_RETRY_MS);
    }

    private void runProductionCapabilityProbeAndFinish() {
        if (!hasWindowFocus()) {
            productionCapabilityProbeStarted = false;
            scheduleProductionCapabilityProbe();
            return;
        }

        final String prefix = pendingResult;
        Thread probeThread = new Thread(() -> {
            String result = prefix;
            try {
                if (!NtdWebPlatform.configureSearchEndpoint(this, "")) {
                    throw new IOException("failed to clear WebSearch endpoint");
                }
                byte[] unconfiguredSearch =
                        NtdWebPlatform.search("NTD97 unconfigured boundary", 5);
                if (unconfiguredSearch.length < 2
                        || unconfiguredSearch[0] != 1
                        || unconfiguredSearch[1] != 0) {
                    throw new IOException("unconfigured WebSearch did not fail closed");
                }
                result = result + "web_search_unconfigured_block=ok\n";

                boolean searchConfigured = NtdWebPlatform.configureSearchEndpoint(
                        this,
                        "https://example.com/?q={query}&n={count}");
                if (!searchConfigured) {
                    throw new IOException("failed to configure provider-independent search probe");
                }
                result = result + new String(
                        NtdNativeRuntimeHost.runProductionCapabilityProbe(this),
                        StandardCharsets.UTF_8);
            } catch (Exception error) {
                String message = error.getMessage();
                if (message == null || message.isEmpty()) {
                    message = error.getClass().getSimpleName();
                }
                result = result + "production_capabilities=failed\nerror="
                        + message.replace('\n', ' ').replace('\r', ' ')
                        + "\n";
            }

            String finalResult = result;
            runOnUiThread(() -> {
                writeResult(finalResult);
                finish();
            });
        }, "ntd97-production-capabilities");
        probeThread.setDaemon(true);
        probeThread.start();
    }

    private String runChatApiProbe() throws IOException {
        NtdNativeRuntimeHost host = NtdNativeRuntimeHost.create(this);
        if (host == null || !host.chatReady()) {
            return "chat_submit=failed\nchat_stream=failed\nchat_reasoning=failed\nchat_reasoning_loop=failed\nchat_action_planner=failed\nchat_action_safety=failed\nchat_governed_e2e=failed\nchat_external_approval=failed\nchat_upload_continuity=failed\nchat_memory=failed\nchat_restore=failed\nchat_store=failed\nchat_cancel=failed\nchat_status=failed\n";
        }

        long requestId = host.submitChat("Once", 2);
        if (requestId < 0) {
            String diagnostic = host.chatLastError().replace('\n', ' ').replace('\r', ' ');
            return "chat_submit=failed\nchat_error=" + diagnostic
                    + "\nchat_stream=failed\nchat_reasoning=failed\nchat_reasoning_loop=failed\nchat_action_planner=failed\nchat_action_safety=failed\nchat_governed_e2e=failed\nchat_external_approval=failed\nchat_upload_continuity=failed\nchat_memory=failed\nchat_restore=failed\nchat_store=failed\nchat_cancel=failed\nchat_status=failed\n";
        }

        int tokenCount = 0;
        boolean completed = false;
        for (int attempt = 0; attempt < 8; attempt++) {
            NtdRuntimeHost.ChatEvent event = host.nextChatEvent(requestId);
            if (event.kind == NtdRuntimeHost.ChatEvent.TOKEN) {
                tokenCount++;
                continue;
            }
            if (event.kind == NtdRuntimeHost.ChatEvent.COMPLETE) {
                completed = true;
                break;
            }
            return "chat_submit=ok\nchat_stream=failed\nchat_reasoning=failed\nchat_reasoning_loop=failed\nchat_action_planner=failed\nchat_action_safety=failed\nchat_governed_e2e=failed\nchat_external_approval=failed\nchat_upload_continuity=failed\nchat_memory=failed\nchat_restore=failed\nchat_store=failed\nchat_cancel=failed\nchat_status=failed\n";
        }

        boolean statusOk = completed && tokenCount > 0 && host.chatStatus(requestId) == 2;
        int firstBudget = host.chatReasoningBudget(requestId);
        int firstIterations = host.chatReasoningIterations(requestId);
        boolean reasoningOk = firstBudget >= 1 && firstBudget <= 4;
        boolean reasoningLoopOk = firstIterations == expectedReasoningIterations(firstBudget);
        boolean actionPlannerOk = actionPlannerStatusKnown(host, requestId);
        boolean actionSafetyOk = actionPlannerInvariantHolds(host, requestId);

        long checkpointRequest = host.submitChat("Once resume this response", 4);
        boolean restoreOk = false;
        boolean storeOk = false;
        boolean memoryOk = false;
        int checkpointMemoryItems = 0;
        int checkpointBudget = 0;
        int checkpointIterations = 0;
        int checkpointFirstKind = 0;
        int checkpointBytes = 0;
        long restoredRequestId = -1L;
        int restoredBudget = 0;
        int restoredIterations = 0;
        int restoredMemory = 0;
        int restoredStatus = 0;
        NtdConversationStore store = new NtdConversationStore(this);
        try {
            if (checkpointRequest >= 0) {
                checkpointMemoryItems = host.chatRecalledMemoryItems(checkpointRequest);
                checkpointBudget = host.chatReasoningBudget(checkpointRequest);
                checkpointIterations = host.chatReasoningIterations(checkpointRequest);
                memoryOk = checkpointMemoryItems > 0
                        && checkpointBudget >= 1
                        && checkpointBudget <= 4;
                reasoningLoopOk = reasoningLoopOk
                        && checkpointIterations == expectedReasoningIterations(checkpointBudget);
                actionPlannerOk = actionPlannerOk
                        && actionPlannerStatusKnown(host, checkpointRequest);
                actionSafetyOk = actionSafetyOk
                        && actionPlannerInvariantHolds(host, checkpointRequest);
                NtdRuntimeHost.ChatEvent first = host.nextChatEvent(checkpointRequest);
                checkpointFirstKind = first.kind;
                byte[] checkpoint = host.chatCheckpoint();
                checkpointBytes = checkpoint.length;
                if (first.kind == NtdRuntimeHost.ChatEvent.TOKEN && checkpoint.length > 0) {
                    store.write(checkpoint);
                    byte[] persisted = store.read();
                    storeOk = persisted != null && Arrays.equals(checkpoint, persisted);
                    long restoredRequest = storeOk
                            ? host.restoreChatCheckpoint(persisted)
                            : -1L;
                    restoredRequestId = restoredRequest;
                    if (restoredRequest > 0) {
                        boolean restoredComplete = false;
                        for (int attempt = 0; attempt < 8; attempt++) {
                            NtdRuntimeHost.ChatEvent event = host.nextChatEvent(restoredRequest);
                            if (event.kind == NtdRuntimeHost.ChatEvent.TOKEN) {
                                continue;
                            }
                            restoredComplete = event.kind == NtdRuntimeHost.ChatEvent.COMPLETE;
                            break;
                        }
                        restoredBudget = host.chatReasoningBudget(restoredRequest);
                        restoredIterations = host.chatReasoningIterations(restoredRequest);
                        restoredMemory = host.chatRecalledMemoryItems(restoredRequest);
                        restoredStatus = host.chatStatus(restoredRequest);
                        restoreOk = restoredComplete
                                && restoredStatus == 2
                                && restoredBudget >= 1
                                && restoredBudget <= 4
                                && restoredIterations == expectedReasoningIterations(restoredBudget)
                                && restoredMemory > 0
                                && actionPlannerStatusKnown(host, restoredRequest)
                                && actionPlannerInvariantHolds(host, restoredRequest)
                                && host.chatTranscript().contains("Once resume this response");
                    }
                }
            }
        } finally {
            store.clear();
        }

        long governedRequest =
                host.submitChat("What is my current battery status?", 4);
        boolean governedE2eOk = false;
        int governedPlannerStatus = 0;
        int governedActionCount = 0;
        int governedVerifiedCount = 0;
        boolean governedSynthesisReady = false;
        boolean governedComplete = false;
        int governedTokens = 0;
        int governedTerminalKind = 0;
        int governedFinalStatus = 0;
        if (governedRequest >= 0) {
            governedPlannerStatus = host.chatActionPlannerStatus(governedRequest);
            governedActionCount = host.chatActionCount(governedRequest);
            governedVerifiedCount = host.chatVerifiedActionCount(governedRequest);
            governedSynthesisReady = host.chatVerifiedSynthesisReady(governedRequest);
            for (int attempt = 0; attempt < 8; attempt++) {
                NtdRuntimeHost.ChatEvent event = host.nextChatEvent(governedRequest);
                if (event.kind == NtdRuntimeHost.ChatEvent.TOKEN) {
                    governedTokens++;
                    continue;
                }
                governedTerminalKind = event.kind;
                governedComplete = event.kind == NtdRuntimeHost.ChatEvent.COMPLETE;
                break;
            }
            governedFinalStatus = host.chatStatus(governedRequest);
            governedE2eOk = governedPlannerStatus == 2
                    && governedActionCount > 0
                    && governedVerifiedCount == governedActionCount
                    && governedSynthesisReady
                    && governedComplete
                    && governedTokens > 0
                    && governedFinalStatus == 2;
        }

        boolean externalApprovalOk = runExternalApprovalProbe(host);
        boolean uploadContinuityOk = false;
        if (externalApprovalOk) {
            uploadContinuityOk = runUploadContinuityProbe(host);
        } else {
            uploadContinuityDiagnostic = "skipped-after-external-approval-failure";
        }

        long cancelRequest = host.submitChat("cancel this response", 8);
        boolean cancelOk = cancelRequest >= 0
                && host.cancelChat(cancelRequest)
                && host.nextChatEvent(cancelRequest).kind == NtdRuntimeHost.ChatEvent.CANCELLED
                && host.chatStatus(cancelRequest) == 3;

        return "chat_submit=ok\n"
                + "chat_stream=" + (completed && tokenCount > 0 ? "ok" : "failed") + "\n"
                + "chat_reasoning=" + (reasoningOk ? "ok" : "failed") + "\n"
                + "chat_reasoning_loop=" + (reasoningLoopOk ? "ok" : "failed") + "\n"
                + "chat_action_planner=" + (actionPlannerOk ? "ok" : "failed") + "\n"
                + "chat_action_safety=" + (actionSafetyOk ? "ok" : "failed") + "\n"
                + "chat_governed_e2e=" + (governedE2eOk ? "ok" : "failed") + "\n"
                + "chat_external_approval=" + (externalApprovalOk ? "ok" : "failed") + "\n"
                + "chat_external_approval_detail=" + externalApprovalDiagnostic + "\n"
                + "chat_upload_continuity=" + (uploadContinuityOk ? "ok" : "failed") + "\n"
                + "chat_upload_continuity_detail=" + uploadContinuityDiagnostic + "\n"
                + "governed_request_id=" + governedRequest + "\n"
                + "governed_planner_status=" + governedPlannerStatus + "\n"
                + "governed_action_count=" + governedActionCount + "\n"
                + "governed_verified_count=" + governedVerifiedCount + "\n"
                + "governed_synthesis_ready=" + governedSynthesisReady + "\n"
                + "governed_token_count=" + governedTokens + "\n"
                + "governed_terminal_kind=" + governedTerminalKind + "\n"
                + "governed_final_status=" + governedFinalStatus + "\n"
                + "checkpoint_request_id=" + checkpointRequest + "\n"
                + "checkpoint_memory_items=" + checkpointMemoryItems + "\n"
                + "checkpoint_budget=" + checkpointBudget + "\n"
                + "checkpoint_iterations=" + checkpointIterations + "\n"
                + "checkpoint_first_kind=" + checkpointFirstKind + "\n"
                + "checkpoint_bytes=" + checkpointBytes + "\n"
                + "restored_request_id=" + restoredRequestId + "\n"
                + "restored_budget=" + restoredBudget + "\n"
                + "restored_iterations=" + restoredIterations + "\n"
                + "restored_memory_items=" + restoredMemory + "\n"
                + "restored_status=" + restoredStatus + "\n"
                + "chat_memory=" + (memoryOk ? "ok" : "failed") + "\n"
                + "chat_restore=" + (restoreOk ? "ok" : "failed") + "\n"
                + "chat_store=" + (storeOk ? "ok" : "failed") + "\n"
                + "chat_cancel=" + (cancelOk ? "ok" : "failed") + "\n"
                + "chat_status=" + (statusOk ? "ok" : "failed") + "\n";
    }

    private boolean runExternalApprovalProbe(NtdRuntimeHost host) {
        long beforeWrites = NtdDeviceAppPlatform.successfulClipboardWrites();

        long deniedRequest = host.submitChat("set clipboard to NTD97-M13-CHAT", 4);
        if (deniedRequest < 0) {
            externalApprovalDiagnostic = "denied-submit:" + safeDiagnostic(host.chatLastError());
            return false;
        }
        NtdRuntimeHost.ChatEvent pending = host.nextChatEvent(deniedRequest);
        if (pending.kind != NtdRuntimeHost.ChatEvent.APPROVAL_REQUIRED) {
            externalApprovalDiagnostic = "denied-event-kind:" + pending.kind;
            return false;
        }
        if (NtdDeviceAppPlatform.successfulClipboardWrites() != beforeWrites) {
            externalApprovalDiagnostic = "denied-side-effect-before-approval";
            return false;
        }

        byte[] checkpoint = host.chatCheckpoint();
        if (checkpoint.length == 0) {
            externalApprovalDiagnostic = "denied-checkpoint-empty";
            return false;
        }
        long restored = host.restoreChatCheckpoint(checkpoint);
        if (restored <= 0) {
            externalApprovalDiagnostic = "denied-restore:" + safeDiagnostic(host.chatLastError());
            return false;
        }
        NtdRuntimeHost.ChatEvent restoredPending = host.nextChatEvent(restored);
        if (restoredPending.kind != NtdRuntimeHost.ChatEvent.APPROVAL_REQUIRED) {
            externalApprovalDiagnostic = "denied-restored-event-kind:" + restoredPending.kind;
            return false;
        }
        if (!host.resolveChatApproval(restored, false)) {
            externalApprovalDiagnostic = "denied-resolve:" + safeDiagnostic(host.chatLastError());
            return false;
        }
        if (host.chatStatus(restored) != 3) {
            externalApprovalDiagnostic = "denied-status:" + host.chatStatus(restored);
            return false;
        }
        if (NtdDeviceAppPlatform.successfulClipboardWrites() != beforeWrites) {
            externalApprovalDiagnostic = "denied-side-effect-after-denial";
            return false;
        }

        long approvedRequest = host.submitChat("set clipboard to NTD97-M13-CHAT", 4);
        if (approvedRequest < 0) {
            externalApprovalDiagnostic = "approved-submit:" + safeDiagnostic(host.chatLastError());
            return false;
        }
        NtdRuntimeHost.ChatEvent approvedPending = host.nextChatEvent(approvedRequest);
        if (approvedPending.kind != NtdRuntimeHost.ChatEvent.APPROVAL_REQUIRED) {
            externalApprovalDiagnostic = "approved-event-kind:" + approvedPending.kind;
            return false;
        }
        if (!host.resolveChatApproval(approvedRequest, true)) {
            externalApprovalDiagnostic = "approved-prepare:" + safeDiagnostic(host.chatLastError());
            return false;
        }
        if (NtdDeviceAppPlatform.successfulClipboardWrites() != beforeWrites) {
            externalApprovalDiagnostic = "side-effect-before-durable-checkpoint";
            return false;
        }

        byte[] approvedCheckpoint = host.chatCheckpoint();
        if (approvedCheckpoint.length == 0) {
            externalApprovalDiagnostic = "approved-checkpoint-empty";
            return false;
        }
        long interruptedRequest = host.restoreChatCheckpoint(approvedCheckpoint);
        if (interruptedRequest <= 0) {
            externalApprovalDiagnostic = "approved-restore:" + safeDiagnostic(host.chatLastError());
            return false;
        }
        NtdRuntimeHost.ChatEvent reconfirm = host.nextChatEvent(interruptedRequest);
        if (reconfirm.kind != NtdRuntimeHost.ChatEvent.APPROVAL_REQUIRED) {
            externalApprovalDiagnostic = "approved-restore-replayed:" + reconfirm.kind;
            return false;
        }
        if (NtdDeviceAppPlatform.successfulClipboardWrites() != beforeWrites) {
            externalApprovalDiagnostic = "side-effect-during-restore";
            return false;
        }
        if (!host.resolveChatApproval(interruptedRequest, true)) {
            externalApprovalDiagnostic = "reconfirm-prepare:" + safeDiagnostic(host.chatLastError());
            return false;
        }
        if (NtdDeviceAppPlatform.successfulClipboardWrites() != beforeWrites) {
            externalApprovalDiagnostic = "side-effect-before-resume-event";
            return false;
        }

        NtdRuntimeHost.ChatEvent actionEvent = host.nextChatEvent(interruptedRequest);
        if (actionEvent.kind != NtdRuntimeHost.ChatEvent.ACTION_CHECKPOINTED) {
            externalApprovalDiagnostic = "resume-event-kind:" + actionEvent.kind;
            return false;
        }
        if (NtdDeviceAppPlatform.successfulClipboardWrites() != beforeWrites + 1) {
            externalApprovalDiagnostic = "approved-write-count:"
                    + NtdDeviceAppPlatform.successfulClipboardWrites();
            return false;
        }
        if (!NtdDeviceAppPlatform.lastSuccessfulClipboardTextEquals("NTD97-M13-CHAT")) {
            externalApprovalDiagnostic = "approved-clipboard-readback";
            return false;
        }
        if (!host.chatVerifiedSynthesisReady(interruptedRequest)) {
            externalApprovalDiagnostic = "approved-synthesis-not-ready";
            return false;
        }
        if (host.chatVerifiedActionCount(interruptedRequest) <= 0) {
            externalApprovalDiagnostic = "approved-verified-count:"
                    + host.chatVerifiedActionCount(interruptedRequest);
            return false;
        }

        for (int attempt = 0; attempt < 8; attempt++) {
            NtdRuntimeHost.ChatEvent event = host.nextChatEvent(interruptedRequest);
            if (event.kind == NtdRuntimeHost.ChatEvent.TOKEN
                    || event.kind == NtdRuntimeHost.ChatEvent.ACTION_CHECKPOINTED) {
                continue;
            }
            boolean ok = event.kind == NtdRuntimeHost.ChatEvent.COMPLETE
                    && host.chatStatus(interruptedRequest) == 2;
            externalApprovalDiagnostic = ok
                    ? "ok"
                    : "approved-terminal-kind:" + event.kind
                            + ":status:" + host.chatStatus(interruptedRequest);
            return ok;
        }
        externalApprovalDiagnostic = "approved-terminal-timeout";
        return false;
    }

    private boolean runUploadContinuityProbe(NtdRuntimeHost host) {
        File source = new File(
                new File(getFilesDir(), "ntd97-capability-files"),
                "m13/process-upload.bin");
        try {
            File parent = source.getParentFile();
            if (parent == null || (!parent.isDirectory() && !parent.mkdirs())) {
                uploadContinuityDiagnostic = "source-directory";
                return false;
            }
            byte[] payload = "NTD97-M13-PERSISTED-UPLOAD".getBytes(StandardCharsets.UTF_8);
            try (FileOutputStream output = new FileOutputStream(source, false)) {
                output.write(payload);
                output.getFD().sync();
            }

            long request = host.submitChat(
                    "upload artifact m13/process-upload.bin to https://127.0.0.1/upload",
                    4);
            if (request < 0) {
                uploadContinuityDiagnostic = "submit:" + safeDiagnostic(host.chatLastError());
                return false;
            }
            NtdRuntimeHost.ChatEvent approval = host.nextChatEvent(request);
            if (approval.kind != NtdRuntimeHost.ChatEvent.APPROVAL_REQUIRED) {
                uploadContinuityDiagnostic = "approval-kind:" + approval.kind;
                return false;
            }
            if (!host.resolveChatApproval(request, true)) {
                uploadContinuityDiagnostic = "prepare:" + safeDiagnostic(host.chatLastError());
                return false;
            }

            NtdRuntimeHost.ChatEvent suspended = host.nextChatEvent(request);
            if (suspended.kind != NtdRuntimeHost.ChatEvent.ACTION_CHECKPOINTED) {
                uploadContinuityDiagnostic = "suspend-kind:" + suspended.kind;
                return false;
            }
            if (host.chatVerifiedSynthesisReady(request)
                    || host.chatVerifiedActionCount(request) != 0) {
                uploadContinuityDiagnostic = "side-effect-committed-before-restore";
                return false;
            }

            byte[] checkpoint = host.chatCheckpoint();
            if (checkpoint.length == 0) {
                uploadContinuityDiagnostic = "checkpoint-empty";
                return false;
            }
            long restored = host.restoreChatCheckpoint(checkpoint);
            if (restored <= 0) {
                uploadContinuityDiagnostic = "restore:" + safeDiagnostic(host.chatLastError());
                return false;
            }
            NtdRuntimeHost.ChatEvent reconfirm = host.nextChatEvent(restored);
            if (reconfirm.kind != NtdRuntimeHost.ChatEvent.APPROVAL_REQUIRED) {
                uploadContinuityDiagnostic = "restore-auto-replayed:" + reconfirm.kind;
                return false;
            }
            if (!host.resolveChatApproval(restored, false)) {
                uploadContinuityDiagnostic = "deny:" + safeDiagnostic(host.chatLastError());
                return false;
            }
            if (host.chatStatus(restored) != 3) {
                uploadContinuityDiagnostic = "deny-status:" + host.chatStatus(restored);
                return false;
            }
            if (!Arrays.equals(payload, readAllBytes(source))) {
                uploadContinuityDiagnostic = "source-changed";
                return false;
            }

            uploadContinuityDiagnostic = "ok";
            return true;
        } catch (IOException error) {
            uploadContinuityDiagnostic = "io:" + safeDiagnostic(error.getMessage());
            return false;
        } finally {
            if (source.exists() && !source.delete()) {
                source.deleteOnExit();
            }
        }
    }

    private String safeDiagnostic(String value) {
        if (value == null || value.trim().isEmpty()) {
            return "none";
        }
        return value.replace('\n', ' ').replace('\r', ' ');
    }

    private boolean actionPlannerStatusKnown(NtdRuntimeHost host, long requestId) {
        int status = host.chatActionPlannerStatus(requestId);
        return status >= 1 && status <= 3;
    }

    private boolean actionPlannerInvariantHolds(NtdRuntimeHost host, long requestId) {
        int status = host.chatActionPlannerStatus(requestId);
        int actionCount = host.chatActionCount(requestId);
        int verifiedCount = host.chatVerifiedActionCount(requestId);
        if (status == 2) {
            return actionCount > 0 && verifiedCount == actionCount;
        }
        return (status == 1 || status == 3)
                && actionCount == 0
                && verifiedCount == 0;
    }

    private int expectedReasoningIterations(int budget) {
        switch (budget) {
            case 1:
                return 1;
            case 2:
                return 2;
            case 3:
                return 4;
            case 4:
                return 3;
            default:
                return 0;
        }
    }

    private void copyAssetTree(
            AssetManager assets,
            String assetPath,
            File destination) throws IOException {
        String[] children = assets.list(assetPath);
        if (children == null) {
            throw new IOException("asset listing failed: " + assetPath);
        }

        if (children.length == 0) {
            File parent = destination.getParentFile();
            if (parent != null && !parent.isDirectory() && !parent.mkdirs()) {
                throw new IOException("failed to create asset parent");
            }
            try (InputStream input = assets.open(assetPath);
                 FileOutputStream output = new FileOutputStream(destination, false)) {
                copy(input, output);
                output.getFD().sync();
            }
            return;
        }

        if (!destination.isDirectory() && !destination.mkdirs()) {
            throw new IOException("failed to create asset directory");
        }
        for (String child : children) {
            copyAssetTree(
                    assets,
                    assetPath + "/" + child,
                    new File(destination, child));
        }
    }

    private byte[] readAllBytes(File file) throws IOException {
        try (FileInputStream input = new FileInputStream(file);
             ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            copy(input, output);
            return output.toByteArray();
        }
    }

    private void copy(InputStream input, java.io.OutputStream output) throws IOException {
        byte[] buffer = new byte[16 * 1024];
        int count;
        while ((count = input.read(buffer)) != -1) {
            output.write(buffer, 0, count);
        }
    }

    private void deleteTree(File file) throws IOException {
        if (!file.exists()) {
            return;
        }
        if (file.isDirectory()) {
            File[] children = file.listFiles();
            if (children == null) {
                throw new IOException("failed to list runtime directory");
            }
            for (File child : children) {
                deleteTree(child);
            }
        }
        if (!file.delete()) {
            throw new IOException("failed to delete " + file.getAbsolutePath());
        }
    }

    private void writeResult(String result) {
        File output = new File(getFilesDir(), RESULT_FILE);
        try (FileOutputStream stream = new FileOutputStream(output, false)) {
            stream.write(result.getBytes(StandardCharsets.UTF_8));
            stream.getFD().sync();
        } catch (IOException error) {
            throw new IllegalStateException("failed to write real-model probe result", error);
        }
    }
}
