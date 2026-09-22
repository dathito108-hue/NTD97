package ai.ntd97.mobile;

import android.app.Activity;
import android.os.Build;
import android.os.Bundle;
import android.os.Process;
import android.os.SystemClock;
import android.util.AtomicFile;
import android.widget.TextView;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.SecureRandom;
import java.util.Locale;
import java.util.Properties;

public final class NtdM13HardwareEvidenceActivity extends Activity {
    private static final String PHASE_PREPARE = "prepare";
    private static final String PHASE_RESUME = "resume";
    private static final String EXTRA_PHASE = "phase";
    private static final String EXTRA_APPROVE_EXTERNAL = "approve_external";
    private static final String EXTRA_SEARCH_QUERY = "search_query";
    private static final String EXTRA_BROWSER_URL = "browser_url";
    private static final String EXTRA_STORAGE_PATH = "storage_path";
    private static final String EXTRA_UPLOAD_URL = "upload_url";
    private static final String EXTRA_ACCESSIBILITY_PACKAGE = "accessibility_package";
    private static final String EXTRA_ACCESSIBILITY_VIEW_ID = "accessibility_view_id";
    private static final String EXTRA_PC_ALIAS = "pc_alias";
    private static final String EXTRA_PC_SURFACE = "pc_surface";
    private static final String EXTRA_PC_PATH = "pc_path";

    private static final String STATE_FILE = "ntd97-m13-hardware-state.properties";
    private static final String SUMMARY_FILE = "ntd97-m13-hardware-evidence.txt";
    private static final String EVIDENCE_FILE = "ntd97-m13-hardware-evidence.m13e97";
    private static final String UPLOAD_FIXTURE = "m13-hardware/upload.txt";

    private static final int GATE_WEB_SEARCH = 1 << 0;
    private static final int GATE_BROWSER = 1 << 1;
    private static final int GATE_SAF_READ = 1 << 2;
    private static final int GATE_SAF_WRITE = 1 << 3;
    private static final int GATE_DEVICE_CLIPBOARD = 1 << 4;
    private static final int GATE_APP_ACCESSIBILITY = 1 << 5;
    private static final int GATE_PC_OBSERVE = 1 << 6;
    private static final int GATE_PC_WRITE = 1 << 7;
    private static final int GATE_UPLOAD_RESTORE = 1 << 8;
    private static final int GATE_MIXED_SEQUENCE = 1 << 9;
    private static final int REQUIRED_GATE_MASK = (1 << 10) - 1;
    private static final int MAX_CHAT_EVENTS = 512;

    private TextView statusView;
    private NtdRuntimeHost host;

    static {
        System.loadLibrary("ntd97_android");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        statusView = new TextView(this);
        statusView.setPadding(32, 32, 32, 32);
        statusView.setText("NTD97 M13 physical hardware acceptance starting…");
        setContentView(statusView);

        String phase = getIntent().getStringExtra(EXTRA_PHASE);
        if (phase == null || phase.trim().isEmpty()) {
            phase = PHASE_PREPARE;
        }
        final String selectedPhase = phase.trim().toLowerCase(Locale.ROOT);
        Thread worker = new Thread(
                () -> run(selectedPhase),
                "ntd97-m13-hardware-evidence");
        worker.start();
    }

    @Override
    protected void onDestroy() {
        if (host != null) {
            NtdSessionController.detachRuntime(host);
        }
        super.onDestroy();
    }

    private void run(String phase) {
        try {
            rejectInvalidEnvironment();
            NtdNativeRuntimeHost nativeHost = NtdNativeRuntimeHost.create(this);
            if (nativeHost == null) {
                throw new IllegalStateException("native runtime unavailable");
            }
            host = nativeHost;
            NtdSessionController.attachRuntime(host);
            if (!host.chatReady()) {
                throw new IllegalStateException("verified native chat model unavailable");
            }

            if (PHASE_PREPARE.equals(phase)) {
                runPrepare();
            } else if (PHASE_RESUME.equals(phase)) {
                runResume();
            } else {
                throw new IllegalArgumentException("unknown hardware acceptance phase");
            }
        } catch (Exception error) {
            String message = safeMessage(error);
            writeSummary(
                    "status=failed\n"
                            + "phase=" + phase + "\n"
                            + "error=" + sanitizeLine(message) + "\n");
            runOnUiThread(() -> {
                statusView.setText("M13 hardware acceptance failed:\n" + message);
                finish();
            });
        }
    }

    private void rejectInvalidEnvironment() {
        if (NtdPhysicalEvidenceActivity.isProbablyEmulator()) {
            throw new IllegalStateException("physical hardware acceptance rejects emulators");
        }
        if (!validBuildRevision(BuildConfig.NTD_GIT_SHA)
                || !BuildConfig.NTD_SOVEREIGNTY_AUDIT_PASSED) {
            throw new IllegalStateException("canonical CI-attested APK is required");
        }
        if (!getIntent().getBooleanExtra(EXTRA_APPROVE_EXTERNAL, false)) {
            throw new IllegalStateException(
                    "explicit approve_external=true is required for hardware side effects");
        }
    }

    private void runPrepare() throws Exception {
        deleteIfExists(file(EVIDENCE_FILE));
        deleteIfExists(file(SUMMARY_FILE));
        deleteIfExists(file(STATE_FILE));
        new NtdConversationStore(this).clear();

        String query = requiredExtra(EXTRA_SEARCH_QUERY, 2048);
        String browserUrl = requiredHttpsExtra(EXTRA_BROWSER_URL);
        String storagePath = requiredRelativeExtra(EXTRA_STORAGE_PATH, 512);
        String uploadUrl = requiredHttpsExtra(EXTRA_UPLOAD_URL);
        String accessibilityPackage = requiredExtra(EXTRA_ACCESSIBILITY_PACKAGE, 255);
        String accessibilityViewId = requiredExtra(EXTRA_ACCESSIBILITY_VIEW_ID, 4096);
        String pcAlias = requiredTokenExtra(EXTRA_PC_ALIAS, 64);
        String pcSurface = requiredTokenExtra(EXTRA_PC_SURFACE, 128);
        String pcPath = requiredRelativeExtra(EXTRA_PC_PATH, 512);

        byte[] nonce = new byte[16];
        new SecureRandom().nextBytes(nonce);
        String marker = "NTD97-M13-" + hex(nonce);
        byte[] nonceHash = sha256(nonce);
        String fingerprintHash = sha256Label(Build.FINGERPRINT);
        int gates = 0;
        int verifiedActions = 0;

        CommandResult result = runCommand("search web for " + query, false);
        gates |= GATE_WEB_SEARCH;
        verifiedActions += result.verifiedActions;

        result = runCommand("observe browser " + browserUrl, false);
        verifiedActions += result.verifiedActions;
        result = runCommand("browser navigate " + browserUrl, false);
        gates |= GATE_BROWSER;
        verifiedActions += result.verifiedActions;

        result = runCommand(
                "write granted file shared/" + storagePath + " to " + marker,
                false);
        gates |= GATE_SAF_WRITE;
        verifiedActions += result.verifiedActions;

        result = runCommand(
                "read granted file shared/" + storagePath,
                false);
        gates |= GATE_SAF_READ;
        verifiedActions += result.verifiedActions;

        result = runCommand("set clipboard to " + marker, false);
        gates |= GATE_DEVICE_CLIPBOARD;
        verifiedActions += result.verifiedActions;

        result = runCommand("open app " + accessibilityPackage, false);
        verifiedActions += result.verifiedActions;
        SystemClock.sleep(1200L);
        result = runCommand(
                "accessibility set text "
                        + accessibilityPackage
                        + " "
                        + accessibilityViewId
                        + " to "
                        + marker,
                false);
        gates |= GATE_APP_ACCESSIBILITY;
        verifiedActions += result.verifiedActions;

        result = runCommand(
                "observe pc " + pcAlias + " " + pcSurface,
                false);
        gates |= GATE_PC_OBSERVE;
        verifiedActions += result.verifiedActions;

        result = runCommand(
                "write pc file " + pcAlias + " " + pcPath + " to " + marker,
                false);
        gates |= GATE_PC_WRITE;
        verifiedActions += result.verifiedActions;

        writeUploadFixture(marker);
        CommandResult upload = runCommand(
                "upload artifact " + UPLOAD_FIXTURE + " to " + uploadUrl,
                true);
        if (!upload.checkpointedBeforeSideEffect) {
            throw new IllegalStateException(
                    "upload did not stop at durable pre-network checkpoint");
        }

        PhaseState state = new PhaseState(
                BuildConfig.NTD_GIT_SHA,
                fingerprintHash,
                hex(nonce),
                gates,
                verifiedActions,
                Process.myPid());
        writeState(state);
        writeSummary(
                "status=awaiting-process-death\n"
                        + "phase=prepare\n"
                        + "build_revision=" + BuildConfig.NTD_GIT_SHA + "\n"
                        + "fingerprint_hash=" + fingerprintHash + "\n"
                        + "gate_mask=" + gates + "\n"
                        + "verified_actions=" + verifiedActions + "\n"
                        + "run_nonce_sha256=" + hex(nonceHash) + "\n"
                        + "process_pid=" + Process.myPid() + "\n");

        runOnUiThread(() -> {
            statusView.setText(
                    "Phase A complete. Force-stop NTD97, then relaunch phase=resume.");
            finish();
        });
    }

    private void runResume() throws Exception {
        PhaseState state = readState();
        if (!BuildConfig.NTD_GIT_SHA.equals(state.buildRevision)) {
            throw new IllegalStateException("build revision changed across process death");
        }
        String fingerprintHash = sha256Label(Build.FINGERPRINT);
        if (!fingerprintHash.equals(state.fingerprintHash)) {
            throw new IllegalStateException("physical device changed across process death");
        }
        if (state.processPid == Process.myPid()) {
            throw new IllegalStateException("process death was not observed");
        }

        long requestId = NtdSessionController.restoreConversation(this);
        if (requestId <= 0L) {
            throw new IllegalStateException("upload conversation checkpoint did not restore");
        }
        NtdRuntimeHost.ChatEvent reconfirm = host.nextChatEvent(requestId);
        if (reconfirm.kind != NtdRuntimeHost.ChatEvent.APPROVAL_REQUIRED) {
            throw new IllegalStateException(
                    "restored upload did not require sovereign reconfirm");
        }
        if (!host.resolveChatApproval(requestId, true)) {
            throw new IllegalStateException("upload reconfirm approval was rejected");
        }
        if (!NtdSessionController.checkpointConversation(this)) {
            throw new IllegalStateException("reconfirmed upload checkpoint persistence failed");
        }

        CommandResult upload = finishRestoredRequest(requestId);
        int gates = state.gates | GATE_UPLOAD_RESTORE;
        int verifiedActions = state.verifiedActions + upload.verifiedActions;
        if ((gates & ~GATE_MIXED_SEQUENCE)
                != (REQUIRED_GATE_MASK & ~GATE_MIXED_SEQUENCE)) {
            throw new IllegalStateException("not all mixed-surface gates completed");
        }
        gates |= GATE_MIXED_SEQUENCE;

        byte[] nonce = decodeHex(state.nonceHex);
        byte[] nonceHash = sha256(nonce);
        byte[] encoded = nativeEncodeM13Evidence(
                BuildConfig.NTD_GIT_SHA,
                fingerprintHash,
                gates,
                verifiedActions,
                true,
                BuildConfig.NTD_SOVEREIGNTY_AUDIT_PASSED,
                nonceHash);
        if (encoded == null || encoded.length == 0) {
            throw new IllegalStateException("native M13E97 evidence encoding failed");
        }
        writeFile(EVIDENCE_FILE, encoded);
        writeSummary(
                "status=ok\n"
                        + "phase=resume\n"
                        + "build_revision=" + BuildConfig.NTD_GIT_SHA + "\n"
                        + "fingerprint_hash=" + fingerprintHash + "\n"
                        + "gate_mask=" + gates + "\n"
                        + "required_gate_mask=" + REQUIRED_GATE_MASK + "\n"
                        + "verified_actions=" + verifiedActions + "\n"
                        + "process_death_reconfirmed=true\n"
                        + "sovereignty_audit_passed="
                        + BuildConfig.NTD_SOVEREIGNTY_AUDIT_PASSED
                        + "\n"
                        + "run_nonce_sha256=" + hex(nonceHash) + "\n");
        deleteIfExists(file(STATE_FILE));

        runOnUiThread(() -> {
            statusView.setText("M13 physical hardware acceptance PASS. M13E97 evidence ready.");
            finish();
        });
    }

    private CommandResult runCommand(String prompt, boolean stopAtCheckpoint) {
        long requestId = host.submitChat(prompt, 8);
        if (requestId < 0L) {
            throw new IllegalStateException("chat command rejected: " + prompt);
        }
        boolean approvalSeen = false;
        for (int eventIndex = 0; eventIndex < MAX_CHAT_EVENTS; eventIndex++) {
            NtdRuntimeHost.ChatEvent event = host.nextChatEvent(requestId);
            if (event.kind == NtdRuntimeHost.ChatEvent.TOKEN) {
                continue;
            }
            if (event.kind == NtdRuntimeHost.ChatEvent.APPROVAL_REQUIRED) {
                approvalSeen = true;
                if (!host.resolveChatApproval(requestId, true)) {
                    throw new IllegalStateException("external action approval failed");
                }
                if (!NtdSessionController.checkpointConversation(this)) {
                    throw new IllegalStateException("approved action checkpoint persistence failed");
                }
                continue;
            }
            if (event.kind == NtdRuntimeHost.ChatEvent.ACTION_CHECKPOINTED) {
                if (!NtdSessionController.checkpointConversation(this)) {
                    throw new IllegalStateException("action checkpoint persistence failed");
                }
                if (stopAtCheckpoint && approvalSeen) {
                    if (host.chatVerifiedActionCount(requestId) != 0
                            || host.chatVerifiedSynthesisReady(requestId)) {
                        throw new IllegalStateException(
                                "external action committed before process-death checkpoint");
                    }
                    return new CommandResult(0, true);
                }
                continue;
            }
            if (event.kind == NtdRuntimeHost.ChatEvent.COMPLETE) {
                return verifiedComplete(requestId);
            }
            throw new IllegalStateException(
                    "chat command failed: "
                            + (event.text == null || event.text.isEmpty()
                                    ? "event=" + event.kind
                                    : sanitizeLine(event.text)));
        }
        throw new IllegalStateException("chat command exceeded event bound");
    }

    private CommandResult finishRestoredRequest(long requestId) {
        for (int eventIndex = 0; eventIndex < MAX_CHAT_EVENTS; eventIndex++) {
            NtdRuntimeHost.ChatEvent event = host.nextChatEvent(requestId);
            if (event.kind == NtdRuntimeHost.ChatEvent.TOKEN) {
                continue;
            }
            if (event.kind == NtdRuntimeHost.ChatEvent.ACTION_CHECKPOINTED) {
                if (!NtdSessionController.checkpointConversation(this)) {
                    throw new IllegalStateException("restored action checkpoint persistence failed");
                }
                continue;
            }
            if (event.kind == NtdRuntimeHost.ChatEvent.APPROVAL_REQUIRED) {
                throw new IllegalStateException("restored upload requested duplicate reconfirm");
            }
            if (event.kind == NtdRuntimeHost.ChatEvent.COMPLETE) {
                return verifiedComplete(requestId);
            }
            throw new IllegalStateException(
                    "restored upload failed: "
                            + (event.text == null || event.text.isEmpty()
                                    ? "event=" + event.kind
                                    : sanitizeLine(event.text)));
        }
        throw new IllegalStateException("restored upload exceeded event bound");
    }

    private CommandResult verifiedComplete(long requestId) {
        int actionCount = host.chatActionCount(requestId);
        int verified = host.chatVerifiedActionCount(requestId);
        if (actionCount <= 0
                || verified != actionCount
                || !host.chatVerifiedSynthesisReady(requestId)) {
            throw new IllegalStateException("chat action did not commit verified evidence");
        }
        NtdSessionController.checkpointConversation(this);
        return new CommandResult(verified, false);
    }

    private void writeUploadFixture(String marker) throws Exception {
        File root = new File(getFilesDir(), "ntd97-capability-files");
        File target = new File(root, UPLOAD_FIXTURE);
        File parent = target.getParentFile();
        if (parent == null || (!parent.isDirectory() && !parent.mkdirs())) {
            throw new IllegalStateException("upload fixture directory unavailable");
        }
        try (FileOutputStream stream = new FileOutputStream(target, false)) {
            stream.write(marker.getBytes(StandardCharsets.UTF_8));
            stream.getFD().sync();
        }
    }

    private String requiredExtra(String name, int maxChars) {
        String value = getIntent().getStringExtra(name);
        if (value == null
                || value.trim().isEmpty()
                || !value.equals(value.trim())
                || value.length() > maxChars
                || containsControl(value)) {
            throw new IllegalArgumentException("invalid required extra: " + name);
        }
        return value;
    }

    private String requiredTokenExtra(String name, int maxChars) {
        String value = requiredExtra(name, maxChars);
        if (value.indexOf(' ') >= 0 || value.indexOf('|') >= 0) {
            throw new IllegalArgumentException("invalid token extra: " + name);
        }
        return value;
    }

    private String requiredRelativeExtra(String name, int maxChars) {
        String value = requiredExtra(name, maxChars);
        if (value.startsWith("/")
                || value.contains("../")
                || value.equals("..")
                || value.contains("|")) {
            throw new IllegalArgumentException("invalid relative path extra: " + name);
        }
        return value;
    }

    private String requiredHttpsExtra(String name) {
        String value = requiredExtra(name, 4096);
        if (!value.toLowerCase(Locale.ROOT).startsWith("https://")) {
            throw new IllegalArgumentException("HTTPS extra required: " + name);
        }
        return value;
    }

    private void writeState(PhaseState state) throws Exception {
        Properties properties = new Properties();
        properties.setProperty("build_revision", state.buildRevision);
        properties.setProperty("fingerprint_hash", state.fingerprintHash);
        properties.setProperty("nonce_hex", state.nonceHex);
        properties.setProperty("gates", Integer.toString(state.gates));
        properties.setProperty("verified_actions", Integer.toString(state.verifiedActions));
        properties.setProperty("process_pid", Integer.toString(state.processPid));
        AtomicFile atomic = new AtomicFile(file(STATE_FILE));
        FileOutputStream stream = null;
        try {
            stream = atomic.startWrite();
            properties.store(stream, null);
            stream.getFD().sync();
            atomic.finishWrite(stream);
        } catch (Exception error) {
            if (stream != null) {
                atomic.failWrite(stream);
            }
            throw error;
        }
    }

    private PhaseState readState() throws Exception {
        AtomicFile atomic = new AtomicFile(file(STATE_FILE));
        if (!atomic.getBaseFile().isFile()) {
            throw new IllegalStateException("phase-A hardware state is missing");
        }
        Properties properties = new Properties();
        try (FileInputStream stream = atomic.openRead()) {
            properties.load(stream);
        }
        return new PhaseState(
                requireProperty(properties, "build_revision"),
                requireProperty(properties, "fingerprint_hash"),
                requireProperty(properties, "nonce_hex"),
                Integer.parseInt(requireProperty(properties, "gates")),
                Integer.parseInt(requireProperty(properties, "verified_actions")),
                Integer.parseInt(requireProperty(properties, "process_pid")));
    }

    private static String requireProperty(Properties properties, String key) {
        String value = properties.getProperty(key);
        if (value == null || value.trim().isEmpty()) {
            throw new IllegalStateException("hardware state missing " + key);
        }
        return value.trim();
    }

    private void writeSummary(String summary) {
        try {
            writeFile(SUMMARY_FILE, summary.getBytes(StandardCharsets.UTF_8));
        } catch (Exception ignored) {
            // The UI still reports the primary failure when summary persistence itself fails.
        }
    }

    private void writeFile(String name, byte[] bytes) throws Exception {
        try (FileOutputStream stream = new FileOutputStream(file(name), false)) {
            stream.write(bytes);
            stream.getFD().sync();
        }
    }

    private File file(String name) {
        return new File(getFilesDir(), name);
    }

    private static void deleteIfExists(File file) {
        if (file.exists() && !file.delete()) {
            throw new IllegalStateException("could not clear stale hardware evidence");
        }
    }

    private static boolean validBuildRevision(String value) {
        return value != null
                && value.length() == 40
                && value.chars().allMatch(ch ->
                        (ch >= '0' && ch <= '9')
                                || (ch >= 'a' && ch <= 'f')
                                || (ch >= 'A' && ch <= 'F'));
    }

    private static String sha256Label(String value) throws Exception {
        return "sha256:" + hex(sha256(value.getBytes(StandardCharsets.UTF_8)));
    }

    private static byte[] sha256(byte[] bytes) throws Exception {
        return MessageDigest.getInstance("SHA-256").digest(bytes);
    }

    private static String hex(byte[] bytes) {
        StringBuilder out = new StringBuilder(bytes.length * 2);
        for (byte value : bytes) {
            out.append(String.format(Locale.ROOT, "%02x", value & 0xff));
        }
        return out.toString();
    }

    private static byte[] decodeHex(String value) {
        if (value == null || (value.length() & 1) != 0) {
            throw new IllegalArgumentException("invalid hex state");
        }
        byte[] out = new byte[value.length() / 2];
        for (int index = 0; index < out.length; index++) {
            int high = Character.digit(value.charAt(index * 2), 16);
            int low = Character.digit(value.charAt(index * 2 + 1), 16);
            if (high < 0 || low < 0) {
                throw new IllegalArgumentException("invalid hex state");
            }
            out[index] = (byte) ((high << 4) | low);
        }
        return out;
    }

    private static boolean containsControl(String value) {
        for (int index = 0; index < value.length(); index++) {
            char ch = value.charAt(index);
            if (ch == '\r' || ch == '\n' || ch == '\t' || ch == '\0') {
                return true;
            }
        }
        return false;
    }

    private static String sanitizeLine(String value) {
        return value == null
                ? ""
                : value.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ');
    }

    private static String safeMessage(Throwable error) {
        String message = error.getMessage();
        return message == null || message.trim().isEmpty()
                ? error.getClass().getSimpleName()
                : message;
    }

    private static final class CommandResult {
        final int verifiedActions;
        final boolean checkpointedBeforeSideEffect;

        CommandResult(int verifiedActions, boolean checkpointedBeforeSideEffect) {
            this.verifiedActions = verifiedActions;
            this.checkpointedBeforeSideEffect = checkpointedBeforeSideEffect;
        }
    }

    private static final class PhaseState {
        final String buildRevision;
        final String fingerprintHash;
        final String nonceHex;
        final int gates;
        final int verifiedActions;
        final int processPid;

        PhaseState(
                String buildRevision,
                String fingerprintHash,
                String nonceHex,
                int gates,
                int verifiedActions,
                int processPid) {
            this.buildRevision = buildRevision;
            this.fingerprintHash = fingerprintHash;
            this.nonceHex = nonceHex;
            this.gates = gates;
            this.verifiedActions = verifiedActions;
            this.processPid = processPid;
        }
    }

    private static native byte[] nativeEncodeM13Evidence(
            String buildRevision,
            String fingerprintHash,
            int gateMask,
            int verifiedActionCount,
            boolean processDeathReconfirmed,
            boolean sovereigntyAuditPassed,
            byte[] runNonceSha256);
}
