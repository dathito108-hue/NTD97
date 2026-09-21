package ai.ntd97.mobile;

import android.content.Context;

import java.io.IOException;
import java.util.concurrent.atomic.AtomicReference;

public final class NtdSessionController {
    private static final AtomicReference<NtdRuntimeHost> HOST = new AtomicReference<>();

    private NtdSessionController() {}

    public static void attachRuntime(NtdRuntimeHost host) {
        if (host == null) {
            throw new IllegalArgumentException("host == null");
        }
        HOST.set(host);
    }

    public static void detachRuntime(NtdRuntimeHost host) {
        HOST.compareAndSet(host, null);
    }

    public static NtdRuntimeHost runtime() {
        return HOST.get();
    }

    public static NtdRuntimeHost.ResumeResult restore(Context context, String wakeReason) {
        try {
            byte[] checkpoint = new NtdContinuityStore(context).read();
            if (checkpoint == null || checkpoint.length == 0) {
                return new NtdRuntimeHost.ResumeResult(
                        false,
                        false,
                        false,
                        "No durable task to restore",
                        null);
            }

            NtdRuntimeHost host = HOST.get();
            if (host == null) {
                return new NtdRuntimeHost.ResumeResult(
                        false,
                        true,
                        false,
                        "Runtime unavailable; durable state preserved",
                        null);
            }

            NtdRuntimeHost.ResumeResult result =
                    host.restoreAndVerify(checkpoint, wakeReason == null ? "unknown" : wakeReason);

            if (result.approval != null) {
                NtdNotificationController.showApproval(
                        context,
                        result.approval.capability,
                        result.approval.rationale);
            }
            return result;
        } catch (IOException error) {
            return new NtdRuntimeHost.ResumeResult(
                    false,
                    false,
                    false,
                    "Continuity read failed",
                    null);
        }
    }

    public static long restoreConversation(Context context) {
        NtdRuntimeHost host = HOST.get();
        if (host == null) {
            return -1L;
        }

        try {
            byte[] checkpoint = new NtdConversationStore(context).read();
            if (checkpoint == null || checkpoint.length == 0) {
                return 0L;
            }
            return host.restoreChatCheckpoint(checkpoint);
        } catch (IOException error) {
            return -1L;
        }
    }

    public static boolean checkpointConversation(Context context) {
        NtdRuntimeHost host = HOST.get();
        if (host == null) {
            return false;
        }

        byte[] checkpoint = host.chatCheckpoint();
        if (checkpoint == null || checkpoint.length == 0) {
            return false;
        }

        try {
            new NtdConversationStore(context).write(checkpoint);
            return true;
        } catch (IOException error) {
            return false;
        }
    }

    public static boolean checkpoint(Context context) {
        NtdRuntimeHost host = HOST.get();
        if (host == null) {
            return false;
        }

        byte[] checkpoint = host.checkpoint();
        if (checkpoint == null || checkpoint.length == 0) {
            return false;
        }

        try {
            new NtdContinuityStore(context).write(checkpoint);
            return true;
        } catch (IOException error) {
            return false;
        }
    }
}
