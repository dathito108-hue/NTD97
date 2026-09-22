package ai.ntd97.mobile;

import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.Intent;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.regex.Pattern;

final class NtdDeviceAppPlatform {
    private static final byte PROTOCOL_VERSION = 1;
    private static final int MAX_TEXT_BYTES = 64 * 1024;
    private static final Pattern PACKAGE_NAME =
            Pattern.compile("[A-Za-z][A-Za-z0-9_]*(\\.[A-Za-z0-9_]+)+");

    private static volatile Context appContext;
    private static volatile long successfulClipboardWrites;
    private static volatile String lastSuccessfulClipboardText;

    private NtdDeviceAppPlatform() {}

    static void initialize(Context context) {
        if (context != null) {
            appContext = context.getApplicationContext();
        }
    }

    static byte[] clipboardSet(String text) {
        try {
            Context context = requireContext();
            if (text == null || text.isEmpty()) {
                throw new IllegalArgumentException("clipboard text is empty");
            }
            byte[] encoded = text.getBytes(StandardCharsets.UTF_8);
            if (encoded.length > MAX_TEXT_BYTES) {
                throw new IllegalArgumentException("clipboard text exceeds size limit");
            }

            ClipboardManager clipboard =
                    (ClipboardManager) context.getSystemService(Context.CLIPBOARD_SERVICE);
            if (clipboard == null) {
                throw new IllegalStateException("clipboard service unavailable");
            }

            clipboard.setPrimaryClip(ClipData.newPlainText("NTD97", text));
            String digest = sha256Hex(encoded);
            lastSuccessfulClipboardText = text;
            successfulClipboardWrites++;
            return encodeSuccess("clipboard-set:" + encoded.length + ":" + digest);
        } catch (Exception error) {
            return encodeError(safeMessage(error));
        }
    }

    static long successfulClipboardWrites() {
        return successfulClipboardWrites;
    }

    static boolean lastSuccessfulClipboardTextEquals(String expected) {
        return expected != null && expected.equals(lastSuccessfulClipboardText);
    }

    static byte[] launchApp(String packageName) {
        try {
            Context context = requireContext();
            if (packageName == null || !PACKAGE_NAME.matcher(packageName).matches()) {
                throw new IllegalArgumentException("invalid package name");
            }

            Intent launch = context.getPackageManager().getLaunchIntentForPackage(packageName);
            if (launch == null) {
                throw new IllegalArgumentException("package has no visible launch activity");
            }
            launch.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            launch.setPackage(packageName);
            context.startActivity(launch);
            return encodeSuccess("app-launch:" + packageName);
        } catch (Exception error) {
            return encodeError(safeMessage(error));
        }
    }

    static byte[] accessibilityInteract(
            String packageName,
            String operation,
            String payload) {
        try {
            if (packageName == null || !PACKAGE_NAME.matcher(packageName).matches()) {
                throw new IllegalArgumentException("invalid package name");
            }
            if (!"accessibility.click".equals(operation)
                    && !"accessibility.set_text".equals(operation)) {
                throw new IllegalArgumentException("unsupported accessibility operation");
            }
            if (payload == null || payload.isEmpty()) {
                throw new IllegalArgumentException("accessibility payload is empty");
            }
            NtdAccessibilityService.perform(packageName, operation, payload);
            String digest = sha256Hex(payload.getBytes(StandardCharsets.UTF_8));
            return encodeSuccess(
                    "accessibility:" + operation + ":" + packageName + ":" + digest);
        } catch (Exception error) {
            return encodeError(safeMessage(error));
        }
    }

    private static Context requireContext() {
        Context context = appContext;
        if (context == null) {
            throw new IllegalStateException("Android app context is not initialized");
        }
        return context;
    }

    private static byte[] encodeSuccess(String receipt) {
        byte[] bytes = receipt.getBytes(StandardCharsets.UTF_8);
        ByteBuffer buffer = ByteBuffer
                .allocate(1 + 1 + 4 + bytes.length)
                .order(ByteOrder.LITTLE_ENDIAN);
        buffer.put(PROTOCOL_VERSION);
        buffer.put((byte) 1);
        putBytes(buffer, bytes);
        return buffer.array();
    }

    private static byte[] encodeError(String message) {
        byte[] bytes = message.getBytes(StandardCharsets.UTF_8);
        ByteBuffer buffer = ByteBuffer
                .allocate(1 + 1 + 4 + bytes.length)
                .order(ByteOrder.LITTLE_ENDIAN);
        buffer.put(PROTOCOL_VERSION);
        buffer.put((byte) 0);
        putBytes(buffer, bytes);
        return buffer.array();
    }

    private static String safeMessage(Throwable error) {
        String message = error.getMessage();
        return message == null || message.trim().isEmpty()
                ? error.getClass().getSimpleName()
                : message;
    }

    private static String sha256Hex(byte[] bytes) throws Exception {
        byte[] digest = MessageDigest.getInstance("SHA-256").digest(bytes);
        StringBuilder hex = new StringBuilder(digest.length * 2);
        for (byte value : digest) {
            hex.append(String.format("%02x", value & 0xff));
        }
        return hex.toString();
    }

    private static void putBytes(ByteBuffer buffer, byte[] bytes) {
        buffer.putInt(bytes.length);
        buffer.put(bytes);
    }
}
