package ai.ntd97.mobile;

import android.content.Context;
import android.util.AtomicFile;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URI;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;

final class NtdAndroidCapabilityHost {
    private static final byte[] RECEIPT_MAGIC =
            new byte[]{'A', 'P', 'R', '9', '7', 0};
    private static final byte[] IDEMPOTENCY_MAGIC =
            new byte[]{'A', 'I', 'D', '9', '7', 0};
    private static final int RECEIPT_VERSION = 1;
    private static final int RECEIPT_COMPLETED = 1;
    private static final int RECEIPT_RETRYABLE = 2;
    private static final int VALUE_NONE = 0;
    private static final int VALUE_BYTES = 2;
    private static final int MAX_BODY_BYTES = 1024 * 1024;
    private static final int CONNECT_TIMEOUT_MS = 8000;
    private static final int READ_TIMEOUT_MS = 12000;

    private static Context appContext;

    private NtdAndroidCapabilityHost() {}

    static synchronized void initialize(Context context) {
        appContext = context.getApplicationContext();
    }

    static synchronized byte[] webFetch(long actionId, String urlText) {
        try {
            URI uri = URI.create(urlText);
            String scheme = uri.getScheme();
            if (!"https".equalsIgnoreCase(scheme)
                    && !"http".equalsIgnoreCase(scheme)) {
                return retryable(actionId, "unsupported URL scheme");
            }

            HttpURLConnection connection =
                    (HttpURLConnection) new URL(urlText).openConnection();
            connection.setInstanceFollowRedirects(true);
            connection.setRequestMethod("GET");
            connection.setConnectTimeout(CONNECT_TIMEOUT_MS);
            connection.setReadTimeout(READ_TIMEOUT_MS);
            connection.setRequestProperty("Accept", "*/*");
            connection.setRequestProperty("User-Agent", "NTD97/1");

            int status = connection.getResponseCode();
            if (status < 200 || status >= 300) {
                connection.disconnect();
                return retryable(actionId, "HTTP status " + status);
            }

            byte[] body;
            try (InputStream input = connection.getInputStream()) {
                body = readBounded(input, MAX_BODY_BYTES);
            } finally {
                connection.disconnect();
            }

            String finalUrl = connection.getURL().toString();
            String contentType = connection.getContentType();
            return completed(
                    actionId,
                    "verified Android HTTP fetch",
                    VALUE_BYTES,
                    body,
                    new String[]{
                            "android-http",
                            "http-status=" + status,
                            "url=" + finalUrl,
                            "content-type=" + (contentType == null ? "" : contentType),
                            "sha256=" + sha256Hex(body)
                    },
                    new byte[0]);
        } catch (Exception error) {
            return retryable(actionId, compactError(error));
        }
    }

    static synchronized byte[] fileRead(long actionId, String relativePath) {
        try {
            File target = resolveScopedPath(relativePath);
            if (!target.isFile()) {
                return retryable(actionId, "scoped file does not exist");
            }
            byte[] bytes;
            try (InputStream input = new FileInputStream(target)) {
                bytes = readBounded(input, MAX_BODY_BYTES);
            }
            return completed(
                    actionId,
                    "verified app-scoped file read",
                    VALUE_BYTES,
                    bytes,
                    new String[]{
                            "android-app-file-read",
                            "path=" + relativePath,
                            "sha256=" + sha256Hex(bytes)
                    },
                    new byte[0]);
        } catch (Exception error) {
            return retryable(actionId, compactError(error));
        }
    }

    static synchronized byte[] fileWrite(long actionId, String relativePath, byte[] bytes) {
        try {
            if (bytes == null || bytes.length > MAX_BODY_BYTES) {
                return retryable(actionId, "file payload exceeds limit");
            }
            File target = resolveScopedPath(relativePath);
            byte[] targetHash = sha256(bytes);

            File receiptFile = idempotencyFile(actionId);
            if (receiptFile.isFile()) {
                IdempotencyRecord record = readIdempotencyRecord(receiptFile);
                if (!record.path.equals(relativePath)
                        || !Arrays.equals(record.targetHash, targetHash)) {
                    return retryable(actionId, "action id reused for a different write");
                }
                return record.receipt;
            }

            byte[] previous = target.isFile()
                    ? readBounded(new FileInputStream(target), MAX_BODY_BYTES)
                    : new byte[0];
            boolean existed = target.isFile();
            byte[] rollbackToken = encodeRollbackToken(existed, previous);

            File parent = target.getParentFile();
            if (parent == null || (!parent.isDirectory() && !parent.mkdirs())) {
                return retryable(actionId, "failed to create scoped file directory");
            }
            writeAtomic(target, bytes);

            byte[] receipt = completed(
                    actionId,
                    "verified app-scoped file write",
                    VALUE_NONE,
                    new byte[0],
                    new String[]{
                            "android-app-file-write",
                            "path=" + relativePath,
                            "sha256=" + sha256Hex(bytes)
                    },
                    rollbackToken);
            writeIdempotencyRecord(receiptFile, relativePath, targetHash, receipt);
            return receipt;
        } catch (Exception error) {
            return retryable(actionId, compactError(error));
        }
    }

    static synchronized boolean rollbackFileWrite(
            long actionId,
            String relativePath,
            byte[] rollbackToken) {
        try {
            File target = resolveScopedPath(relativePath);
            RollbackState rollback = decodeRollbackToken(rollbackToken);
            if (rollback.existed) {
                File parent = target.getParentFile();
                if (parent == null || (!parent.isDirectory() && !parent.mkdirs())) {
                    return false;
                }
                writeAtomic(target, rollback.previous);
            } else if (target.exists() && !target.delete()) {
                return false;
            }

            File receipt = idempotencyFile(actionId);
            return !receipt.exists() || receipt.delete();
        } catch (Exception error) {
            return false;
        }
    }

    private static Context requireContext() throws IOException {
        if (appContext == null) {
            throw new IOException("Android capability host is not initialized");
        }
        return appContext;
    }

    private static File capabilityRoot() throws IOException {
        File root = new File(requireContext().getFilesDir(), "ntd97-capabilities");
        if (!root.isDirectory() && !root.mkdirs()) {
            throw new IOException("failed to create capability root");
        }
        return root.getCanonicalFile();
    }

    private static File resolveScopedPath(String relativePath) throws IOException {
        if (relativePath == null
                || relativePath.isEmpty()
                || new File(relativePath).isAbsolute()) {
            throw new IOException("invalid scoped path");
        }
        File root = capabilityRoot();
        File target = new File(root, relativePath).getCanonicalFile();
        String rootPath = root.getPath();
        String targetPath = target.getPath();
        if (!targetPath.startsWith(rootPath + File.separator)) {
            throw new IOException("scoped path escapes capability root");
        }
        return target;
    }

    private static File idempotencyFile(long actionId) throws IOException {
        File receipts = new File(capabilityRoot(), ".receipts");
        if (!receipts.isDirectory() && !receipts.mkdirs()) {
            throw new IOException("failed to create receipt directory");
        }
        return new File(receipts, Long.toUnsignedString(actionId) + ".aid97");
    }

    private static void writeAtomic(File target, byte[] bytes) throws IOException {
        AtomicFile atomic = new AtomicFile(target);
        FileOutputStream output = null;
        try {
            output = atomic.startWrite();
            output.write(bytes);
            output.getFD().sync();
            atomic.finishWrite(output);
        } catch (IOException error) {
            if (output != null) {
                atomic.failWrite(output);
            }
            throw error;
        }
    }

    private static void writeIdempotencyRecord(
            File file,
            String path,
            byte[] targetHash,
            byte[] receipt) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        out.write(IDEMPOTENCY_MAGIC);
        writeString(out, path);
        writeBytes(out, targetHash);
        writeBytes(out, receipt);
        writeAtomic(file, out.toByteArray());
    }

    private static IdempotencyRecord readIdempotencyRecord(File file) throws IOException {
        byte[] bytes;
        try (InputStream input = new FileInputStream(file)) {
            bytes = readBounded(input, MAX_BODY_BYTES * 2);
        }
        Cursor cursor = new Cursor(bytes);
        if (!Arrays.equals(cursor.take(IDEMPOTENCY_MAGIC.length), IDEMPOTENCY_MAGIC)) {
            throw new IOException("invalid idempotency receipt");
        }
        String path = cursor.string();
        byte[] targetHash = cursor.bytes();
        if (targetHash.length != 32) {
            throw new IOException("invalid idempotency digest");
        }
        byte[] receipt = cursor.bytes();
        if (!cursor.finished()) {
            throw new IOException("non-canonical idempotency receipt");
        }
        return new IdempotencyRecord(path, targetHash, receipt);
    }

    private static byte[] encodeRollbackToken(boolean existed, byte[] previous)
            throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        out.write(1);
        out.write(existed ? 1 : 0);
        writeBytes(out, previous);
        return out.toByteArray();
    }

    private static RollbackState decodeRollbackToken(byte[] bytes) throws IOException {
        Cursor cursor = new Cursor(bytes);
        if (cursor.u8() != 1) {
            throw new IOException("unsupported rollback token");
        }
        int existed = cursor.u8();
        if (existed != 0 && existed != 1) {
            throw new IOException("invalid rollback state");
        }
        byte[] previous = cursor.bytes();
        if (!cursor.finished()) {
            throw new IOException("non-canonical rollback token");
        }
        return new RollbackState(existed == 1, previous);
    }

    private static byte[] completed(
            long actionId,
            String summary,
            int valueKind,
            byte[] value,
            String[] evidence,
            byte[] rollbackToken) throws IOException {
        return encodeReceipt(
                RECEIPT_COMPLETED,
                actionId,
                summary,
                valueKind,
                value,
                evidence,
                rollbackToken);
    }

    private static byte[] retryable(long actionId, String reason) {
        try {
            return encodeReceipt(
                    RECEIPT_RETRYABLE,
                    actionId,
                    reason,
                    VALUE_NONE,
                    new byte[0],
                    new String[0],
                    new byte[0]);
        } catch (IOException impossible) {
            return new byte[0];
        }
    }

    private static byte[] encodeReceipt(
            int status,
            long actionId,
            String summary,
            int valueKind,
            byte[] value,
            String[] evidence,
            byte[] rollbackToken) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        out.write(RECEIPT_MAGIC);
        out.write(RECEIPT_VERSION);
        out.write(status);
        writeLong(out, actionId);
        writeString(out, summary);
        out.write(valueKind);
        writeBytes(out, value);
        writeShort(out, evidence.length);
        for (String item : evidence) {
            writeString(out, item);
        }
        writeBytes(out, rollbackToken);
        return out.toByteArray();
    }

    private static byte[] readBounded(InputStream input, int limit) throws IOException {
        try (InputStream closeable = input;
             ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[16 * 1024];
            int total = 0;
            int count;
            while ((count = closeable.read(buffer)) != -1) {
                total += count;
                if (total > limit) {
                    throw new IOException("payload exceeds limit");
                }
                out.write(buffer, 0, count);
            }
            return out.toByteArray();
        }
    }

    private static byte[] sha256(byte[] bytes) throws IOException {
        try {
            return MessageDigest.getInstance("SHA-256").digest(bytes);
        } catch (NoSuchAlgorithmException error) {
            throw new IOException("SHA-256 unavailable", error);
        }
    }

    private static String sha256Hex(byte[] bytes) throws IOException {
        byte[] digest = sha256(bytes);
        StringBuilder out = new StringBuilder(64);
        for (byte value : digest) {
            out.append(String.format("%02x", value & 0xff));
        }
        return out.toString();
    }

    private static String compactError(Exception error) {
        String message = error.getMessage();
        if (message == null || message.isEmpty()) {
            return error.getClass().getSimpleName();
        }
        return message.replace('\n', ' ').replace('\r', ' ');
    }

    private static void writeShort(ByteArrayOutputStream out, int value) {
        out.write(value & 0xff);
        out.write((value >>> 8) & 0xff);
    }

    private static void writeInt(ByteArrayOutputStream out, int value) {
        out.write(value & 0xff);
        out.write((value >>> 8) & 0xff);
        out.write((value >>> 16) & 0xff);
        out.write((value >>> 24) & 0xff);
    }

    private static void writeLong(ByteArrayOutputStream out, long value) {
        for (int shift = 0; shift < 64; shift += 8) {
            out.write((int) ((value >>> shift) & 0xff));
        }
    }

    private static void writeBytes(ByteArrayOutputStream out, byte[] bytes)
            throws IOException {
        writeInt(out, bytes.length);
        out.write(bytes);
    }

    private static void writeString(ByteArrayOutputStream out, String value)
            throws IOException {
        writeBytes(out, value.getBytes(StandardCharsets.UTF_8));
    }

    private static final class IdempotencyRecord {
        final String path;
        final byte[] targetHash;
        final byte[] receipt;

        IdempotencyRecord(String path, byte[] targetHash, byte[] receipt) {
            this.path = path;
            this.targetHash = targetHash;
            this.receipt = receipt;
        }
    }

    private static final class RollbackState {
        final boolean existed;
        final byte[] previous;

        RollbackState(boolean existed, byte[] previous) {
            this.existed = existed;
            this.previous = previous;
        }
    }

    private static final class Cursor {
        private final byte[] bytes;
        private int offset;

        Cursor(byte[] bytes) {
            this.bytes = bytes;
        }

        int u8() throws IOException {
            return take(1)[0] & 0xff;
        }

        byte[] bytes() throws IOException {
            int length = intLe();
            if (length < 0) {
                throw new IOException("negative length");
            }
            return take(length);
        }

        String string() throws IOException {
            return new String(bytes(), StandardCharsets.UTF_8);
        }

        byte[] take(int length) throws IOException {
            if (length < 0 || offset > bytes.length - length) {
                throw new IOException("truncated record");
            }
            byte[] value = Arrays.copyOfRange(bytes, offset, offset + length);
            offset += length;
            return value;
        }

        int intLe() throws IOException {
            byte[] value = take(4);
            return (value[0] & 0xff)
                    | ((value[1] & 0xff) << 8)
                    | ((value[2] & 0xff) << 16)
                    | ((value[3] & 0xff) << 24);
        }

        boolean finished() {
            return offset == bytes.length;
        }
    }
}
