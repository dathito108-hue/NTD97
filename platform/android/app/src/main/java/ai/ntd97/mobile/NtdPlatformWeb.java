package ai.ntd97.mobile;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Locale;

final class NtdPlatformWeb {
    private static final byte[] MAGIC = new byte[]{'N', 'W', 'R', '9', '7', 0};
    private static final int VERSION = 1;
    private static final int MAX_URL_BYTES = 4096;
    private static final int MAX_CONTENT_TYPE_BYTES = 512;

    byte[] fetch(
            String requestedUrl,
            int maxBytes,
            int connectTimeoutMs,
            int readTimeoutMs,
            int maxRedirects) {
        if (requestedUrl == null
                || requestedUrl.isEmpty()
                || requestedUrl.getBytes(StandardCharsets.UTF_8).length > MAX_URL_BYTES
                || maxBytes <= 0
                || maxBytes > 64 * 1024
                || connectTimeoutMs <= 0
                || readTimeoutMs <= 0
                || maxRedirects < 0
                || maxRedirects > 5) {
            return encodeFailure("invalid-request");
        }

        try {
            URL current = new URL(requestedUrl);
            if (!supportedScheme(current)) {
                return encodeFailure("unsupported-scheme");
            }

            int redirects = 0;
            while (true) {
                HttpURLConnection connection = (HttpURLConnection) current.openConnection();
                connection.setConnectTimeout(connectTimeoutMs);
                connection.setReadTimeout(readTimeoutMs);
                connection.setInstanceFollowRedirects(false);
                connection.setUseCaches(false);
                connection.setRequestMethod("GET");
                connection.setRequestProperty("Accept-Encoding", "identity");
                connection.setRequestProperty("User-Agent", "NTD97-Mobile/0.1");
                connection.connect();

                int status = connection.getResponseCode();
                if (isRedirect(status)) {
                    String location = connection.getHeaderField("Location");
                    connection.disconnect();
                    if (location == null || location.isEmpty() || redirects >= maxRedirects) {
                        return encodeFailure("redirect-limit");
                    }
                    URL next = new URL(current, location);
                    if (!supportedScheme(next)) {
                        return encodeFailure("redirect-unsupported-scheme");
                    }
                    if ("https".equalsIgnoreCase(current.getProtocol())
                            && "http".equalsIgnoreCase(next.getProtocol())) {
                        return encodeFailure("redirect-downgrade");
                    }
                    current = next;
                    redirects++;
                    continue;
                }

                long declaredLength = connection.getContentLengthLong();
                if (declaredLength > maxBytes) {
                    connection.disconnect();
                    return encodeFailure("body-too-large");
                }

                String contentType = connection.getContentType();
                if (contentType == null) {
                    contentType = "";
                }
                contentType = truncateUtf8(contentType, MAX_CONTENT_TYPE_BYTES);

                InputStream stream = status >= 400
                        ? connection.getErrorStream()
                        : connection.getInputStream();
                byte[] body = readBounded(stream, maxBytes);
                String finalUrl = truncateUtf8(
                        connection.getURL().toExternalForm(),
                        MAX_URL_BYTES);
                connection.disconnect();

                if (body == null) {
                    return encodeFailure("body-too-large");
                }
                return encodeResponse(
                        status,
                        redirects,
                        finalUrl,
                        contentType,
                        body,
                        "");
            }
        } catch (IOException | RuntimeException error) {
            String name = error.getClass().getSimpleName();
            return encodeFailure("transport-" + name.toLowerCase(Locale.ROOT));
        }
    }

    private static boolean supportedScheme(URL url) {
        String scheme = url.getProtocol();
        return "http".equalsIgnoreCase(scheme) || "https".equalsIgnoreCase(scheme);
    }

    private static boolean isRedirect(int status) {
        return status == 301
                || status == 302
                || status == 303
                || status == 307
                || status == 308;
    }

    private static byte[] readBounded(InputStream input, int maxBytes) throws IOException {
        if (input == null) {
            return new byte[0];
        }
        try (InputStream stream = input;
             ByteArrayOutputStream output = new ByteArrayOutputStream(Math.min(maxBytes, 8192))) {
            byte[] buffer = new byte[8192];
            int total = 0;
            while (true) {
                int count = stream.read(buffer);
                if (count == -1) {
                    return output.toByteArray();
                }
                total += count;
                if (total > maxBytes) {
                    return null;
                }
                output.write(buffer, 0, count);
            }
        }
    }

    private static String truncateUtf8(String value, int maxBytes) {
        byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
        if (bytes.length <= maxBytes) {
            return value;
        }
        int end = maxBytes;
        while (end > 0 && (bytes[end] & 0xC0) == 0x80) {
            end--;
        }
        return new String(bytes, 0, end, StandardCharsets.UTF_8);
    }

    private static byte[] encodeFailure(String error) {
        return encodeResponse(0, 0, "", "", new byte[0], error);
    }

    private static byte[] encodeResponse(
            int status,
            int redirects,
            String finalUrl,
            String contentType,
            byte[] body,
            String error) {
        try {
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            DataOutputStream output = new DataOutputStream(bytes);
            output.write(MAGIC);
            writeU16Le(output, VERSION);
            writeI32Le(output, status);
            writeU32Le(output, redirects);
            writeBytes(output, finalUrl.getBytes(StandardCharsets.UTF_8));
            writeBytes(output, contentType.getBytes(StandardCharsets.UTF_8));
            writeBytes(output, body);
            writeBytes(output, error.getBytes(StandardCharsets.UTF_8));
            output.flush();
            return bytes.toByteArray();
        } catch (IOException impossible) {
            return new byte[0];
        }
    }

    private static void writeBytes(DataOutputStream output, byte[] value) throws IOException {
        writeU32Le(output, value.length);
        output.write(value);
    }

    private static void writeU16Le(DataOutputStream output, int value) throws IOException {
        output.writeByte(value & 0xff);
        output.writeByte((value >>> 8) & 0xff);
    }

    private static void writeU32Le(DataOutputStream output, int value) throws IOException {
        output.writeByte(value & 0xff);
        output.writeByte((value >>> 8) & 0xff);
        output.writeByte((value >>> 16) & 0xff);
        output.writeByte((value >>> 24) & 0xff);
    }

    private static void writeI32Le(DataOutputStream output, int value) throws IOException {
        writeU32Le(output, value);
    }
}
