package ai.ntd97.mobile;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.InetAddress;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import javax.net.ssl.HttpsURLConnection;

final class NtdWebPlatform {
    private static final int CONNECT_TIMEOUT_MS = 10_000;
    private static final int READ_TIMEOUT_MS = 10_000;
    private static final int MAX_BODY_BYTES = 512 * 1024;
    private static final int MAX_REDIRECTS = 3;
    private static final int PLATFORM_TIMEOUT_SECONDS = 45;
    private static final byte PROTOCOL_VERSION = 1;
    private static final ExecutorService NETWORK_EXECUTOR =
            Executors.newSingleThreadExecutor(runnable -> {
                Thread thread = new Thread(runnable, "ntd97-https");
                thread.setDaemon(true);
                return thread;
            });

    private NtdWebPlatform() {}

    static byte[] fetch(String rawUrl) {
        Future<byte[]> future = NETWORK_EXECUTOR.submit(() -> fetchBlocking(rawUrl));
        try {
            return future.get(PLATFORM_TIMEOUT_SECONDS, TimeUnit.SECONDS);
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            future.cancel(true);
            return encodeError("HTTPS fetch interrupted");
        } catch (TimeoutException error) {
            future.cancel(true);
            return encodeError("HTTPS fetch timed out");
        } catch (ExecutionException error) {
            Throwable cause = error.getCause();
            return encodeError(cause == null ? "HTTPS fetch failed" : safeMessage(cause));
        }
    }

    private static byte[] fetchBlocking(String rawUrl) {
        try {
            URL current = validateUrl(rawUrl);
            for (int redirect = 0; redirect <= MAX_REDIRECTS; redirect++) {
                HttpsURLConnection connection = (HttpsURLConnection) current.openConnection();
                connection.setInstanceFollowRedirects(false);
                connection.setConnectTimeout(CONNECT_TIMEOUT_MS);
                connection.setReadTimeout(READ_TIMEOUT_MS);
                connection.setRequestMethod("GET");
                connection.setRequestProperty(
                        "Accept",
                        "text/plain,text/html,application/json,application/xml;q=0.9,*/*;q=0.1");
                connection.setRequestProperty("User-Agent", "NTD97-Mobile/1");
                connection.connect();

                int status = connection.getResponseCode();
                if (status >= 300 && status < 400) {
                    if (redirect == MAX_REDIRECTS) {
                        throw new IOException("redirect limit exceeded");
                    }
                    String location = connection.getHeaderField("Location");
                    connection.disconnect();
                    if (location == null || location.trim().isEmpty()) {
                        throw new IOException("redirect missing location");
                    }
                    current = validateUrl(new URL(current, location).toExternalForm());
                    continue;
                }

                if (status < 200 || status >= 300) {
                    connection.disconnect();
                    throw new IOException("HTTP status " + status);
                }

                long declared = connection.getContentLengthLong();
                if (declared > MAX_BODY_BYTES) {
                    connection.disconnect();
                    throw new IOException("response exceeds size limit");
                }

                byte[] body;
                try (InputStream input = connection.getInputStream()) {
                    body = readBounded(input);
                } finally {
                    connection.disconnect();
                }

                String contentType = connection.getContentType();
                return encode(
                        status,
                        current.toExternalForm(),
                        contentType == null ? "" : contentType,
                        body);
            }
            throw new IOException("unreachable redirect state");
        } catch (Exception error) {
            return encodeError(safeMessage(error));
        }
    }

    private static String safeMessage(Throwable error) {
        String message = error.getMessage();
        return message == null || message.trim().isEmpty()
                ? error.getClass().getSimpleName()
                : message;
    }

    private static URL validateUrl(String rawUrl) throws IOException {
        if (rawUrl == null || rawUrl.length() > 4096) {
            throw new IOException("invalid URL");
        }
        URL url = new URL(rawUrl.trim());
        if (!"https".equalsIgnoreCase(url.getProtocol())) {
            throw new IOException("only HTTPS is allowed");
        }
        if (url.getUserInfo() != null || url.getHost() == null || url.getHost().isEmpty()) {
            throw new IOException("invalid HTTPS authority");
        }
        if (url.getPort() != -1 && url.getPort() != 443) {
            throw new IOException("non-standard HTTPS port is not allowed");
        }

        InetAddress[] addresses = InetAddress.getAllByName(url.getHost());
        if (addresses.length == 0) {
            throw new IOException("host did not resolve");
        }
        for (InetAddress address : addresses) {
            if (isPrivateAddress(address)) {
                throw new IOException("private or local address is not allowed");
            }
        }
        return url;
    }

    private static boolean isPrivateAddress(InetAddress address) {
        if (address.isAnyLocalAddress()
                || address.isLoopbackAddress()
                || address.isLinkLocalAddress()
                || address.isSiteLocalAddress()
                || address.isMulticastAddress()) {
            return true;
        }
        byte[] bytes = address.getAddress();
        if (bytes.length == 16) {
            int first = Byte.toUnsignedInt(bytes[0]);
            if ((first & 0xfe) == 0xfc) {
                return true;
            }
        }
        return false;
    }

    private static byte[] readBounded(InputStream input) throws IOException {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        byte[] buffer = new byte[8192];
        int total = 0;
        while (true) {
            int count = input.read(buffer);
            if (count < 0) {
                break;
            }
            total += count;
            if (total > MAX_BODY_BYTES) {
                throw new IOException("response exceeds size limit");
            }
            output.write(buffer, 0, count);
        }
        return output.toByteArray();
    }

    private static byte[] encode(int status, String finalUrl, String contentType, byte[] body) {
        byte[] url = finalUrl.getBytes(StandardCharsets.UTF_8);
        byte[] type = contentType.getBytes(StandardCharsets.UTF_8);
        ByteBuffer buffer = ByteBuffer
                .allocate(1 + 1 + 4 + 4 + url.length + 4 + type.length + 4 + body.length)
                .order(ByteOrder.LITTLE_ENDIAN);
        buffer.put(PROTOCOL_VERSION);
        buffer.put((byte) 1);
        buffer.putInt(status);
        putBytes(buffer, url);
        putBytes(buffer, type);
        putBytes(buffer, body);
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

    private static void putBytes(ByteBuffer buffer, byte[] bytes) {
        buffer.putInt(bytes.length);
        buffer.put(bytes);
    }
}
