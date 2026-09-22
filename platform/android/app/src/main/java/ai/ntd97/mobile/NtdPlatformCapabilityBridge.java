package ai.ntd97.mobile;

import android.content.Context;
import android.net.Uri;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.Inet6Address;
import java.net.InetAddress;
import java.net.URI;
import java.net.URL;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

final class NtdPlatformCapabilityBridge {
    private static final int PROTOCOL_VERSION = 1;
    private static final int MAX_WEB_BYTES = 64 * 1024;
    private static final int MAX_FILE_BYTES = 256 * 1024;
    private static final int MAX_REDIRECTS = 3;
    private static final int CONNECT_TIMEOUT_MS = 5000;
    private static final int READ_TIMEOUT_MS = 8000;
    private static final Pattern SEARCH_LINK = Pattern.compile(
            "<a[^>]+class=\\\"[^\\\"]*result__a[^\\\"]*\\\"[^>]+href=\\\"([^\\\"]+)\\\"[^>]*>(.*?)</a>",
            Pattern.CASE_INSENSITIVE | Pattern.DOTALL);
    private static final Pattern TAGS = Pattern.compile("<[^>]+>");

    private static volatile Context appContext;

    private NtdPlatformCapabilityBridge() {}

    static void initialize(Context context) {
        appContext = context.getApplicationContext();
    }

    static byte[] webFetch(String value, int requestedMaxBytes) {
        try {
            int maxBytes = bounded(requestedMaxBytes, MAX_WEB_BYTES);
            FetchResult result = fetchHttps(value, maxBytes);
            return success(
                    result.statusCode,
                    result.finalUrl,
                    "android-https-fetch",
                    result.body);
        } catch (Exception error) {
            return failure(safeMessage(error));
        }
    }

    static byte[] webSearch(String query, int maxResults) {
        try {
            String normalized = query == null ? "" : query.trim();
            if (normalized.isEmpty()) {
                throw new IOException("empty search query");
            }
            int limit = Math.max(1, Math.min(maxResults, 10));
            String encoded = URLEncoder.encode(normalized, StandardCharsets.UTF_8.name());
            FetchResult result = fetchHttps(
                    "https://duckduckgo.com/html/?q=" + encoded,
                    MAX_WEB_BYTES);
            String html = new String(result.body, StandardCharsets.UTF_8);
            List<String> rows = parseSearchRows(html, limit);
            if (rows.isEmpty()) {
                throw new IOException("search provider returned no parseable results");
            }
            return success(
                    result.statusCode,
                    result.finalUrl,
                    "android-https-search:duckduckgo",
                    String.join("\n", rows).getBytes(StandardCharsets.UTF_8));
        } catch (Exception error) {
            return failure(safeMessage(error));
        }
    }

    static byte[] fileRead(String virtualPath, int requestedMaxBytes) {
        try {
            Context context = appContext;
            if (context == null) {
                throw new IOException("platform bridge not initialized");
            }
            int maxBytes = bounded(requestedMaxBytes, MAX_FILE_BYTES);
            File target = resolveAppPrivatePath(context, virtualPath);
            if (!target.isFile()) {
                throw new IOException("scoped file does not exist");
            }
            byte[] bytes;
            try (FileInputStream input = new FileInputStream(target)) {
                bytes = readBounded(input, maxBytes);
            }
            return success(
                    0,
                    canonicalVirtualPath(context, target),
                    "android-app-private-file",
                    bytes);
        } catch (Exception error) {
            return failure(safeMessage(error));
        }
    }

    private static FetchResult fetchHttps(String value, int maxBytes) throws Exception {
        URI current = validatePublicHttps(value);
        for (int redirect = 0; redirect <= MAX_REDIRECTS; redirect++) {
            HttpURLConnection connection = (HttpURLConnection) current.toURL().openConnection();
            connection.setInstanceFollowRedirects(false);
            connection.setConnectTimeout(CONNECT_TIMEOUT_MS);
            connection.setReadTimeout(READ_TIMEOUT_MS);
            connection.setRequestMethod("GET");
            connection.setRequestProperty("Accept", "text/plain,text/html,application/json,application/xml;q=0.9,*/*;q=0.1");
            connection.setRequestProperty("User-Agent", "NTD97-Mobile/0.1");
            connection.connect();

            int status = connection.getResponseCode();
            if (status >= 300 && status < 400) {
                String location = connection.getHeaderField("Location");
                connection.disconnect();
                if (location == null || location.trim().isEmpty()) {
                    throw new IOException("redirect without location");
                }
                if (redirect == MAX_REDIRECTS) {
                    throw new IOException("too many redirects");
                }
                current = validatePublicHttps(current.resolve(location.trim()).toString());
                continue;
            }
            if (status < 200 || status >= 300) {
                connection.disconnect();
                throw new IOException("HTTP status " + status);
            }

            int declared = connection.getContentLength();
            if (declared > maxBytes) {
                connection.disconnect();
                throw new IOException("response exceeds byte limit");
            }
            byte[] body;
            try (InputStream input = connection.getInputStream()) {
                body = readBounded(input, maxBytes);
            } finally {
                connection.disconnect();
            }
            return new FetchResult(status, current.toString(), body);
        }
        throw new IOException("redirect loop");
    }

    private static URI validatePublicHttps(String value) throws Exception {
        if (value == null) {
            throw new IOException("missing URL");
        }
        URI uri = new URI(value.trim()).normalize();
        if (!"https".equalsIgnoreCase(uri.getScheme())
                || uri.getHost() == null
                || uri.getUserInfo() != null
                || (uri.getPort() != -1 && uri.getPort() != 443)) {
            throw new IOException("only public HTTPS URLs on port 443 are allowed");
        }
        InetAddress[] addresses = InetAddress.getAllByName(uri.getHost());
        if (addresses.length == 0) {
            throw new IOException("hostname did not resolve");
        }
        for (InetAddress address : addresses) {
            if (!isPublicAddress(address)) {
                throw new IOException("private or local network destination rejected");
            }
        }
        return uri;
    }

    private static boolean isPublicAddress(InetAddress address) {
        if (address.isAnyLocalAddress()
                || address.isLoopbackAddress()
                || address.isLinkLocalAddress()
                || address.isSiteLocalAddress()
                || address.isMulticastAddress()) {
            return false;
        }

        byte[] bytes = address.getAddress();
        if (address instanceof Inet6Address) {
            int first = Byte.toUnsignedInt(bytes[0]);
            return (first & 0xfe) != 0xfc;
        }

        if (bytes.length == 4) {
            int a = Byte.toUnsignedInt(bytes[0]);
            int b = Byte.toUnsignedInt(bytes[1]);
            if (a == 0
                    || a == 10
                    || a == 127
                    || (a == 100 && b >= 64 && b <= 127)
                    || (a == 169 && b == 254)
                    || (a == 172 && b >= 16 && b <= 31)
                    || (a == 192 && b == 168)
                    || (a == 198 && (b == 18 || b == 19))
                    || a >= 224) {
                return false;
            }
        }
        return true;
    }

    private static File resolveAppPrivatePath(Context context, String virtualPath) throws IOException {
        if (virtualPath == null) {
            throw new IOException("missing virtual path");
        }
        String path = virtualPath.trim();
        final File root;
        final String relative;
        if (path.startsWith("app://files/")) {
            root = context.getFilesDir();
            relative = path.substring("app://files/".length());
        } else if (path.startsWith("app://cache/")) {
            root = context.getCacheDir();
            relative = path.substring("app://cache/".length());
        } else {
            throw new IOException("unsupported scoped file namespace");
        }
        if (relative.isEmpty() || relative.indexOf('\0') >= 0) {
            throw new IOException("invalid scoped file path");
        }

        File canonicalRoot = root.getCanonicalFile();
        File target = new File(canonicalRoot, relative).getCanonicalFile();
        String rootPath = canonicalRoot.getPath();
        String targetPath = target.getPath();
        if (!targetPath.startsWith(rootPath + File.separator)) {
            throw new IOException("scoped file traversal rejected");
        }
        return target;
    }

    private static String canonicalVirtualPath(Context context, File target) throws IOException {
        File files = context.getFilesDir().getCanonicalFile();
        File cache = context.getCacheDir().getCanonicalFile();
        File canonical = target.getCanonicalFile();
        if (canonical.getPath().startsWith(files.getPath() + File.separator)) {
            return "app://files/" + relativePath(files, canonical);
        }
        if (canonical.getPath().startsWith(cache.getPath() + File.separator)) {
            return "app://cache/" + relativePath(cache, canonical);
        }
        throw new IOException("file escaped scoped roots");
    }

    private static String relativePath(File root, File target) {
        return root.toPath().relativize(target.toPath()).toString().replace(File.separatorChar, '/');
    }

    private static List<String> parseSearchRows(String html, int limit) throws Exception {
        Matcher matcher = SEARCH_LINK.matcher(html);
        Set<String> seen = new LinkedHashSet<>();
        List<String> rows = new ArrayList<>();
        while (matcher.find() && rows.size() < limit) {
            String rawHref = htmlDecode(matcher.group(1));
            String title = htmlDecode(TAGS.matcher(matcher.group(2)).replaceAll(" "))
                    .replaceAll("\\s+", " ")
                    .trim();
            String href = canonicalSearchUrl(rawHref);
            if (href == null || !seen.add(href)) {
                continue;
            }
            rows.add(title.replace('\t', ' ') + "\t" + href);
        }
        return rows;
    }

    private static String canonicalSearchUrl(String raw) throws Exception {
        URI uri = new URI(raw);
        if (!uri.isAbsolute()) {
            uri = new URI("https://duckduckgo.com").resolve(uri);
        }
        if ("duckduckgo.com".equalsIgnoreCase(uri.getHost())
                && uri.getPath() != null
                && uri.getPath().startsWith("/l/")) {
            String encoded = Uri.parse(uri.toString()).getQueryParameter("uddg");
            if (encoded != null) {
                uri = new URI(URLDecoder.decode(encoded, StandardCharsets.UTF_8.name()));
            }
        }
        URI validated = validatePublicHttps(uri.toString());
        return validated.toString();
    }

    private static String htmlDecode(String value) {
        return value
                .replace("&amp;", "&")
                .replace("&quot;", "\"")
                .replace("&#39;", "'")
                .replace("&lt;", "<")
                .replace("&gt;", ">");
    }

    private static byte[] readBounded(InputStream input, int maxBytes) throws IOException {
        ByteArrayOutputStream output = new ByteArrayOutputStream(Math.min(maxBytes, 16 * 1024));
        byte[] buffer = new byte[8192];
        int total = 0;
        int read;
        while ((read = input.read(buffer)) != -1) {
            total += read;
            if (total > maxBytes) {
                throw new IOException("payload exceeds byte limit");
            }
            output.write(buffer, 0, read);
        }
        return output.toByteArray();
    }

    private static int bounded(int requested, int hardMax) {
        if (requested <= 0) {
            return hardMax;
        }
        return Math.min(requested, hardMax);
    }

    private static byte[] success(
            int code,
            String identifier,
            String evidence,
            byte[] payload) {
        return frame(true, code, identifier, evidence, payload);
    }

    private static byte[] failure(String message) {
        return frame(
                false,
                0,
                "",
                "android-platform-error",
                message.getBytes(StandardCharsets.UTF_8));
    }

    private static byte[] frame(
            boolean success,
            int code,
            String identifier,
            String evidence,
            byte[] payload) {
        byte[] identifierBytes = identifier.getBytes(StandardCharsets.UTF_8);
        byte[] evidenceBytes = evidence.getBytes(StandardCharsets.UTF_8);
        ByteArrayOutputStream output = new ByteArrayOutputStream(
                20 + identifierBytes.length + evidenceBytes.length + payload.length);
        output.write(PROTOCOL_VERSION);
        output.write(success ? 1 : 0);
        output.write(0);
        output.write(0);
        writeIntLe(output, code);
        writeIntLe(output, identifierBytes.length);
        output.write(identifierBytes, 0, identifierBytes.length);
        writeIntLe(output, evidenceBytes.length);
        output.write(evidenceBytes, 0, evidenceBytes.length);
        writeIntLe(output, payload.length);
        output.write(payload, 0, payload.length);
        return output.toByteArray();
    }

    private static void writeIntLe(ByteArrayOutputStream output, int value) {
        output.write(value & 0xff);
        output.write((value >>> 8) & 0xff);
        output.write((value >>> 16) & 0xff);
        output.write((value >>> 24) & 0xff);
    }

    private static String safeMessage(Exception error) {
        String message = error.getMessage();
        if (message == null || message.trim().isEmpty()) {
            return error.getClass().getSimpleName();
        }
        return message.replace('\n', ' ').replace('\r', ' ').trim();
    }

    private static final class FetchResult {
        final int statusCode;
        final String finalUrl;
        final byte[] body;

        FetchResult(int statusCode, String finalUrl, byte[] body) {
            this.statusCode = statusCode;
            this.finalUrl = finalUrl;
            this.body = body;
        }
    }
}
