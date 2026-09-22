package ai.ntd97.mobile;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.InetAddress;
import java.net.URL;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.net.ssl.HttpsURLConnection;

final class NtdWebPlatform {
    private static final int CONNECT_TIMEOUT_MS = 10_000;
    private static final int READ_TIMEOUT_MS = 10_000;
    private static final int MAX_BODY_BYTES = 512 * 1024;
    private static final int MAX_REDIRECTS = 3;
    private static final int PLATFORM_TIMEOUT_SECONDS = 45;
    private static final int MAX_SEARCH_QUERY_CHARS = 1024;
    private static final int MAX_SEARCH_RESULTS = 10;
    private static final int MAX_SEARCH_FIELD_CHARS = 4096;
    private static final byte PROTOCOL_VERSION = 1;

    private static final ExecutorService NETWORK_EXECUTOR =
            Executors.newSingleThreadExecutor(runnable -> {
                Thread thread = new Thread(runnable, "ntd97-https");
                thread.setDaemon(true);
                return thread;
            });

    private static final SearchProvider[] SEARCH_PROVIDERS = {
        new DuckDuckGoHtmlProvider()
    };

    private NtdWebPlatform() {}

    static byte[] fetch(String rawUrl) {
        Future<byte[]> future = NETWORK_EXECUTOR.submit(() -> fetchBlocking(rawUrl));
        return awaitPlatformResult(future, "HTTPS fetch");
    }

    static byte[] search(String query, int maxResults) {
        if (query == null
                || query.trim().isEmpty()
                || query.length() > MAX_SEARCH_QUERY_CHARS
                || maxResults <= 0
                || maxResults > MAX_SEARCH_RESULTS) {
            return encodeError("invalid web search request");
        }
        String normalized = query.trim();
        Future<byte[]> future =
                NETWORK_EXECUTOR.submit(() -> searchBlocking(normalized, maxResults));
        return awaitPlatformResult(future, "web search");
    }

    private static byte[] awaitPlatformResult(Future<byte[]> future, String operation) {
        try {
            return future.get(PLATFORM_TIMEOUT_SECONDS, TimeUnit.SECONDS);
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            future.cancel(true);
            return encodeError(operation + " interrupted");
        } catch (TimeoutException error) {
            future.cancel(true);
            return encodeError(operation + " timed out");
        } catch (ExecutionException error) {
            Throwable cause = error.getCause();
            return encodeError(cause == null ? operation + " failed" : safeMessage(cause));
        }
    }

    private static byte[] fetchBlocking(String rawUrl) {
        try {
            HttpResponse response = fetchResponseBlocking(
                    rawUrl,
                    "text/plain,text/html,application/json,application/xml;q=0.9,*/*;q=0.1");
            return encodeFetch(
                    response.status,
                    response.finalUrl,
                    response.contentType,
                    response.body);
        } catch (Exception error) {
            return encodeError(safeMessage(error));
        }
    }

    private static byte[] searchBlocking(String query, int maxResults) {
        StringBuilder failures = new StringBuilder();
        for (SearchProvider provider : SEARCH_PROVIDERS) {
            try {
                List<SearchResult> results = provider.search(query, maxResults);
                if (!results.isEmpty()) {
                    return encodeSearch(provider.id(), results);
                }
                appendFailure(failures, provider.id() + ": no results");
            } catch (Exception error) {
                appendFailure(failures, provider.id() + ": " + safeMessage(error));
            }
        }
        return encodeError(
                failures.length() == 0
                        ? "no web search provider available"
                        : failures.toString());
    }

    private static void appendFailure(StringBuilder failures, String value) {
        if (failures.length() > 0) {
            failures.append("; ");
        }
        failures.append(value);
    }

    private static HttpResponse fetchResponseBlocking(String rawUrl, String accept)
            throws IOException {
        URL current = validateUrl(rawUrl);
        for (int redirect = 0; redirect <= MAX_REDIRECTS; redirect++) {
            HttpsURLConnection connection = (HttpsURLConnection) current.openConnection();
            connection.setInstanceFollowRedirects(false);
            connection.setConnectTimeout(CONNECT_TIMEOUT_MS);
            connection.setReadTimeout(READ_TIMEOUT_MS);
            connection.setRequestMethod("GET");
            connection.setRequestProperty("Accept", accept);
            connection.setRequestProperty(
                    "User-Agent",
                    "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 NTD97-Mobile/1");
            connection.connect();

            int status = connection.getResponseCode();
            if (status >= 300 && status < 400) {
                if (redirect == MAX_REDIRECTS) {
                    connection.disconnect();
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
            return new HttpResponse(
                    status,
                    current.toExternalForm(),
                    contentType == null ? "" : contentType,
                    body);
        }
        throw new IOException("unreachable redirect state");
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

    private static byte[] encodeFetch(
            int status,
            String finalUrl,
            String contentType,
            byte[] body) {
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

    private static byte[] encodeSearch(String provider, List<SearchResult> results) {
        byte[] providerBytes = provider.getBytes(StandardCharsets.UTF_8);
        int size = 1 + 1 + 4 + providerBytes.length + 4;
        List<byte[][]> encoded = new ArrayList<>(results.size());
        for (SearchResult result : results) {
            byte[][] fields = {
                result.title.getBytes(StandardCharsets.UTF_8),
                result.url.getBytes(StandardCharsets.UTF_8),
                result.snippet.getBytes(StandardCharsets.UTF_8)
            };
            for (byte[] field : fields) {
                size = Math.addExact(size, 4 + field.length);
            }
            encoded.add(fields);
        }

        ByteBuffer buffer = ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN);
        buffer.put(PROTOCOL_VERSION);
        buffer.put((byte) 1);
        putBytes(buffer, providerBytes);
        buffer.putInt(results.size());
        for (byte[][] fields : encoded) {
            for (byte[] field : fields) {
                putBytes(buffer, field);
            }
        }
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

    private interface SearchProvider {
        String id();

        List<SearchResult> search(String query, int maxResults) throws IOException;
    }

    private static final class DuckDuckGoHtmlProvider implements SearchProvider {
        private static final String ID = "duckduckgo-html";
        private static final Pattern RESULT_LINK = Pattern.compile(
                "<a[^>]*class=\\"[^\\"]*result__a[^\\"]*\\"[^>]*href=\\"([^\\"]+)\\"[^>]*>(.*?)</a>",
                Pattern.CASE_INSENSITIVE | Pattern.DOTALL);
        private static final Pattern SNIPPET = Pattern.compile(
                "<(?:a|div)[^>]*class=\\"[^\\"]*result__snippet[^\\"]*\\"[^>]*>(.*?)</(?:a|div)>",
                Pattern.CASE_INSENSITIVE | Pattern.DOTALL);
        private static final Pattern TAG = Pattern.compile("<[^>]+>");

        @Override
        public String id() {
            return ID;
        }

        @Override
        public List<SearchResult> search(String query, int maxResults) throws IOException {
            String encoded = URLEncoder.encode(query, StandardCharsets.UTF_8.name());
            HttpResponse response = fetchResponseBlocking(
                    "https://html.duckduckgo.com/html/?q=" + encoded,
                    "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1");
            String html = new String(response.body, StandardCharsets.UTF_8);
            Matcher links = RESULT_LINK.matcher(html);
            List<SearchResult> results = new ArrayList<>();
            Set<String> seen = new HashSet<>();

            while (links.find() && results.size() < maxResults) {
                String title = normalizeText(links.group(2));
                String target = normalizeResultUrl(links.group(1));
                if (title.isEmpty() || target.isEmpty() || !seen.add(target)) {
                    continue;
                }

                int snippetEnd = Math.min(html.length(), links.end() + 6000);
                Matcher snippetMatcher = SNIPPET.matcher(html.substring(links.end(), snippetEnd));
                String snippet = snippetMatcher.find()
                        ? normalizeText(snippetMatcher.group(1))
                        : "";

                results.add(new SearchResult(
                        limitField(title),
                        target,
                        limitField(snippet)));
            }
            return results;
        }

        private static String normalizeResultUrl(String raw) {
            try {
                String decoded = decodeHtmlEntities(raw.trim());
                URL link;
                if (decoded.startsWith("//")) {
                    link = new URL("https:" + decoded);
                } else if (decoded.startsWith("/")) {
                    link = new URL("https://html.duckduckgo.com" + decoded);
                } else {
                    link = new URL(decoded);
                }

                String target = queryParameter(link, "uddg");
                if (target == null || target.isEmpty()) {
                    target = link.toExternalForm();
                }
                return validateUrl(target).toExternalForm();
            } catch (Exception error) {
                return "";
            }
        }

        private static String queryParameter(URL url, String name) throws IOException {
            String query = url.getQuery();
            if (query == null || query.isEmpty()) {
                return null;
            }
            for (String part : query.split("&")) {
                int equals = part.indexOf('=');
                String key = equals >= 0 ? part.substring(0, equals) : part;
                if (!name.equals(URLDecoder.decode(key, StandardCharsets.UTF_8.name()))) {
                    continue;
                }
                String value = equals >= 0 ? part.substring(equals + 1) : "";
                return URLDecoder.decode(value, StandardCharsets.UTF_8.name());
            }
            return null;
        }

        private static String normalizeText(String html) {
            String withoutTags = TAG.matcher(html).replaceAll(" ");
            String decoded = decodeHtmlEntities(withoutTags);
            return decoded.replaceAll("\\s+", " ").trim();
        }

        private static String decodeHtmlEntities(String value) {
            return value
                    .replace("&amp;", "&")
                    .replace("&quot;", "\"")
                    .replace("&#x27;", "'")
                    .replace("&#39;", "'")
                    .replace("&lt;", "<")
                    .replace("&gt;", ">")
                    .replace("&nbsp;", " ");
        }

        private static String limitField(String value) {
            return value.length() <= MAX_SEARCH_FIELD_CHARS
                    ? value
                    : value.substring(0, MAX_SEARCH_FIELD_CHARS);
        }
    }

    private static final class HttpResponse {
        final int status;
        final String finalUrl;
        final String contentType;
        final byte[] body;

        HttpResponse(int status, String finalUrl, String contentType, byte[] body) {
            this.status = status;
            this.finalUrl = finalUrl;
            this.contentType = contentType;
            this.body = body;
        }
    }

    private static final class SearchResult {
        final String title;
        final String url;
        final String snippet;

        SearchResult(String title, String url, String snippet) {
            this.title = title;
            this.url = url;
            this.snippet = snippet;
        }
    }
}
