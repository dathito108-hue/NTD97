package ai.ntd97.mobile;

import android.content.Context;
import android.content.SharedPreferences;

import org.json.JSONArray;
import org.json.JSONObject;
import org.json.JSONTokener;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.URL;
import java.net.URLEncoder;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
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
    private static final int MAX_SEARCH_QUERY_BYTES = 2048;
    private static final int MAX_SEARCH_RESULTS = 20;
    private static final int MAX_SEARCH_TITLE_BYTES = 512;
    private static final int MAX_SEARCH_SNIPPET_BYTES = 4096;
    private static final byte PROTOCOL_VERSION = 1;
    private static final String PREFS_NAME = "ntd97-web";
    private static final String SEARCH_TEMPLATE_KEY = "search-endpoint-template";
    private static final String SEARCH_RESULTS_PATH_KEY = "search-results-path";
    private static final String SEARCH_TITLE_PATH_KEY = "search-title-path";
    private static final String SEARCH_URL_PATH_KEY = "search-url-path";
    private static final String SEARCH_SNIPPET_PATH_KEY = "search-snippet-path";

    private static volatile SearchConfiguration searchConfiguration = SearchConfiguration.empty();

    private static final ExecutorService NETWORK_EXECUTOR =
            Executors.newSingleThreadExecutor(runnable -> {
                Thread thread = new Thread(runnable, "ntd97-https");
                thread.setDaemon(true);
                return thread;
            });

    private NtdWebPlatform() {}

    static final class SearchConfiguration {
        final String endpointTemplate;
        final String resultsPath;
        final String titlePath;
        final String urlPath;
        final String snippetPath;

        SearchConfiguration(
                String endpointTemplate,
                String resultsPath,
                String titlePath,
                String urlPath,
                String snippetPath) {
            this.endpointTemplate = endpointTemplate;
            this.resultsPath = resultsPath;
            this.titlePath = titlePath;
            this.urlPath = urlPath;
            this.snippetPath = snippetPath;
        }

        static SearchConfiguration empty() {
            return new SearchConfiguration("", "", "", "", "");
        }

        boolean configured() {
            return !endpointTemplate.isEmpty();
        }
    }

    private static final class SearchItem {
        final String title;
        final String url;
        final String snippet;

        SearchItem(String title, String url, String snippet) {
            this.title = title;
            this.url = url;
            this.snippet = snippet;
        }
    }

    private static final class HttpResult {
        final int status;
        final String finalUrl;
        final String contentType;
        final byte[] body;

        HttpResult(int status, String finalUrl, String contentType, byte[] body) {
            this.status = status;
            this.finalUrl = finalUrl;
            this.contentType = contentType;
            this.body = body;
        }
    }

    static void initialize(Context context) {
        Context appContext = context.getApplicationContext();
        SharedPreferences preferences =
                appContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        String template = value(preferences, SEARCH_TEMPLATE_KEY);
        if (template.isEmpty()) {
            searchConfiguration = SearchConfiguration.empty();
            return;
        }
        try {
            SearchConfiguration configuration = buildSearchConfiguration(
                    template,
                    value(preferences, SEARCH_RESULTS_PATH_KEY),
                    value(preferences, SEARCH_TITLE_PATH_KEY),
                    value(preferences, SEARCH_URL_PATH_KEY),
                    value(preferences, SEARCH_SNIPPET_PATH_KEY));
            searchConfiguration = configuration;
        } catch (IOException ignored) {
            clearSearchPreferences(preferences);
            searchConfiguration = SearchConfiguration.empty();
        }
    }

    static SearchConfiguration searchConfiguration() {
        return searchConfiguration;
    }

    static boolean configureSearchProvider(
            Context context,
            String template,
            String resultsPath,
            String titlePath,
            String urlPath,
            String snippetPath) {
        Context appContext = context.getApplicationContext();
        SharedPreferences preferences =
                appContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        String normalizedTemplate = template == null ? "" : template.trim();
        if (normalizedTemplate.isEmpty()) {
            clearSearchPreferences(preferences);
            searchConfiguration = SearchConfiguration.empty();
            return true;
        }

        final SearchConfiguration configuration;
        try {
            configuration = buildSearchConfiguration(
                    normalizedTemplate,
                    resultsPath,
                    titlePath,
                    urlPath,
                    snippetPath);
        } catch (IOException error) {
            return false;
        }

        preferences.edit()
                .putString(SEARCH_TEMPLATE_KEY, configuration.endpointTemplate)
                .putString(SEARCH_RESULTS_PATH_KEY, configuration.resultsPath)
                .putString(SEARCH_TITLE_PATH_KEY, configuration.titlePath)
                .putString(SEARCH_URL_PATH_KEY, configuration.urlPath)
                .putString(SEARCH_SNIPPET_PATH_KEY, configuration.snippetPath)
                .apply();
        searchConfiguration = configuration;
        return true;
    }

    static boolean configureSearchEndpoint(Context context, String template) {
        SearchConfiguration current = searchConfiguration;
        String resultsPath = current.configured() ? current.resultsPath : "items";
        String titlePath = current.configured() ? current.titlePath : "title";
        String urlPath = current.configured() ? current.urlPath : "url";
        String snippetPath = current.configured() ? current.snippetPath : "snippet";
        return configureSearchProvider(
                context,
                template,
                resultsPath,
                titlePath,
                urlPath,
                snippetPath);
    }

    static byte[] search(String query, int maxResults) {
        Future<byte[]> future = NETWORK_EXECUTOR.submit(() -> searchBlocking(query, maxResults));
        try {
            return future.get(PLATFORM_TIMEOUT_SECONDS, TimeUnit.SECONDS);
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            future.cancel(true);
            return encodeError("web search interrupted");
        } catch (TimeoutException error) {
            future.cancel(true);
            return encodeError("web search timed out");
        } catch (ExecutionException error) {
            Throwable cause = error.getCause();
            return encodeError(cause == null ? "web search failed" : safeMessage(cause));
        }
    }

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

    static byte[] put(String rawUrl, byte[] body, String expectedSha256) {
        Future<byte[]> future = NETWORK_EXECUTOR.submit(
                () -> putBlocking(rawUrl, body, expectedSha256));
        try {
            return future.get(PLATFORM_TIMEOUT_SECONDS, TimeUnit.SECONDS);
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            future.cancel(true);
            return encodeError("HTTPS upload interrupted");
        } catch (TimeoutException error) {
            future.cancel(true);
            return encodeError("HTTPS upload timed out");
        } catch (ExecutionException error) {
            Throwable cause = error.getCause();
            return encodeError(cause == null ? "HTTPS upload failed" : safeMessage(cause));
        }
    }

    private static byte[] searchBlocking(String query, int maxResults) {
        try {
            if (query == null || query.trim().isEmpty()) {
                throw new IOException("empty search query");
            }
            byte[] queryBytes = query.getBytes(StandardCharsets.UTF_8);
            if (queryBytes.length > MAX_SEARCH_QUERY_BYTES) {
                throw new IOException("search query exceeds size limit");
            }
            if (maxResults <= 0 || maxResults > MAX_SEARCH_RESULTS) {
                throw new IOException("invalid search result limit");
            }

            SearchConfiguration configuration = searchConfiguration;
            if (!configuration.configured()) {
                throw new IOException("web search endpoint is not configured");
            }

            String encodedQuery = URLEncoder
                    .encode(query.trim(), StandardCharsets.UTF_8.name())
                    .replace("+", "%20");
            String rendered = configuration.endpointTemplate
                    .replace("{query}", encodedQuery)
                    .replace("{count}", Integer.toString(maxResults));
            if (rendered.indexOf('{') >= 0 || rendered.indexOf('}') >= 0) {
                throw new IOException("unsupported search endpoint placeholder");
            }

            HttpResult response = fetchResponse(rendered);
            List<SearchItem> items =
                    normalizeSearchResults(response.body, configuration, maxResults);
            return encodeSearch(
                    response.status,
                    searchSource(response.finalUrl),
                    items);
        } catch (Exception error) {
            return encodeError(safeMessage(error));
        }
    }

    private static SearchConfiguration buildSearchConfiguration(
            String template,
            String resultsPath,
            String titlePath,
            String urlPath,
            String snippetPath) throws IOException {
        String normalizedTemplate = template == null ? "" : template.trim();
        String normalizedResults = normalizeMappingPath(resultsPath, true);
        String normalizedTitle = normalizeMappingPath(titlePath, false);
        String normalizedUrl = normalizeMappingPath(urlPath, false);
        String normalizedSnippet = normalizeMappingPath(snippetPath, true);
        validateSearchTemplate(normalizedTemplate);
        return new SearchConfiguration(
                normalizedTemplate,
                normalizedResults,
                normalizedTitle,
                normalizedUrl,
                normalizedSnippet);
    }

    private static void validateSearchTemplate(String template) throws IOException {
        if (template.isEmpty() || template.length() > 4096 || !template.contains("{query}")) {
            throw new IOException("search endpoint must contain {query}");
        }
        String rendered = template
                .replace("{query}", "ntd97")
                .replace("{count}", "5");
        if (rendered.indexOf('{') >= 0 || rendered.indexOf('}') >= 0) {
            throw new IOException("unsupported search endpoint placeholder");
        }
        validateHttpsSyntax(rendered);
    }

    private static String normalizeMappingPath(String rawPath, boolean allowEmpty)
            throws IOException {
        String path = rawPath == null ? "" : rawPath.trim();
        if (path.isEmpty()) {
            if (allowEmpty) {
                return "";
            }
            throw new IOException("required search mapping path is empty");
        }
        if (path.length() > 512) {
            throw new IOException("search mapping path exceeds limit");
        }
        String[] segments = path.split("\\.", -1);
        for (String segment : segments) {
            if (segment.isEmpty() || segment.length() > 128) {
                throw new IOException("invalid search mapping path");
            }
            for (int index = 0; index < segment.length(); index++) {
                char value = segment.charAt(index);
                if (!Character.isLetterOrDigit(value) && value != '_' && value != '-') {
                    throw new IOException("invalid search mapping path");
                }
            }
        }
        return String.join(".", segments);
    }

    private static List<SearchItem> normalizeSearchResults(
            byte[] body,
            SearchConfiguration configuration,
            int maxResults) throws Exception {
        if (body.length == 0) {
            throw new IOException("search provider returned empty body");
        }
        String json = new String(body, StandardCharsets.UTF_8);
        Object root = new JSONTokener(json).nextValue();
        Object resultContainer = resolveJsonPath(root, configuration.resultsPath);
        if (!(resultContainer instanceof JSONArray)) {
            throw new IOException("search results mapping did not resolve to an array");
        }

        JSONArray array = (JSONArray) resultContainer;
        List<SearchItem> results = new ArrayList<>();
        Set<String> seenUrls = new HashSet<>();
        int limit = Math.min(maxResults, MAX_SEARCH_RESULTS);
        for (int index = 0; index < array.length() && results.size() < limit; index++) {
            Object rawItem = array.get(index);
            if (!(rawItem instanceof JSONObject)) {
                throw new IOException("search result item is not an object");
            }

            JSONObject item = (JSONObject) rawItem;
            String title = normalizeSearchText(
                    requiredJsonString(item, configuration.titlePath),
                    MAX_SEARCH_TITLE_BYTES,
                    true);
            String rawUrl = requiredJsonString(item, configuration.urlPath);
            String url;
            try {
                url = validatePublicHttpsUrl(rawUrl).toExternalForm();
            } catch (IOException unsafeUrl) {
                continue;
            }
            if (!seenUrls.add(url)) {
                continue;
            }

            String snippet = "";
            if (!configuration.snippetPath.isEmpty()) {
                snippet = normalizeSearchText(
                        optionalJsonString(item, configuration.snippetPath),
                        MAX_SEARCH_SNIPPET_BYTES,
                        false);
            }
            results.add(new SearchItem(title, url, snippet));
        }

        if (array.length() > 0 && results.isEmpty()) {
            throw new IOException("search provider returned no safe normalized result");
        }
        return results;
    }

    private static Object resolveJsonPath(Object root, String path) throws IOException {
        if (path == null || path.isEmpty()) {
            return root;
        }
        Object current = root;
        for (String segment : path.split("\\.")) {
            if (!(current instanceof JSONObject)) {
                throw new IOException("search mapping traversed a non-object value");
            }
            JSONObject object = (JSONObject) current;
            if (!object.has(segment) || object.isNull(segment)) {
                throw new IOException("search mapping field is missing");
            }
            try {
                current = object.get(segment);
            } catch (org.json.JSONException error) {
                throw new IOException("search mapping field could not be read", error);
            }
        }
        return current;
    }

    private static String requiredJsonString(JSONObject item, String path) throws IOException {
        Object value = resolveJsonPath(item, path);
        if (!(value instanceof String) || ((String) value).trim().isEmpty()) {
            throw new IOException("required search result field is not a string");
        }
        return (String) value;
    }

    private static String optionalJsonString(JSONObject item, String path) throws IOException {
        try {
            Object value = resolveJsonPath(item, path);
            return value instanceof String ? (String) value : "";
        } catch (IOException missing) {
            return "";
        }
    }

    private static String normalizeSearchText(
            String value,
            int maxBytes,
            boolean required) throws IOException {
        StringBuilder collapsed = new StringBuilder();
        boolean pendingSpace = false;
        for (int offset = 0; offset < value.length(); ) {
            int codePoint = value.codePointAt(offset);
            offset += Character.charCount(codePoint);
            if (Character.isWhitespace(codePoint) || Character.isISOControl(codePoint)) {
                pendingSpace = collapsed.length() > 0;
                continue;
            }
            if (pendingSpace) {
                collapsed.append(' ');
                pendingSpace = false;
            }
            collapsed.appendCodePoint(codePoint);
        }

        String normalized = collapsed.toString().trim();
        if (required && normalized.isEmpty()) {
            throw new IOException("required search result text is empty");
        }
        if (normalized.isEmpty()) {
            return "";
        }

        StringBuilder bounded = new StringBuilder();
        int used = 0;
        for (int offset = 0; offset < normalized.length(); ) {
            int codePoint = normalized.codePointAt(offset);
            offset += Character.charCount(codePoint);
            String scalar = new String(Character.toChars(codePoint));
            int bytes = scalar.getBytes(StandardCharsets.UTF_8).length;
            if (used + bytes > maxBytes) {
                break;
            }
            bounded.appendCodePoint(codePoint);
            used += bytes;
        }
        if (required && bounded.length() == 0) {
            throw new IOException("required search result text exceeds normalization limit");
        }
        return bounded.toString();
    }

    private static byte[] putBlocking(
            String rawUrl,
            byte[] body,
            String expectedSha256) {
        HttpsURLConnection connection = null;
        try {
            if (body == null || body.length == 0 || body.length > MAX_BODY_BYTES) {
                throw new IOException("invalid upload body size");
            }
            String digest = sha256Hex(body);
            if (expectedSha256 == null
                    || !digest.equals(expectedSha256.toLowerCase(Locale.ROOT))) {
                throw new IOException("upload body hash mismatch");
            }

            URL target = validatePublicHttpsUrl(rawUrl);
            connection = (HttpsURLConnection) target.openConnection();
            connection.setInstanceFollowRedirects(false);
            connection.setConnectTimeout(CONNECT_TIMEOUT_MS);
            connection.setReadTimeout(READ_TIMEOUT_MS);
            connection.setRequestMethod("PUT");
            connection.setDoOutput(true);
            connection.setFixedLengthStreamingMode(body.length);
            connection.setRequestProperty("Content-Type", "application/octet-stream");
            connection.setRequestProperty("Accept", "application/json,text/plain,*/*;q=0.1");
            connection.setRequestProperty("User-Agent", "NTD97-Mobile/1");
            connection.setRequestProperty("X-NTD97-SHA256", digest);
            connection.connect();

            try (OutputStream output = connection.getOutputStream()) {
                output.write(body);
                output.flush();
            }

            int status = connection.getResponseCode();
            if (status >= 300 && status < 400) {
                throw new IOException("upload redirects are not allowed");
            }
            if (status < 200 || status >= 300) {
                throw new IOException("HTTP status " + status);
            }

            byte[] responseBody = new byte[0];
            long declared = connection.getContentLengthLong();
            if (declared > MAX_BODY_BYTES) {
                throw new IOException("upload response exceeds size limit");
            }
            InputStream input = connection.getInputStream();
            if (input != null) {
                try (InputStream response = input) {
                    responseBody = readBounded(response);
                }
            }
            String contentType = connection.getContentType();
            return encode(
                    status,
                    target.toExternalForm(),
                    contentType == null ? "" : contentType,
                    responseBody);
        } catch (Exception error) {
            return encodeError(safeMessage(error));
        } finally {
            if (connection != null) {
                connection.disconnect();
            }
        }
    }

    private static byte[] fetchBlocking(String rawUrl) {
        try {
            HttpResult result = fetchResponse(rawUrl);
            return encode(
                    result.status,
                    result.finalUrl,
                    result.contentType,
                    result.body);
        } catch (Exception error) {
            return encodeError(safeMessage(error));
        }
    }

    private static HttpResult fetchResponse(String rawUrl) throws IOException {
        URL current = validatePublicHttpsUrl(rawUrl);
        for (int redirect = 0; redirect <= MAX_REDIRECTS; redirect++) {
            HttpsURLConnection connection = (HttpsURLConnection) current.openConnection();
            connection.setInstanceFollowRedirects(false);
            connection.setConnectTimeout(CONNECT_TIMEOUT_MS);
            connection.setReadTimeout(READ_TIMEOUT_MS);
            connection.setRequestMethod("GET");
            connection.setRequestProperty(
                    "Accept",
                    "application/json,text/plain,text/html,application/xml;q=0.9,*/*;q=0.1");
            connection.setRequestProperty("User-Agent", "NTD97-Mobile/1");
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
                current = validatePublicHttpsUrl(new URL(current, location).toExternalForm());
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

            String contentType = connection.getContentType();
            byte[] body;
            try (InputStream input = connection.getInputStream()) {
                body = readBounded(input);
            } finally {
                connection.disconnect();
            }

            return new HttpResult(
                    status,
                    current.toExternalForm(),
                    contentType == null ? "" : contentType,
                    body);
        }
        throw new IOException("unreachable redirect state");
    }

    private static String searchSource(String finalUrl) throws IOException {
        URL url = validateHttpsSyntax(finalUrl);
        return "https://" + url.getHost().toLowerCase(Locale.ROOT);
    }

    private static String safeMessage(Throwable error) {
        String message = error.getMessage();
        return message == null || message.trim().isEmpty()
                ? error.getClass().getSimpleName()
                : message;
    }

    private static URL validateHttpsSyntax(String rawUrl) throws IOException {
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
        return url;
    }

    static URL validatePublicHttpsUrl(String rawUrl) throws IOException {
        URL url = validateHttpsSyntax(rawUrl);
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

    private static String sha256Hex(byte[] bytes) throws Exception {
        byte[] digest = MessageDigest.getInstance("SHA-256").digest(bytes);
        StringBuilder out = new StringBuilder(digest.length * 2);
        for (byte value : digest) {
            out.append(String.format(Locale.ROOT, "%02x", value & 0xff));
        }
        return out.toString();
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

    private static byte[] encodeSearch(
            int status,
            String source,
            List<SearchItem> items) {
        byte[] sourceBytes = source.getBytes(StandardCharsets.UTF_8);
        int capacity = 1 + 1 + 4 + 4 + sourceBytes.length + 4;
        List<byte[][]> encodedItems = new ArrayList<>(items.size());
        for (SearchItem item : items) {
            byte[] title = item.title.getBytes(StandardCharsets.UTF_8);
            byte[] url = item.url.getBytes(StandardCharsets.UTF_8);
            byte[] snippet = item.snippet.getBytes(StandardCharsets.UTF_8);
            encodedItems.add(new byte[][] {title, url, snippet});
            capacity += 4 + title.length + 4 + url.length + 4 + snippet.length;
        }

        ByteBuffer buffer = ByteBuffer.allocate(capacity).order(ByteOrder.LITTLE_ENDIAN);
        buffer.put(PROTOCOL_VERSION);
        buffer.put((byte) 1);
        buffer.putInt(status);
        putBytes(buffer, sourceBytes);
        buffer.putInt(items.size());
        for (byte[][] item : encodedItems) {
            putBytes(buffer, item[0]);
            putBytes(buffer, item[1]);
            putBytes(buffer, item[2]);
        }
        return buffer.array();
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

    private static void clearSearchPreferences(SharedPreferences preferences) {
        preferences.edit()
                .remove(SEARCH_TEMPLATE_KEY)
                .remove(SEARCH_RESULTS_PATH_KEY)
                .remove(SEARCH_TITLE_PATH_KEY)
                .remove(SEARCH_URL_PATH_KEY)
                .remove(SEARCH_SNIPPET_PATH_KEY)
                .apply();
    }

    private static String value(SharedPreferences preferences, String key) {
        String value = preferences.getString(key, "");
        return value == null ? "" : value.trim();
    }

    private static void putBytes(ByteBuffer buffer, byte[] bytes) {
        buffer.putInt(bytes.length);
        buffer.put(bytes);
    }
}
