package ai.ntd97.mobile;

import android.content.Context;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.util.Xml;

import org.json.JSONArray;
import org.json.JSONException;
import org.xmlpull.v1.XmlPullParser;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.InetAddress;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Locale;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import javax.net.ssl.HttpsURLConnection;

final class NtdWebPlatform {
    private static final String OPENSEARCH_METADATA =
            "ai.ntd97.web_search.opensearch_description";
    private static final int CONNECT_TIMEOUT_MS = 10_000;
    private static final int READ_TIMEOUT_MS = 10_000;
    private static final int MAX_BODY_BYTES = 512 * 1024;
    private static final int MAX_SEARCH_QUERY_CHARS = 1024;
    private static final int MAX_SEARCH_RESULTS = 20;
    private static final int MAX_REDIRECTS = 3;
    private static final int PLATFORM_TIMEOUT_SECONDS = 45;
    private static final byte PROTOCOL_VERSION = 1;
    private static final ExecutorService NETWORK_EXECUTOR =
            Executors.newSingleThreadExecutor(runnable -> {
                Thread thread = new Thread(runnable, "ntd97-https");
                thread.setDaemon(true);
                return thread;
            });

    private static volatile Context appContext;

    private NtdWebPlatform() {}

    static void initialize(Context context) {
        if (context != null) {
            appContext = context.getApplicationContext();
        }
    }

    static byte[] fetch(String rawUrl) {
        return runNetwork(() -> encodeFetch(fetchRaw(rawUrl)));
    }

    static byte[] search(String query, int maxResults) {
        return runNetwork(() -> searchBlocking(query, maxResults));
    }

    private static byte[] runNetwork(NetworkOperation operation) {
        Future<byte[]> future = NETWORK_EXECUTOR.submit(operation::run);
        try {
            return future.get(PLATFORM_TIMEOUT_SECONDS, TimeUnit.SECONDS);
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            future.cancel(true);
            return encodeError("HTTPS operation interrupted");
        } catch (TimeoutException error) {
            future.cancel(true);
            return encodeError("HTTPS operation timed out");
        } catch (ExecutionException error) {
            Throwable cause = error.getCause();
            return encodeError(cause == null ? "HTTPS operation failed" : safeMessage(cause));
        }
    }

    private static byte[] searchBlocking(String query, int maxResults) {
        try {
            if (query == null
                    || query.trim().isEmpty()
                    || query.length() > MAX_SEARCH_QUERY_CHARS) {
                throw new IOException("invalid search query");
            }
            if (maxResults <= 0 || maxResults > MAX_SEARCH_RESULTS) {
                throw new IOException("invalid search result limit");
            }

            String descriptorUrl = configuredOpenSearchDescription();
            HttpsResult descriptor = fetchRaw(descriptorUrl);
            String template = parseSuggestionTemplate(descriptor.body);
            String searchUrl = expandOpenSearchTemplate(template, query.trim(), maxResults);
            HttpsResult response = fetchRaw(searchUrl);
            return encodeSearchResults(
                    descriptorUrl,
                    response.finalUrl,
                    parseSuggestionResults(response.body, maxResults));
        } catch (Exception error) {
            return encodeError(safeMessage(error));
        }
    }

    private static String configuredOpenSearchDescription() throws IOException {
        Context context = appContext;
        if (context == null) {
            throw new IOException("web platform is not initialized");
        }
        try {
            ApplicationInfo info = context.getPackageManager().getApplicationInfo(
                    context.getPackageName(),
                    PackageManager.GET_META_DATA);
            String configured = info.metaData == null
                    ? null
                    : info.metaData.getString(OPENSEARCH_METADATA);
            if (configured == null || configured.trim().isEmpty()) {
                throw new IOException("OpenSearch provider is not configured");
            }
            return validateUrl(configured.trim()).toExternalForm();
        } catch (PackageManager.NameNotFoundException error) {
            throw new IOException("application metadata unavailable", error);
        }
    }

    private static String parseSuggestionTemplate(byte[] descriptorBytes) throws Exception {
        if (descriptorBytes.length == 0) {
            throw new IOException("empty OpenSearch description");
        }
        XmlPullParser parser = Xml.newPullParser();
        parser.setFeature(XmlPullParser.FEATURE_PROCESS_NAMESPACES, true);
        parser.setInput(
                new java.io.ByteArrayInputStream(descriptorBytes),
                StandardCharsets.UTF_8.name());

        String fallback = null;
        int event;
        while ((event = parser.next()) != XmlPullParser.END_DOCUMENT) {
            if (event != XmlPullParser.START_TAG || !"Url".equals(parser.getName())) {
                continue;
            }
            String method = parser.getAttributeValue(null, "method");
            if (method != null && !"get".equalsIgnoreCase(method.trim())) {
                continue;
            }
            String template = parser.getAttributeValue(null, "template");
            if (template == null || !template.contains("{searchTerms}")) {
                continue;
            }
            String type = parser.getAttributeValue(null, "type");
            if (type == null) {
                continue;
            }
            String normalizedType = type.trim().toLowerCase(Locale.ROOT);
            if ("application/x-suggestions+json".equals(normalizedType)) {
                return template;
            }
            if ("application/json".equals(normalizedType) && fallback == null) {
                fallback = template;
            }
        }
        if (fallback != null) {
            return fallback;
        }
        throw new IOException("OpenSearch description has no JSON suggestion template");
    }

    private static String expandOpenSearchTemplate(
            String template,
            String query,
            int maxResults) throws IOException {
        String expanded = template
                .replace("{searchTerms}", Uri.encode(query))
                .replace("{count}", Integer.toString(maxResults))
                .replace("{count?}", Integer.toString(maxResults))
                .replace("{startIndex}", "0")
                .replace("{startIndex?}", "0")
                .replace("{startPage}", "1")
                .replace("{startPage?}", "1")
                .replace("{language}", Uri.encode(Locale.getDefault().toLanguageTag()))
                .replace("{language?}", Uri.encode(Locale.getDefault().toLanguageTag()))
                .replace("{inputEncoding}", "UTF-8")
                .replace("{inputEncoding?}", "UTF-8")
                .replace("{outputEncoding}", "UTF-8")
                .replace("{outputEncoding?}", "UTF-8");
        if (expanded.indexOf('{') >= 0 || expanded.indexOf('}') >= 0) {
            throw new IOException("OpenSearch template contains unsupported parameters");
        }
        return validateUrl(expanded).toExternalForm();
    }

    private static SearchItem[] parseSuggestionResults(byte[] bytes, int maxResults)
            throws IOException {
        try {
            JSONArray root = new JSONArray(new String(bytes, StandardCharsets.UTF_8));
            if (root.length() < 4) {
                throw new IOException("OpenSearch JSON response is incomplete");
            }
            JSONArray titles = root.optJSONArray(1);
            JSONArray descriptions = root.optJSONArray(2);
            JSONArray urls = root.optJSONArray(3);
            if (titles == null || urls == null) {
                throw new IOException("OpenSearch JSON response lacks title/URL arrays");
            }

            int count = Math.min(maxResults, Math.min(titles.length(), urls.length()));
            SearchItem[] items = new SearchItem[count];
            int accepted = 0;
            for (int index = 0; index < count; index++) {
                String title = titles.optString(index, "").trim();
                String rawUrl = urls.optString(index, "").trim();
                if (title.isEmpty() || rawUrl.isEmpty()) {
                    continue;
                }
                URL resultUrl;
                try {
                    resultUrl = validateUrl(rawUrl);
                } catch (IOException ignored) {
                    continue;
                }
                String description = descriptions == null
                        ? ""
                        : descriptions.optString(index, "").trim();
                items[accepted++] = new SearchItem(
                        title,
                        resultUrl.toExternalForm(),
                        description);
            }
            if (accepted == 0) {
                throw new IOException("OpenSearch returned no valid public HTTPS results");
            }
            if (accepted == items.length) {
                return items;
            }
            SearchItem[] compact = new SearchItem[accepted];
            System.arraycopy(items, 0, compact, 0, accepted);
            return compact;
        } catch (JSONException error) {
            throw new IOException("invalid OpenSearch JSON response", error);
        }
    }

    private static HttpsResult fetchRaw(String rawUrl) throws IOException {
        URL current = validateUrl(rawUrl);
        for (int redirect = 0; redirect <= MAX_REDIRECTS; redirect++) {
            HttpsURLConnection connection = (HttpsURLConnection) current.openConnection();
            connection.setInstanceFollowRedirects(false);
            connection.setConnectTimeout(CONNECT_TIMEOUT_MS);
            connection.setReadTimeout(READ_TIMEOUT_MS);
            connection.setRequestMethod("GET");
            connection.setRequestProperty(
                    "Accept",
                    "application/opensearchdescription+xml,application/json,text/plain,text/html,application/xml;q=0.9,*/*;q=0.1");
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

            String contentType = connection.getContentType();
            byte[] body;
            try (InputStream input = connection.getInputStream()) {
                body = readBounded(input);
            } finally {
                connection.disconnect();
            }
            return new HttpsResult(
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

    private static byte[] encodeFetch(HttpsResult result) {
        byte[] url = result.finalUrl.getBytes(StandardCharsets.UTF_8);
        byte[] type = result.contentType.getBytes(StandardCharsets.UTF_8);
        ByteBuffer buffer = ByteBuffer
                .allocate(1 + 1 + 4 + 4 + url.length + 4 + type.length + 4 + result.body.length)
                .order(ByteOrder.LITTLE_ENDIAN);
        buffer.put(PROTOCOL_VERSION);
        buffer.put((byte) 1);
        buffer.putInt(result.status);
        putBytes(buffer, url);
        putBytes(buffer, type);
        putBytes(buffer, result.body);
        return buffer.array();
    }

    private static byte[] encodeSearchResults(
            String descriptorUrl,
            String responseUrl,
            SearchItem[] items) {
        byte[] descriptor = descriptorUrl.getBytes(StandardCharsets.UTF_8);
        byte[] response = responseUrl.getBytes(StandardCharsets.UTF_8);
        int size = 1 + 1 + 4 + descriptor.length + 4 + response.length + 4;
        byte[][] titles = new byte[items.length][];
        byte[][] urls = new byte[items.length][];
        byte[][] descriptions = new byte[items.length][];
        for (int index = 0; index < items.length; index++) {
            titles[index] = items[index].title.getBytes(StandardCharsets.UTF_8);
            urls[index] = items[index].url.getBytes(StandardCharsets.UTF_8);
            descriptions[index] = items[index].description.getBytes(StandardCharsets.UTF_8);
            size += 12 + titles[index].length + urls[index].length + descriptions[index].length;
        }

        ByteBuffer buffer = ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN);
        buffer.put(PROTOCOL_VERSION);
        buffer.put((byte) 1);
        putBytes(buffer, descriptor);
        putBytes(buffer, response);
        buffer.putInt(items.length);
        for (int index = 0; index < items.length; index++) {
            putBytes(buffer, titles[index]);
            putBytes(buffer, urls[index]);
            putBytes(buffer, descriptions[index]);
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

    private interface NetworkOperation {
        byte[] run() throws Exception;
    }

    private static final class HttpsResult {
        final int status;
        final String finalUrl;
        final String contentType;
        final byte[] body;

        HttpsResult(int status, String finalUrl, String contentType, byte[] body) {
            this.status = status;
            this.finalUrl = finalUrl;
            this.contentType = contentType;
            this.body = body;
        }
    }

    private static final class SearchItem {
        final String title;
        final String url;
        final String description;

        SearchItem(String title, String url, String description) {
            this.title = title;
            this.url = url;
            this.description = description;
        }
    }
}
