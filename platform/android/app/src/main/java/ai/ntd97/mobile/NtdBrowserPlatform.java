package ai.ntd97.mobile;

import android.content.Context;
import android.content.SharedPreferences;
import android.net.http.SslError;
import android.os.Handler;
import android.os.Looper;
import android.webkit.CookieManager;
import android.webkit.SslErrorHandler;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.webkit.WebResourceResponse;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Collections;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

final class NtdBrowserPlatform {
    private static final byte PROTOCOL_VERSION = 1;
    private static final int OPERATION_TIMEOUT_SECONDS = 20;
    private static final int INTERACTION_SETTLE_MS = 600;
    private static final int MAX_TARGET_BYTES = 1024;
    private static final int MAX_VALUE_BYTES = 4096;
    private static final int MAX_OBSERVED_TEXT_CHARS = 64 * 1024;
    private static final String PREFS_NAME = "ntd97-browser";
    private static final String LAST_VERIFIED_URL_KEY = "last-verified-url";
    private static final String LAST_VERIFIED_ORIGIN_KEY = "last-verified-origin";
    private static final String SESSION_TARGET = "session";
    private static final Handler MAIN = new Handler(Looper.getMainLooper());
    private static final AtomicBoolean BUSY = new AtomicBoolean();
    private static final AtomicLong RECEIPT_COUNTER = new AtomicLong();
    private static final ExecutorService URL_VALIDATION_EXECUTOR =
            Executors.newSingleThreadExecutor(runnable -> {
                Thread thread = new Thread(runnable, "ntd97-browser-url");
                thread.setDaemon(true);
                return thread;
            });

    private static volatile Context appContext;
    private static WebView webView;

    private NtdBrowserPlatform() {}

    static void initialize(Context context) {
        appContext = context.getApplicationContext();
    }

    static byte[] observe(String rawTarget) {
        if (Looper.myLooper() == Looper.getMainLooper()) {
            return encodeError("browser observe cannot block the Android main thread");
        }
        if (!BUSY.compareAndSet(false, true)) {
            return encodeError("browser session is busy");
        }
        try {
            Context context = requireContext();
            String requestedTarget = normalizeObserveTarget(rawTarget);
            URL target = resolveObserveTarget(context, requestedTarget);
            String expectedOrigin = browserOrigin(target);
            AtomicReference<byte[]> result = new AtomicReference<>();
            CountDownLatch latch = new CountDownLatch(1);
            MAIN.post(() -> performObserve(
                    context,
                    requestedTarget,
                    target.toExternalForm(),
                    expectedOrigin,
                    result,
                    latch));
            if (!latch.await(OPERATION_TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
                return encodeError("browser observe timed out");
            }
            byte[] encoded = result.get();
            return encoded == null ? encodeError("browser observe produced no result") : encoded;
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            return encodeError("browser observe interrupted");
        } catch (Exception error) {
            return encodeError(safeMessage(error));
        } finally {
            BUSY.set(false);
        }
    }

    static byte[] interact(String target, String operation, String value) {
        if (Looper.myLooper() == Looper.getMainLooper()) {
            return encodeError("browser interaction cannot block the Android main thread");
        }
        if (!BUSY.compareAndSet(false, true)) {
            return encodeError("browser session is busy");
        }
        try {
            validateInteraction(target, operation, value);
            Context context = requireContext();
            if (!"navigate".equals(operation) && webView == null) {
                return encodeError("browser session is not initialized");
            }
            AtomicReference<byte[]> result = new AtomicReference<>();
            CountDownLatch latch = new CountDownLatch(1);
            MAIN.post(() -> performInteract(
                    context,
                    target,
                    operation,
                    value,
                    result,
                    latch));
            if (!latch.await(OPERATION_TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
                return encodeError("browser interaction timed out");
            }
            byte[] encoded = result.get();
            return encoded == null ? encodeError("browser interaction produced no result") : encoded;
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            return encodeError("browser interaction interrupted");
        } catch (Exception error) {
            return encodeError(safeMessage(error));
        } finally {
            BUSY.set(false);
        }
    }

    private static void performObserve(
            Context context,
            String requestedTarget,
            String target,
            String expectedOrigin,
            AtomicReference<byte[]> result,
            CountDownLatch latch) {
        try {
            WebView view = ensureWebView(context);
            AtomicBoolean completed = new AtomicBoolean();
            view.setWebViewClient(new GuardedClient(result, latch, completed, expectedOrigin) {
                @Override
                public void onPageFinished(WebView finishedView, String url) {
                    if (completed.get()) {
                        return;
                    }
                    try {
                        URL finalUrl = validatePublicHttps(url);
                        if (!browserOrigin(finalUrl).equals(expectedOrigin)) {
                            finish(
                                    completed,
                                    result,
                                    latch,
                                    encodeError("browser main-frame origin changed during observation"));
                            return;
                        }
                        String script = "(function(){"
                                + "var t=document.body?document.body.innerText:'';"
                                + "if(t.length>" + MAX_OBSERVED_TEXT_CHARS + "){t=t.slice(0,"
                                + MAX_OBSERVED_TEXT_CHARS + ");}"
                                + "return location.href+'\\u001f'+document.title+'\\u001f'+t;"
                                + "})()";
                        finishedView.evaluateJavascript(script, raw -> {
                            if (completed.get()) {
                                return;
                            }
                            try {
                                String decoded = decodeJsString(raw);
                                String[] fields = decoded.split("\\u001f", 3);
                                if (fields.length != 3) {
                                    finish(completed, result, latch,
                                            encodeError("browser observe returned invalid DOM evidence"));
                                    return;
                                }
                                URL observed = validatePublicHttps(fields[0]);
                                if (!observed.toExternalForm().equals(finalUrl.toExternalForm())) {
                                    finish(completed, result, latch,
                                            encodeError("browser URL changed during observation"));
                                    return;
                                }
                                persistVerifiedSessionUrl(context, observed);
                                String receipt = browserReceipt(
                                        "observe",
                                        requestedTarget,
                                        "");
                                finish(
                                        completed,
                                        result,
                                        latch,
                                        encodeSuccess(
                                                "receipt=" + receipt
                                                        + "\noperation=observe"
                                                        + "\nsession="
                                                        + (SESSION_TARGET.equals(requestedTarget)
                                                                ? "resumed"
                                                                : "direct")
                                                        + "\nurl=" + fields[0]
                                                        + "\ntitle=" + sanitizeLine(fields[1])
                                                        + "\ntext=" + fields[2]));
                            } catch (Exception error) {
                                finish(completed, result, latch, encodeError(safeMessage(error)));
                            }
                        });
                    } catch (Exception error) {
                        finish(completed, result, latch, encodeError(safeMessage(error)));
                    }
                }
            });
            view.stopLoading();
            view.loadUrl(target);
        } catch (Exception error) {
            result.set(encodeError(safeMessage(error)));
            latch.countDown();
        }
    }

    private static void performInteract(
            Context context,
            String target,
            String operation,
            String value,
            AtomicReference<byte[]> result,
            CountDownLatch latch) {
        AtomicBoolean completed = new AtomicBoolean();
        try {
            if ("navigate".equals(operation)) {
                URL destination = validatePublicHttps(target);
                String expectedOrigin = browserOrigin(destination);
                WebView view = ensureWebView(context);
                view.setWebViewClient(new GuardedClient(
                        result, latch, completed, expectedOrigin) {
                    @Override
                    public void onPageFinished(WebView finishedView, String url) {
                        if (completed.get()) {
                            return;
                        }
                        try {
                            URL finalUrl = validatePublicHttps(url);
                            if (!browserOrigin(finalUrl).equals(expectedOrigin)) {
                                finish(
                                        completed,
                                        result,
                                        latch,
                                        encodeError("browser navigation changed origin"));
                                return;
                            }
                            persistVerifiedSessionUrl(context, finalUrl);
                            long receiptId = RECEIPT_COUNTER.incrementAndGet();
                            String receipt = browserReceipt(
                                    operation,
                                    target,
                                    value == null ? "" : value,
                                    receiptId);
                            finish(
                                    completed,
                                    result,
                                    latch,
                                    encodeSuccess(
                                            "receipt=" + receipt
                                                    + "\noperation=" + operation
                                                    + "\nurl=" + finalUrl.toExternalForm()
                                                    + "\ntitle=" + sanitizeLine(finishedView.getTitle())));
                        } catch (Exception error) {
                            finish(completed, result, latch, encodeError(safeMessage(error)));
                        }
                    }
                });
                view.stopLoading();
                view.loadUrl(destination.toExternalForm());
                return;
            }

            WebView view = webView;
            if (view == null) {
                finish(completed, result, latch, encodeError("browser session disappeared"));
                return;
            }
            String current = view.getUrl();
            if (current == null) {
                finish(completed, result, latch, encodeError("browser has no active page"));
                return;
            }
            URL currentUrl = validatePublicHttps(current);
            String expectedOrigin = browserOrigin(currentUrl);
            if ("submit".equals(operation)) {
                view.setWebViewClient(new GuardedClient(
                        result, latch, completed, expectedOrigin) {
                    @Override
                    public void onPageFinished(WebView finishedView, String url) {
                        if (completed.get()) {
                            return;
                        }
                        try {
                            URL finalUrl = validatePublicHttps(url);
                            if (!browserOrigin(finalUrl).equals(expectedOrigin)) {
                                finish(
                                        completed,
                                        result,
                                        latch,
                                        encodeError("browser submit changed origin"));
                                return;
                            }
                            persistVerifiedSessionUrl(context, finalUrl);
                            long receiptId = RECEIPT_COUNTER.incrementAndGet();
                            String receipt = browserReceipt(
                                    operation,
                                    target,
                                    value == null ? "" : value,
                                    receiptId);
                            finish(
                                    completed,
                                    result,
                                    latch,
                                    encodeSuccess(
                                            "receipt=" + receipt
                                                    + "\noperation=" + operation
                                                    + "\ntarget=" + sanitizeLine(target)
                                                    + "\nurl=" + finalUrl.toExternalForm()
                                                    + "\ntitle=" + sanitizeLine(finishedView.getTitle())
                                                    + "\ntag=FORM"));
                        } catch (Exception error) {
                            finish(completed, result, latch, encodeError(safeMessage(error)));
                        }
                    }
                });
            } else {
                view.setWebViewClient(new GuardedClient(
                        result, latch, completed, expectedOrigin));
            }

            String selector = JSONObject.quote(target);
            String uniquePrefix = "try{var m=document.querySelectorAll(" + selector + ");"
                    + "if(m.length===0){return 'ERR\\u001fmissing-target';}"
                    + "if(m.length!==1){return 'ERR\\u001fambiguous-target';}"
                    + "var e=m[0];var tag=e.tagName||'';";
            String script;
            if ("click".equals(operation)) {
                script = "(function(){" + uniquePrefix
                        + "if(typeof e.click!=='function'){return 'ERR\\u001fnot-clickable';}"
                        + "e.click();return 'OK\\u001f'+tag;"
                        + "}catch(x){return 'ERR\\u001fscript-error';}})()";
            } else if ("set_value".equals(operation)) {
                String quotedValue = JSONObject.quote(value);
                script = "(function(){" + uniquePrefix
                        + "if(!('value' in e)){return 'ERR\\u001fnot-value-target';}"
                        + "e.value=" + quotedValue + ";"
                        + "e.dispatchEvent(new Event('input',{bubbles:true}));"
                        + "e.dispatchEvent(new Event('change',{bubbles:true}));"
                        + "return 'OK\\u001f'+tag+'\\u001f'+String(e.value);"
                        + "}catch(x){return 'ERR\\u001fscript-error';}})()";
            } else if ("submit".equals(operation)) {
                script = "(function(){" + uniquePrefix
                        + "if(tag.toUpperCase()!=='FORM'){return 'ERR\\u001fnot-form';}"
                        + "setTimeout(function(){"
                        + "if(typeof e.requestSubmit==='function'){e.requestSubmit();}else{e.submit();}"
                        + "},0);return 'OK\\u001f'+tag;"
                        + "}catch(x){return 'ERR\\u001fscript-error';}})()";
            } else {
                finish(completed, result, latch, encodeError("unsupported browser operation"));
                return;
            }

            view.evaluateJavascript(script, raw -> {
                if (completed.get()) {
                    return;
                }
                try {
                    String decoded = decodeJsString(raw);
                    String[] fields = decoded.split("\\u001f", 3);
                    if (fields.length < 2 || !"OK".equals(fields[0])) {
                        String reason = fields.length >= 2 ? fields[1] : "invalid-result";
                        finish(completed, result, latch,
                                encodeError("browser interaction rejected: " + sanitizeLine(reason)));
                        return;
                    }
                    if ("set_value".equals(operation)
                            && (fields.length != 3 || value == null || !value.equals(fields[2]))) {
                        finish(
                                completed,
                                result,
                                latch,
                                encodeError("browser set_value read-back mismatch"));
                        return;
                    }
                    Runnable finishInteraction = () -> {
                        try {
                            String finalUrl = view.getUrl();
                            if (finalUrl == null) {
                                throw new IOException("browser interaction lost active URL");
                            }
                            URL verifiedFinalUrl = validatePublicHttps(finalUrl);
                            if (!browserOrigin(verifiedFinalUrl).equals(expectedOrigin)) {
                                throw new IOException("browser interaction changed origin");
                            }
                            persistVerifiedSessionUrl(context, verifiedFinalUrl);
                            long receiptId = RECEIPT_COUNTER.incrementAndGet();
                            String receipt = browserReceipt(
                                    operation,
                                    target,
                                    value == null ? "" : value,
                                    receiptId);
                            StringBuilder message = new StringBuilder()
                                    .append("receipt=").append(receipt)
                                    .append("\noperation=").append(operation)
                                    .append("\ntarget=").append(sanitizeLine(target))
                                    .append("\nurl=").append(verifiedFinalUrl.toExternalForm())
                                    .append("\ntitle=").append(sanitizeLine(view.getTitle()))
                                    .append("\ntag=").append(sanitizeLine(fields[1]));
                            if ("set_value".equals(operation)) {
                                message.append("\nvalue_sha256=")
                                        .append(sha256Hex(value.getBytes(StandardCharsets.UTF_8)));
                            }
                            finish(
                                    completed,
                                    result,
                                    latch,
                                    encodeSuccess(message.toString()));
                        } catch (Exception error) {
                            finish(completed, result, latch, encodeError(safeMessage(error)));
                        }
                    };
                    if ("submit".equals(operation)) {
                        return;
                    }
                    if ("click".equals(operation)) {
                        MAIN.postDelayed(finishInteraction, INTERACTION_SETTLE_MS);
                    } else {
                        finishInteraction.run();
                    }
                } catch (Exception error) {
                    finish(completed, result, latch, encodeError(safeMessage(error)));
                }
            });
        } catch (Exception error) {
            finish(completed, result, latch, encodeError(safeMessage(error)));
        }
    }

    private static WebView ensureWebView(Context context) {
        if (webView != null) {
            return webView;
        }
        WebView view = new WebView(context);
        WebSettings settings = view.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setJavaScriptCanOpenWindowsAutomatically(false);
        settings.setSupportMultipleWindows(false);
        settings.setAllowFileAccess(false);
        settings.setAllowContentAccess(false);
        settings.setAllowFileAccessFromFileURLs(false);
        settings.setAllowUniversalAccessFromFileURLs(false);
        settings.setDomStorageEnabled(false);
        settings.setGeolocationEnabled(false);
        settings.setMediaPlaybackRequiresUserGesture(true);
        settings.setMixedContentMode(WebSettings.MIXED_CONTENT_NEVER_ALLOW);
        settings.setSafeBrowsingEnabled(true);
        CookieManager.getInstance().setAcceptThirdPartyCookies(view, false);
        webView = view;
        return view;
    }

    private static void validateInteraction(String target, String operation, String value)
            throws IOException {
        if (target == null || target.trim().isEmpty() || !target.equals(target.trim())) {
            throw new IOException("empty or untrimmed browser target");
        }
        if (target.getBytes(StandardCharsets.UTF_8).length > MAX_TARGET_BYTES
                || containsControl(target)) {
            throw new IOException("invalid browser target");
        }
        if (!"click".equals(operation)
                && !"set_value".equals(operation)
                && !"submit".equals(operation)
                && !"navigate".equals(operation)) {
            throw new IOException("unsupported browser operation");
        }

        boolean hasValue = value != null && !value.isEmpty();
        if (("click".equals(operation)
                        || "submit".equals(operation)
                        || "navigate".equals(operation))
                && hasValue) {
            throw new IOException("browser operation does not accept a value");
        }
        if ("navigate".equals(operation)) {
            validatePublicHttps(target);
            return;
        }
        if ("set_value".equals(operation)) {
            if (!hasValue) {
                throw new IOException("browser set_value requires a value");
            }
            if (value.getBytes(StandardCharsets.UTF_8).length > MAX_VALUE_BYTES
                    || containsControl(value)) {
                throw new IOException("invalid browser value");
            }
        }
    }

    private static String normalizeObserveTarget(String rawTarget) throws IOException {
        if (rawTarget == null
                || rawTarget.isEmpty()
                || !rawTarget.equals(rawTarget.trim())
                || rawTarget.getBytes(StandardCharsets.UTF_8).length > MAX_TARGET_BYTES
                || containsControl(rawTarget)) {
            throw new IOException("invalid browser observe target");
        }
        if (!SESSION_TARGET.equals(rawTarget)) {
            validatePublicHttps(rawTarget);
        }
        return rawTarget;
    }

    private static URL resolveObserveTarget(Context context, String requestedTarget)
            throws IOException {
        if (!SESSION_TARGET.equals(requestedTarget)) {
            return validatePublicHttps(requestedTarget);
        }
        SharedPreferences preferences =
                context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        String stored = preferences.getString(LAST_VERIFIED_URL_KEY, "");
        String storedOrigin = preferences.getString(LAST_VERIFIED_ORIGIN_KEY, "");
        if (stored == null || stored.isEmpty() || storedOrigin == null || storedOrigin.isEmpty()) {
            throw new IOException("browser session has no persisted verified location");
        }
        try {
            URL verified = validatePublicHttps(stored);
            if (!browserOrigin(verified).equals(storedOrigin)) {
                throw new IOException("persisted browser session origin mismatch");
            }
            return verified;
        } catch (IOException error) {
            preferences.edit()
                    .remove(LAST_VERIFIED_URL_KEY)
                    .remove(LAST_VERIFIED_ORIGIN_KEY)
                    .commit();
            throw new IOException("persisted browser session location is invalid", error);
        }
    }

    private static void persistVerifiedSessionUrl(Context context, URL url) throws IOException {
        URL verified = validatePublicHttps(url.toExternalForm());
        String origin = browserOrigin(verified);
        boolean committed = context
                .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                .edit()
                .putString(LAST_VERIFIED_URL_KEY, verified.toExternalForm())
                .putString(LAST_VERIFIED_ORIGIN_KEY, origin)
                .commit();
        if (!committed) {
            throw new IOException("browser session location persistence failed");
        }
    }

    private static String browserOrigin(URL url) throws IOException {
        if (!"https".equalsIgnoreCase(url.getProtocol())
                || url.getHost() == null
                || url.getHost().isEmpty()) {
            throw new IOException("browser origin is not HTTPS");
        }
        int port = url.getPort() == -1 ? url.getDefaultPort() : url.getPort();
        if (port <= 0) {
            port = 443;
        }
        return "https://" + url.getHost().toLowerCase(java.util.Locale.ROOT) + ":" + port;
    }

    private static String browserReceipt(
            String operation,
            String target,
            String value) throws Exception {
        return browserReceipt(
                operation,
                target,
                value,
                RECEIPT_COUNTER.incrementAndGet());
    }

    private static String browserReceipt(
            String operation,
            String target,
            String value,
            long receiptId) throws Exception {
        String canonical = operation + "\n" + target + "\n" + value;
        return "android-webview:"
                + operation
                + ":"
                + receiptId
                + ":"
                + sha256Hex(canonical.getBytes(StandardCharsets.UTF_8));
    }

    private static String sha256Hex(byte[] bytes) throws Exception {
        byte[] digest = MessageDigest.getInstance("SHA-256").digest(bytes);
        StringBuilder encoded = new StringBuilder(digest.length * 2);
        for (byte value : digest) {
            encoded.append(String.format(java.util.Locale.ROOT, "%02x", value & 0xff));
        }
        return encoded.toString();
    }

    static boolean dropInMemorySessionForTest() {
        if (!BuildConfig.DEBUG) {
            return false;
        }
        if (Looper.myLooper() == Looper.getMainLooper()) {
            destroyInMemoryWebView();
            return true;
        }
        CountDownLatch latch = new CountDownLatch(1);
        MAIN.post(() -> {
            destroyInMemoryWebView();
            latch.countDown();
        });
        try {
            return latch.await(3, TimeUnit.SECONDS);
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            return false;
        }
    }

    private static void destroyInMemoryWebView() {
        WebView view = webView;
        webView = null;
        if (view != null) {
            view.stopLoading();
            view.destroy();
        }
    }

    private static boolean containsControl(String value) {
        for (int i = 0; i < value.length(); i++) {
            char ch = value.charAt(i);
            if (ch == '\r' || ch == '\n' || ch == '\t' || ch == '\0') {
                return true;
            }
        }
        return false;
    }

    private static Context requireContext() throws IOException {
        Context context = appContext;
        if (context == null) {
            throw new IOException("browser platform is not initialized");
        }
        return context;
    }

    private static URL validatePublicHttps(String rawUrl) throws IOException {
        if (Looper.myLooper() != Looper.getMainLooper()) {
            return NtdWebPlatform.validatePublicHttpsUrl(rawUrl);
        }
        Future<URL> future = URL_VALIDATION_EXECUTOR.submit(
                () -> NtdWebPlatform.validatePublicHttpsUrl(rawUrl));
        try {
            return future.get(3, TimeUnit.SECONDS);
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            future.cancel(true);
            throw new IOException("browser URL validation interrupted", error);
        } catch (TimeoutException error) {
            future.cancel(true);
            throw new IOException("browser URL validation timed out", error);
        } catch (ExecutionException error) {
            Throwable cause = error.getCause();
            if (cause instanceof IOException) {
                throw (IOException) cause;
            }
            throw new IOException("browser URL validation failed", cause);
        }
    }

    private static String decodeJsString(String raw) throws Exception {
        if (raw == null || "null".equals(raw)) {
            throw new IOException("browser script returned no value");
        }
        return new JSONArray("[" + raw + "]").getString(0);
    }

    private static String sanitizeLine(String value) {
        if (value == null) {
            return "";
        }
        return value.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ');
    }

    private static WebResourceResponse blockedResponse() {
        return new WebResourceResponse(
                "text/plain",
                "UTF-8",
                451,
                "Blocked by NTD97",
                Collections.emptyMap(),
                new ByteArrayInputStream(new byte[0]));
    }

    private static byte[] encodeSuccess(String message) {
        return encode((byte) 1, message);
    }

    private static byte[] encodeError(String message) {
        return encode((byte) 0, message);
    }

    private static byte[] encode(byte status, String message) {
        byte[] bytes = message.getBytes(StandardCharsets.UTF_8);
        ByteBuffer buffer = ByteBuffer
                .allocate(1 + 1 + 4 + bytes.length)
                .order(ByteOrder.LITTLE_ENDIAN);
        buffer.put(PROTOCOL_VERSION);
        buffer.put(status);
        buffer.putInt(bytes.length);
        buffer.put(bytes);
        return buffer.array();
    }

    private static void finish(
            AtomicBoolean completed,
            AtomicReference<byte[]> result,
            CountDownLatch latch,
            byte[] encoded) {
        if (completed.compareAndSet(false, true)) {
            result.set(encoded);
            latch.countDown();
        }
    }

    private static String safeMessage(Throwable error) {
        String message = error.getMessage();
        return message == null || message.trim().isEmpty()
                ? error.getClass().getSimpleName()
                : sanitizeLine(message);
    }

    private static class GuardedClient extends WebViewClient {
        private final AtomicReference<byte[]> result;
        private final CountDownLatch latch;
        private final AtomicBoolean completed;
        private final String expectedMainFrameOrigin;

        GuardedClient(
                AtomicReference<byte[]> result,
                CountDownLatch latch,
                AtomicBoolean completed,
                String expectedMainFrameOrigin) {
            this.result = result;
            this.latch = latch;
            this.completed = completed;
            this.expectedMainFrameOrigin = expectedMainFrameOrigin;
        }

        @Override
        public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
            try {
                URL verified = validatePublicHttps(request.getUrl().toString());
                if (request.isForMainFrame()
                        && !browserOrigin(verified).equals(expectedMainFrameOrigin)) {
                    finish(
                            completed,
                            result,
                            latch,
                            encodeError("blocked cross-origin browser navigation"));
                    return true;
                }
                return false;
            } catch (Exception error) {
                finish(completed, result, latch, encodeError("blocked browser navigation"));
                return true;
            }
        }

        @Override
        public WebResourceResponse shouldInterceptRequest(
                WebView view,
                WebResourceRequest request) {
            try {
                validatePublicHttps(request.getUrl().toString());
                return null;
            } catch (Exception error) {
                return blockedResponse();
            }
        }

        @Override
        public void onReceivedError(
                WebView view,
                WebResourceRequest request,
                WebResourceError error) {
            if (request.isForMainFrame()) {
                finish(
                        completed,
                        result,
                        latch,
                        encodeError("browser main-frame load failed: " + error.getErrorCode()));
            }
        }

        @Override
        public void onReceivedSslError(WebView view, SslErrorHandler handler, SslError error) {
            handler.cancel();
            finish(completed, result, latch, encodeError("browser TLS validation failed"));
        }
    }
}
