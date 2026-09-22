package ai.ntd97.mobile;

import android.accessibilityservice.AccessibilityService;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.accessibility.AccessibilityEvent;
import android.view.accessibility.AccessibilityNodeInfo;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

public final class NtdAccessibilityService extends AccessibilityService {
    private static final int MAX_SELECTOR_CHARS = 4096;
    private static final int MAX_TEXT_CHARS = 64 * 1024;
    private static final long ACTION_TIMEOUT_MILLIS = 3_000L;

    private static volatile NtdAccessibilityService instance;

    @Override
    protected void onServiceConnected() {
        super.onServiceConnected();
        instance = this;
    }

    @Override
    public void onAccessibilityEvent(AccessibilityEvent event) {
        // The service is command-driven. Events only keep Android's accessibility
        // connection/window cache current; NTD97 does not infer actions from events.
    }

    @Override
    public void onInterrupt() {
        // No queued autonomous work exists to cancel.
    }

    @Override
    public void onDestroy() {
        if (instance == this) {
            instance = null;
        }
        super.onDestroy();
    }

    static String perform(
            String packageName,
            String operation,
            String payload) throws Exception {
        NtdAccessibilityService service = instance;
        if (service == null) {
            throw new IllegalStateException("accessibility service is not connected");
        }
        if (packageName == null || packageName.isEmpty()) {
            throw new IllegalArgumentException("accessibility package is empty");
        }
        if (operation == null || operation.isEmpty()) {
            throw new IllegalArgumentException("accessibility operation is empty");
        }
        if (payload == null || payload.isEmpty()) {
            throw new IllegalArgumentException("accessibility payload is empty");
        }

        if (Looper.myLooper() == Looper.getMainLooper()) {
            return service.performOnServiceThread(packageName, operation, payload);
        }

        CountDownLatch latch = new CountDownLatch(1);
        AtomicReference<String> result = new AtomicReference<>();
        AtomicReference<Exception> failure = new AtomicReference<>();
        new Handler(Looper.getMainLooper()).post(() -> {
            try {
                result.set(service.performOnServiceThread(packageName, operation, payload));
            } catch (Exception error) {
                failure.set(error);
            } finally {
                latch.countDown();
            }
        });

        if (!latch.await(ACTION_TIMEOUT_MILLIS, TimeUnit.MILLISECONDS)) {
            throw new IllegalStateException("accessibility action timed out");
        }
        Exception error = failure.get();
        if (error != null) {
            throw error;
        }
        String receipt = result.get();
        if (receipt == null || receipt.isEmpty()) {
            throw new IllegalStateException("accessibility action returned no receipt");
        }
        return receipt;
    }

    private String performOnServiceThread(
            String packageName,
            String operation,
            String payload) {
        AccessibilityNodeInfo root = getRootInActiveWindow();
        if (root == null) {
            throw new IllegalStateException("accessibility active window is unavailable");
        }
        CharSequence activePackage = root.getPackageName();
        if (activePackage == null || !packageName.contentEquals(activePackage)) {
            throw new IllegalStateException("accessibility foreground package mismatch");
        }

        String[] parts = payload.split("\\t", -1);
        if ("accessibility.click".equals(operation)) {
            if (parts.length != 2) {
                throw new IllegalArgumentException("accessibility click payload is invalid");
            }
            AccessibilityNodeInfo node = uniqueNode(root, packageName, parts[0], parts[1]);
            if (!node.isClickable()) {
                throw new IllegalStateException("accessibility target is not clickable");
            }
            if (!node.performAction(AccessibilityNodeInfo.ACTION_CLICK)) {
                throw new IllegalStateException("accessibility click was rejected by Android");
            }
            return "click";
        }

        if ("accessibility.set_text".equals(operation)) {
            if (parts.length != 3 || parts[2].isEmpty() || parts[2].length() > MAX_TEXT_CHARS) {
                throw new IllegalArgumentException("accessibility set_text payload is invalid");
            }
            AccessibilityNodeInfo node = uniqueNode(root, packageName, parts[0], parts[1]);
            if (!node.isEditable()) {
                throw new IllegalStateException("accessibility target is not editable");
            }
            Bundle arguments = new Bundle();
            arguments.putCharSequence(
                    AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE,
                    parts[2]);
            if (!node.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, arguments)) {
                throw new IllegalStateException("accessibility set_text was rejected by Android");
            }
            if (!node.refresh()
                    || node.getText() == null
                    || !parts[2].contentEquals(node.getText())) {
                throw new IllegalStateException("accessibility set_text read-back mismatch");
            }
            return "set_text";
        }

        throw new IllegalArgumentException("unsupported accessibility operation");
    }

    private AccessibilityNodeInfo uniqueNode(
            AccessibilityNodeInfo root,
            String packageName,
            String selectorKind,
            String selectorValue) {
        if (!"view_id".equals(selectorKind)) {
            throw new IllegalArgumentException("unsupported accessibility selector");
        }
        if (selectorValue == null
                || selectorValue.isEmpty()
                || selectorValue.length() > MAX_SELECTOR_CHARS
                || selectorValue.indexOf('\n') >= 0
                || selectorValue.indexOf('\r') >= 0
                || selectorValue.indexOf('\t') >= 0) {
            throw new IllegalArgumentException("accessibility selector is invalid");
        }

        List<AccessibilityNodeInfo> matches = root.findAccessibilityNodeInfosByViewId(selectorValue);
        if (matches == null || matches.size() != 1) {
            throw new IllegalStateException("accessibility selector is missing or ambiguous");
        }
        AccessibilityNodeInfo node = matches.get(0);
        CharSequence nodePackage = node.getPackageName();
        if (nodePackage == null || !packageName.contentEquals(nodePackage)) {
            throw new IllegalStateException("accessibility node package mismatch");
        }
        return node;
    }
}
