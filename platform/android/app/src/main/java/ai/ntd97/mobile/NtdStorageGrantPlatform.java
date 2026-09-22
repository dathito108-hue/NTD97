package ai.ntd97.mobile;

import android.content.ContentResolver;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.UriPermission;
import android.database.Cursor;
import android.net.Uri;
import android.os.ParcelFileDescriptor;
import android.provider.DocumentsContract;

import java.io.ByteArrayOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.regex.Pattern;

final class NtdStorageGrantPlatform {
    private static final byte PROTOCOL_VERSION = 1;
    private static final int MAX_FILE_BYTES = 512 * 1024;
    private static final int MAX_PATH_BYTES = 4096;
    private static final int MAX_DEPTH = 32;
    private static final String PREFS_NAME = "ntd97-storage-grants";
    private static final String KEY_PREFIX = "tree:";
    private static final Pattern ALIAS = Pattern.compile("[A-Za-z0-9_-]{1,32}");

    private static volatile Context appContext;

    private NtdStorageGrantPlatform() {}

    static void initialize(Context context) {
        appContext = context.getApplicationContext();
    }

    static Intent buildGrantIntent() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE);
        intent.addFlags(
                Intent.FLAG_GRANT_READ_URI_PERMISSION
                        | Intent.FLAG_GRANT_WRITE_URI_PERMISSION
                        | Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION
                        | Intent.FLAG_GRANT_PREFIX_URI_PERMISSION);
        return intent;
    }

    static boolean persistGrant(Context context, Uri treeUri, String alias, int resultFlags) {
        if (context == null || treeUri == null || !validAlias(alias)) {
            return false;
        }
        int takeFlags = resultFlags
                & (Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
        if ((takeFlags & Intent.FLAG_GRANT_READ_URI_PERMISSION) == 0) {
            return false;
        }
        try {
            ContentResolver resolver = context.getContentResolver();
            resolver.takePersistableUriPermission(treeUri, takeFlags);
            if (!hasPersistedPermission(resolver, treeUri, true, false)) {
                return false;
            }
            SharedPreferences prefs = preferences(context);
            String previous = prefs.getString(KEY_PREFIX + alias, "");
            prefs.edit().putString(KEY_PREFIX + alias, treeUri.toString()).apply();
            if (previous != null
                    && !previous.isEmpty()
                    && !previous.equals(treeUri.toString())) {
                releasePersistedPermission(resolver, Uri.parse(previous));
            }
            return true;
        } catch (Exception error) {
            return false;
        }
    }

    static boolean hasGrant(Context context, String alias) {
        if (context == null || !validAlias(alias)) {
            return false;
        }
        String raw = preferences(context).getString(KEY_PREFIX + alias, "");
        if (raw == null || raw.isEmpty()) {
            return false;
        }
        try {
            Uri tree = Uri.parse(raw);
            return hasPersistedPermission(context.getContentResolver(), tree, true, false);
        } catch (Exception error) {
            return false;
        }
    }

    static byte[] read(String encodedPath) {
        try {
            GrantPath path = parseGrantPath(encodedPath);
            Context context = requireContext();
            Uri tree = grantedTree(context, path.alias, false);
            Uri document = resolveExisting(context.getContentResolver(), tree, path.segments);
            byte[] bytes = readBounded(context.getContentResolver(), document);
            String text = new String(bytes, StandardCharsets.UTF_8);
            if (!Arrays.equals(bytes, text.getBytes(StandardCharsets.UTF_8))) {
                throw new IOException("granted file is not canonical UTF-8 text");
            }
            return encodeSuccess(text);
        } catch (Exception error) {
            return encodeError(safeMessage(error));
        }
    }

    static byte[] write(String encodedPath, byte[] bytes) {
        try {
            if (bytes == null || bytes.length == 0 || bytes.length > MAX_FILE_BYTES) {
                throw new IOException("invalid granted file write size");
            }
            GrantPath path = parseGrantPath(encodedPath);
            Context context = requireContext();
            ContentResolver resolver = context.getContentResolver();
            Uri tree = grantedTree(context, path.alias, true);
            Uri document = resolveForWrite(resolver, tree, path.segments);

            try (ParcelFileDescriptor descriptor =
                            resolver.openFileDescriptor(document, "rwt");
                    FileOutputStream output = descriptor == null
                            ? null
                            : new FileOutputStream(descriptor.getFileDescriptor())) {
                if (descriptor == null || output == null) {
                    throw new IOException("granted file descriptor unavailable");
                }
                output.write(bytes);
                output.flush();
                output.getFD().sync();
            }

            byte[] observed = readBounded(resolver, document);
            if (!Arrays.equals(bytes, observed)) {
                throw new IOException("granted file read-back mismatch");
            }
            String digest = sha256Hex(observed);
            return encodeSuccess(
                    "grant-write:"
                            + path.alias
                            + ":"
                            + sanitizeReceipt(path.relative)
                            + ":"
                            + observed.length
                            + ":"
                            + digest);
        } catch (Exception error) {
            return encodeError(safeMessage(error));
        }
    }

    private static Uri grantedTree(Context context, String alias, boolean write) throws IOException {
        String raw = preferences(context).getString(KEY_PREFIX + alias, "");
        if (raw == null || raw.isEmpty()) {
            throw new IOException("storage grant is not registered: " + alias);
        }
        Uri tree = Uri.parse(raw);
        if (!hasPersistedPermission(context.getContentResolver(), tree, true, write)) {
            throw new IOException("storage grant permission is missing or revoked: " + alias);
        }
        return tree;
    }

    private static void releasePersistedPermission(ContentResolver resolver, Uri tree) {
        for (UriPermission permission : resolver.getPersistedUriPermissions()) {
            if (!tree.equals(permission.getUri())) {
                continue;
            }
            int flags = 0;
            if (permission.isReadPermission()) {
                flags |= Intent.FLAG_GRANT_READ_URI_PERMISSION;
            }
            if (permission.isWritePermission()) {
                flags |= Intent.FLAG_GRANT_WRITE_URI_PERMISSION;
            }
            if (flags != 0) {
                try {
                    resolver.releasePersistableUriPermission(tree, flags);
                } catch (Exception ignored) {
                    // The alias no longer references this URI; stale platform permission
                    // is not used by NTD97 and can still be revoked by the user/OS.
                }
            }
            return;
        }
    }

    private static boolean hasPersistedPermission(
            ContentResolver resolver,
            Uri tree,
            boolean read,
            boolean write) {
        List<UriPermission> permissions = resolver.getPersistedUriPermissions();
        for (UriPermission permission : permissions) {
            if (!tree.equals(permission.getUri())) {
                continue;
            }
            if (read && !permission.isReadPermission()) {
                return false;
            }
            if (write && !permission.isWritePermission()) {
                return false;
            }
            return true;
        }
        return false;
    }

    private static Uri resolveExisting(
            ContentResolver resolver,
            Uri tree,
            String[] segments) throws IOException {
        Uri current = rootDocument(tree);
        String currentId = DocumentsContract.getTreeDocumentId(tree);
        for (int index = 0; index < segments.length; index++) {
            Child child = findChild(resolver, tree, currentId, segments[index]);
            if (child == null) {
                throw new IOException("granted file path does not exist");
            }
            if (index + 1 < segments.length
                    && !DocumentsContract.Document.MIME_TYPE_DIR.equals(child.mimeType)) {
                throw new IOException("granted path segment is not a directory");
            }
            current = DocumentsContract.buildDocumentUriUsingTree(tree, child.documentId);
            currentId = child.documentId;
        }
        return current;
    }

    private static Uri resolveForWrite(
            ContentResolver resolver,
            Uri tree,
            String[] segments) throws IOException {
        Uri parent = rootDocument(tree);
        String parentId = DocumentsContract.getTreeDocumentId(tree);
        for (int index = 0; index + 1 < segments.length; index++) {
            String name = segments[index];
            Child child = findChild(resolver, tree, parentId, name);
            if (child == null) {
                Uri created = DocumentsContract.createDocument(
                        resolver,
                        parent,
                        DocumentsContract.Document.MIME_TYPE_DIR,
                        name);
                if (created == null) {
                    throw new IOException("failed to create granted directory");
                }
                parent = created;
                parentId = DocumentsContract.getDocumentId(created);
                continue;
            }
            if (!DocumentsContract.Document.MIME_TYPE_DIR.equals(child.mimeType)) {
                throw new IOException("granted path segment is not a directory");
            }
            parent = DocumentsContract.buildDocumentUriUsingTree(tree, child.documentId);
            parentId = child.documentId;
        }

        String leaf = segments[segments.length - 1];
        Child existing = findChild(resolver, tree, parentId, leaf);
        if (existing != null) {
            if (DocumentsContract.Document.MIME_TYPE_DIR.equals(existing.mimeType)) {
                throw new IOException("granted write target is a directory");
            }
            return DocumentsContract.buildDocumentUriUsingTree(tree, existing.documentId);
        }
        Uri created = DocumentsContract.createDocument(
                resolver,
                parent,
                "application/octet-stream",
                leaf);
        if (created == null) {
            throw new IOException("failed to create granted file");
        }
        return created;
    }

    private static Uri rootDocument(Uri tree) throws IOException {
        String rootId = DocumentsContract.getTreeDocumentId(tree);
        if (rootId == null || rootId.isEmpty()) {
            throw new IOException("invalid storage tree URI");
        }
        return DocumentsContract.buildDocumentUriUsingTree(tree, rootId);
    }

    private static Child findChild(
            ContentResolver resolver,
            Uri tree,
            String parentId,
            String name) throws IOException {
        Uri children = DocumentsContract.buildChildDocumentsUriUsingTree(tree, parentId);
        String[] projection = new String[] {
                DocumentsContract.Document.COLUMN_DOCUMENT_ID,
                DocumentsContract.Document.COLUMN_DISPLAY_NAME,
                DocumentsContract.Document.COLUMN_MIME_TYPE
        };
        Child match = null;
        try (Cursor cursor = resolver.query(children, projection, null, null, null)) {
            if (cursor == null) {
                throw new IOException("storage provider returned no cursor");
            }
            while (cursor.moveToNext()) {
                String display = cursor.getString(1);
                if (!name.equals(display)) {
                    continue;
                }
                if (match != null) {
                    throw new IOException("ambiguous storage provider child name");
                }
                match = new Child(cursor.getString(0), cursor.getString(2));
            }
        }
        return match;
    }

    private static byte[] readBounded(ContentResolver resolver, Uri document) throws IOException {
        try (InputStream input = resolver.openInputStream(document);
                ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            if (input == null) {
                throw new IOException("granted file input unavailable");
            }
            byte[] buffer = new byte[8192];
            int total = 0;
            while (true) {
                int count = input.read(buffer);
                if (count < 0) {
                    break;
                }
                total += count;
                if (total > MAX_FILE_BYTES) {
                    throw new IOException("granted file exceeds size limit");
                }
                output.write(buffer, 0, count);
            }
            return output.toByteArray();
        }
    }

    private static GrantPath parseGrantPath(String encoded) throws IOException {
        if (encoded == null
                || encoded.isEmpty()
                || encoded.getBytes(StandardCharsets.UTF_8).length > MAX_PATH_BYTES) {
            throw new IOException("invalid granted storage path");
        }
        int separator = encoded.indexOf('\t');
        if (separator <= 0 || separator + 1 >= encoded.length()) {
            throw new IOException("granted storage path must contain alias and relative path");
        }
        String alias = encoded.substring(0, separator);
        String relative = encoded.substring(separator + 1);
        if (!validAlias(alias)
                || relative.startsWith("/")
                || relative.endsWith("/")
                || relative.contains("\\")
                || containsControl(relative)) {
            throw new IOException("invalid granted storage path");
        }
        String[] segments = relative.split("/", -1);
        if (segments.length == 0 || segments.length > MAX_DEPTH) {
            throw new IOException("invalid granted storage depth");
        }
        for (String segment : segments) {
            if (segment.isEmpty()
                    || ".".equals(segment)
                    || "..".equals(segment)
                    || containsControl(segment)) {
                throw new IOException("granted storage traversal is not allowed");
            }
        }
        return new GrantPath(alias, relative, segments);
    }

    private static boolean validAlias(String alias) {
        return alias != null && ALIAS.matcher(alias).matches();
    }

    private static boolean containsControl(String value) {
        for (int index = 0; index < value.length(); index++) {
            if (Character.isISOControl(value.charAt(index))) {
                return true;
            }
        }
        return false;
    }

    private static Context requireContext() throws IOException {
        Context context = appContext;
        if (context == null) {
            throw new IOException("storage grant platform is not initialized");
        }
        return context;
    }

    private static SharedPreferences preferences(Context context) {
        return context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
    }

    private static String sha256Hex(byte[] bytes) throws Exception {
        byte[] digest = MessageDigest.getInstance("SHA-256").digest(bytes);
        StringBuilder out = new StringBuilder(digest.length * 2);
        for (byte value : digest) {
            out.append(String.format(Locale.ROOT, "%02x", value & 0xff));
        }
        return out.toString();
    }

    private static String sanitizeReceipt(String value) {
        return value.replace(':', '_').replace('\n', '_').replace('\r', '_').replace('\t', '_');
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

    private static String safeMessage(Throwable error) {
        String message = error.getMessage();
        return message == null || message.trim().isEmpty()
                ? error.getClass().getSimpleName()
                : message.replace('\n', ' ').replace('\r', ' ');
    }

    private static final class GrantPath {
        final String alias;
        final String relative;
        final String[] segments;

        GrantPath(String alias, String relative, String[] segments) {
            this.alias = alias;
            this.relative = relative;
            this.segments = segments;
        }
    }

    private static final class Child {
        final String documentId;
        final String mimeType;

        Child(String documentId, String mimeType) {
            this.documentId = documentId;
            this.mimeType = mimeType;
        }
    }
}
