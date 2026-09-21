package ai.ntd97.mobile;

import android.content.Context;
import android.content.res.AssetManager;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

final class NtdNativeModelStore {
    private static final String BUNDLED_ASSET_ROOT = "ntd97-real-model";
    private static final String BUNDLED_RUNTIME_ROOT = "ntd97-bundled-chat-model";

    static final class ModelFiles {
        final File capsule;
        final File shardRoot;
        final byte[] verifyKey;

        ModelFiles(File capsule, File shardRoot, byte[] verifyKey) {
            this.capsule = capsule;
            this.shardRoot = shardRoot;
            this.verifyKey = verifyKey;
        }
    }

    private NtdNativeModelStore() {}

    static ModelFiles prepareBundledModel(Context context) throws IOException {
        AssetManager assets = context.getAssets();
        String[] children = assets.list(BUNDLED_ASSET_ROOT);
        if (children == null || children.length == 0) {
            return null;
        }

        File root = new File(context.getFilesDir(), BUNDLED_RUNTIME_ROOT);
        deleteTree(root);
        copyAssetTree(assets, BUNDLED_ASSET_ROOT, root);

        File capsule = new File(root, "model.ncc97");
        File shardRoot = new File(root, "ntp97-shards");
        File verifyKey = new File(root, "verify-key.bin");
        if (!capsule.isFile() || !shardRoot.isDirectory() || !verifyKey.isFile()) {
            throw new IOException("bundled native model is incomplete");
        }

        byte[] key = readAllBytes(verifyKey);
        if (key.length != 32) {
            throw new IOException("invalid native model verify key");
        }
        return new ModelFiles(capsule, shardRoot, key);
    }

    private static void copyAssetTree(
            AssetManager assets,
            String assetPath,
            File destination) throws IOException {
        String[] children = assets.list(assetPath);
        if (children == null) {
            throw new IOException("asset listing failed: " + assetPath);
        }

        if (children.length == 0) {
            File parent = destination.getParentFile();
            if (parent != null && !parent.isDirectory() && !parent.mkdirs()) {
                throw new IOException("failed to create model asset parent");
            }
            try (InputStream input = assets.open(assetPath);
                 FileOutputStream output = new FileOutputStream(destination, false)) {
                copy(input, output);
                output.getFD().sync();
            }
            return;
        }

        if (!destination.isDirectory() && !destination.mkdirs()) {
            throw new IOException("failed to create model asset directory");
        }
        for (String child : children) {
            copyAssetTree(
                    assets,
                    assetPath + "/" + child,
                    new File(destination, child));
        }
    }

    private static byte[] readAllBytes(File file) throws IOException {
        try (FileInputStream input = new FileInputStream(file);
             ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            copy(input, output);
            return output.toByteArray();
        }
    }

    private static void copy(InputStream input, java.io.OutputStream output) throws IOException {
        byte[] buffer = new byte[16 * 1024];
        int count;
        while ((count = input.read(buffer)) != -1) {
            output.write(buffer, 0, count);
        }
    }

    private static void deleteTree(File file) throws IOException {
        if (!file.exists()) {
            return;
        }
        if (file.isDirectory()) {
            File[] children = file.listFiles();
            if (children == null) {
                throw new IOException("failed to list native model directory");
            }
            for (File child : children) {
                deleteTree(child);
            }
        }
        if (!file.delete()) {
            throw new IOException("failed to replace native model asset");
        }
    }
}
