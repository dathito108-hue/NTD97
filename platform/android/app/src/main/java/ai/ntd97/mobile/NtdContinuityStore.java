package ai.ntd97.mobile;

import android.content.Context;
import android.util.AtomicFile;

import java.io.File;
import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

public final class NtdContinuityStore {
    private static final String FILE_NAME = "ntd97-mobile-continuity.mcs97";

    private final AtomicFile atomicFile;

    public NtdContinuityStore(Context context) {
        Context storageContext = context.createDeviceProtectedStorageContext();
        File file = new File(storageContext.getNoBackupFilesDir(), FILE_NAME);
        atomicFile = new AtomicFile(file);
    }

    public synchronized void write(byte[] bytes) throws IOException {
        FileOutputStream stream = null;
        try {
            stream = atomicFile.startWrite();
            stream.write(bytes);
            stream.getFD().sync();
            atomicFile.finishWrite(stream);
        } catch (IOException error) {
            if (stream != null) {
                atomicFile.failWrite(stream);
            }
            throw error;
        }
    }

    public synchronized byte[] read() throws IOException {
        if (!atomicFile.getBaseFile().exists()) {
            return null;
        }

        try (FileInputStream stream = atomicFile.openRead();
             ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[8192];
            int read;
            while ((read = stream.read(buffer)) != -1) {
                output.write(buffer, 0, read);
            }
            return output.toByteArray();
        }
    }

    public synchronized boolean exists() {
        return atomicFile.getBaseFile().exists();
    }

    public synchronized void clear() {
        atomicFile.delete();
    }
}
