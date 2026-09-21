package ai.ntd97.mobile;

import android.content.Context;
import android.util.AtomicFile;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

final class NtdConversationStore {
    private static final String FILE_NAME = "ntd97-conversation-state.ncs97";

    private final AtomicFile atomicFile;

    NtdConversationStore(Context context) {
        Context storageContext = context.createDeviceProtectedStorageContext();
        File file = new File(storageContext.getNoBackupFilesDir(), FILE_NAME);
        atomicFile = new AtomicFile(file);
    }

    synchronized void write(byte[] bytes) throws IOException {
        if (bytes == null || bytes.length == 0) {
            throw new IOException("empty NCS97 checkpoint");
        }

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

    synchronized byte[] read() throws IOException {
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

    synchronized void clear() {
        atomicFile.delete();
    }
}
