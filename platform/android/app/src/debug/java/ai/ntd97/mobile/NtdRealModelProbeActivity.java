package ai.ntd97.mobile;

import android.app.Activity;
import android.os.Bundle;

import java.io.File;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;

public final class NtdRealModelProbeActivity extends Activity {
    private static final String MODEL_ROOT = "ntd97-real-model";
    private static final String CAPSULE = "stories260K.ncc97";
    private static final String SHARDS = "shards";
    private static final String RESULT = "ntd97-real-model-probe.txt";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        File root = new File(getFilesDir(), MODEL_ROOT);
        File capsule = new File(root, CAPSULE);
        File shards = new File(root, SHARDS);
        byte[] result;

        if (!capsule.isFile() || !shards.isDirectory()) {
            result = "real_model_generation=failed\nreason=native assets missing\n"
                    .getBytes(StandardCharsets.UTF_8);
        } else {
            result = NtdNativeRuntimeHost.probeRealModel(capsule, shards);
        }

        writeResult(result);
        finish();
    }

    private void writeResult(byte[] result) {
        File output = new File(getFilesDir(), RESULT);
        try (FileOutputStream stream = new FileOutputStream(output, false)) {
            stream.write(result == null ? new byte[0] : result);
            stream.getFD().sync();
        } catch (Exception error) {
            throw new IllegalStateException("failed to write real-model probe", error);
        }
    }
}
