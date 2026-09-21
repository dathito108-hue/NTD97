package ai.ntd97.mobile;

import android.app.Activity;
import android.app.ActivityManager;
import android.content.Intent;
import android.content.IntentFilter;
import android.net.Uri;
import android.os.BatteryManager;
import android.os.Build;
import android.os.Bundle;
import android.os.SystemClock;
import android.widget.TextView;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class NtdPhysicalEvidenceActivity extends Activity {
    private static final String EVIDENCE_FILE = "ntd97-device-evidence.nde97";
    private static final String SUMMARY_FILE = "ntd97-device-evidence.txt";
    private static final String EXTRA_HEADLESS = "headless";
    private static final int EXPORT_REQUEST = 197;
    private static final int MIN_SAMPLES = 32;
    private static final int MAX_SAMPLES = 8192;
    private static final long MIN_MEASUREMENT_NANOS = 5_000_000_000L;
    private static final int RECOVERY_ATTEMPTS = 12;
    private static final long GIB = 1024L * 1024L * 1024L;

    private TextView statusView;
    private byte[] pendingExport = new byte[0];
    private boolean headless;

    static {
        System.loadLibrary("ntd97_android");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        headless = getIntent().getBooleanExtra(EXTRA_HEADLESS, false);

        statusView = new TextView(this);
        statusView.setPadding(32, 32, 32, 32);
        statusView.setText("NTD97 physical validation in progress…\nKeep the phone unplugged.");
        setContentView(statusView);

        Thread worker = new Thread(this::collect, "ntd97-physical-evidence");
        worker.start();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != EXPORT_REQUEST) {
            return;
        }
        if (resultCode != RESULT_OK || data == null || data.getData() == null) {
            statusView.append("\nExport canceled. Evidence remains inside the debug app.");
            return;
        }

        Uri destination = data.getData();
        try (OutputStream stream = getContentResolver().openOutputStream(destination, "w")) {
            if (stream == null) {
                throw new IllegalStateException("document output stream unavailable");
            }
            stream.write(pendingExport);
            stream.flush();
            statusView.append("\nNDE97 evidence saved successfully.");
        } catch (Exception error) {
            statusView.append(
                    "\nEvidence export failed: " + error.getClass().getSimpleName());
        }
    }

    private void collect() {
        String status = "ok";
        byte[] encoded = new byte[0];
        String summary;
        try {
            if (isProbablyEmulator()) {
                summary = "status=emulator-rejected\n";
                writeFile(SUMMARY_FILE, summary.getBytes(StandardCharsets.UTF_8));
                String rejection = summary;
                runOnUiThread(() -> {
                    statusView.setText(rejection);
                    finish();
                });
                return;
            }

            long totalRamBytes = totalRamBytes();
            String profile = profileForRam(totalRamBytes);
            String fingerprintHash = sha256Hex(Build.FINGERPRINT);
            String buildRevision = BuildConfig.NTD_GIT_SHA;

            for (int i = 0; i < 4; i++) {
                nativeValidationWorkload();
            }

            EnergyReading beforeEnergy = readEnergy();
            long measurementStart = SystemClock.elapsedRealtimeNanos();
            List<Long> latencies = new ArrayList<>();
            int attempts = 0;
            int successes = 0;

            while (attempts < MAX_SAMPLES) {
                long started = SystemClock.elapsedRealtimeNanos();
                long checksum = nativeValidationWorkload();
                long elapsed = SystemClock.elapsedRealtimeNanos() - started;
                attempts++;
                if (checksum >= 0L) {
                    successes++;
                    latencies.add(Math.max(1L, elapsed));
                }

                long duration = SystemClock.elapsedRealtimeNanos() - measurementStart;
                if (attempts >= MIN_SAMPLES && duration >= MIN_MEASUREMENT_NANOS) {
                    break;
                }
            }

            EnergyReading afterEnergy = readEnergy();
            int recoverySuccesses = 0;
            for (int i = 0; i < RECOVERY_ATTEMPTS; i++) {
                if (nativeRecoveryProbe()) {
                    recoverySuccesses++;
                }
            }

            long p95LatencyNanos = percentile95(latencies);
            int reliabilityPermille = permille(successes, attempts);
            int recoveryPermille = permille(recoverySuccesses, RECOVERY_ATTEMPTS);
            EnergyResult energy = energyPerTask(beforeEnergy, afterEnergy, successes);

            encoded = nativeEncodeEvidence(
                    profile,
                    fingerprintHash,
                    buildRevision,
                    totalRamBytes,
                    p95LatencyNanos,
                    energy.microjoulesPerTask,
                    reliabilityPermille,
                    recoveryPermille,
                    BuildConfig.NTD_SOVEREIGNTY_AUDIT_PASSED,
                    attempts,
                    energy.source);

            if (encoded == null || encoded.length == 0) {
                status = "native-evidence-encoding-failed";
                encoded = new byte[0];
            }
            summary = summary(
                    status,
                    profile,
                    fingerprintHash,
                    buildRevision,
                    totalRamBytes,
                    attempts,
                    p95LatencyNanos,
                    energy,
                    reliabilityPermille,
                    recoveryPermille);
        } catch (Exception error) {
            status = "collector-failed:" + error.getClass().getSimpleName();
            summary = "status=" + status + "\n";
        }

        if (encoded.length > 0) {
            writeFile(EVIDENCE_FILE, encoded);
        }
        writeFile(SUMMARY_FILE, summary.getBytes(StandardCharsets.UTF_8));

        byte[] export = encoded.clone();
        String finalSummary = summary;
        runOnUiThread(() -> collectionFinished(export, finalSummary));
    }

    private void collectionFinished(byte[] encoded, String summary) {
        statusView.setText(summary);
        if (encoded.length == 0) {
            if (headless) {
                finish();
            }
            return;
        }

        pendingExport = encoded;
        if (headless) {
            finish();
            return;
        }

        Intent create = new Intent(Intent.ACTION_CREATE_DOCUMENT);
        create.addCategory(Intent.CATEGORY_OPENABLE);
        create.setType("application/octet-stream");
        create.putExtra(
                Intent.EXTRA_TITLE,
                "ntd97-physical-evidence-" + shortBuildRevision() + ".nde97");
        statusView.append("\nChoose where to save the NDE97 evidence file.");
        try {
            startActivityForResult(create, EXPORT_REQUEST);
        } catch (RuntimeException error) {
            statusView.append(
                    "\nDocument picker unavailable: " + error.getClass().getSimpleName());
        }
    }

    private static String shortBuildRevision() {
        String revision = BuildConfig.NTD_GIT_SHA;
        if (revision == null || revision.length() < 8) {
            return "unattested";
        }
        return revision.substring(0, 8);
    }

    private static boolean isProbablyEmulator() {
        String fingerprint = Build.FINGERPRINT == null ? "" : Build.FINGERPRINT.toLowerCase();
        String model = Build.MODEL == null ? "" : Build.MODEL.toLowerCase();
        String manufacturer =
                Build.MANUFACTURER == null ? "" : Build.MANUFACTURER.toLowerCase();
        String brand = Build.BRAND == null ? "" : Build.BRAND.toLowerCase();
        String device = Build.DEVICE == null ? "" : Build.DEVICE.toLowerCase();
        String product = Build.PRODUCT == null ? "" : Build.PRODUCT.toLowerCase();

        return fingerprint.startsWith("generic")
                || fingerprint.startsWith("unknown")
                || model.contains("google_sdk")
                || model.contains("emulator")
                || model.contains("android sdk built for")
                || manufacturer.contains("genymotion")
                || (brand.startsWith("generic") && device.startsWith("generic"))
                || product.contains("sdk_gphone")
                || product.contains("google_sdk")
                || product.contains("emulator")
                || product.contains("simulator");
    }

    private long totalRamBytes() {
        ActivityManager manager = getSystemService(ActivityManager.class);
        if (manager == null) {
            return 0L;
        }
        ActivityManager.MemoryInfo info = new ActivityManager.MemoryInfo();
        manager.getMemoryInfo(info);
        return Math.max(0L, info.totalMem);
    }

    private static String profileForRam(long totalRamBytes) {
        if (totalRamBytes <= 6L * GIB) {
            return "mobile-4gb";
        }
        if (totalRamBytes <= 10L * GIB) {
            return "mobile-8gb";
        }
        return "mobile-12gb";
    }

    private EnergyReading readEnergy() {
        Intent battery = registerReceiver(null, new IntentFilter(Intent.ACTION_BATTERY_CHANGED));
        if (battery != null) {
            int status = battery.getIntExtra(
                    BatteryManager.EXTRA_STATUS,
                    BatteryManager.BATTERY_STATUS_UNKNOWN);
            if (status == BatteryManager.BATTERY_STATUS_CHARGING
                    || status == BatteryManager.BATTERY_STATUS_FULL) {
                return EnergyReading.unavailable("charging");
            }
        }

        BatteryManager manager = getSystemService(BatteryManager.class);
        if (manager == null) {
            return EnergyReading.unavailable("battery-manager-unavailable");
        }

        long energyCounter =
                manager.getLongProperty(BatteryManager.BATTERY_PROPERTY_ENERGY_COUNTER);
        if (energyCounter > 0L && energyCounter != Long.MIN_VALUE) {
            return new EnergyReading(true, "energy-counter", energyCounter);
        }

        int chargeMicroAmpHours =
                manager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER);
        int voltageMilliVolts = battery == null
                ? -1
                : battery.getIntExtra(BatteryManager.EXTRA_VOLTAGE, -1);
        if (chargeMicroAmpHours > 0
                && chargeMicroAmpHours != Integer.MIN_VALUE
                && voltageMilliVolts > 0) {
            long nanoWattHours;
            try {
                nanoWattHours =
                        Math.multiplyExact((long) chargeMicroAmpHours, (long) voltageMilliVolts);
            } catch (ArithmeticException error) {
                return EnergyReading.unavailable("charge-counter-overflow");
            }
            return new EnergyReading(true, "charge-counter-voltage", nanoWattHours);
        }
        return EnergyReading.unavailable("energy-counter-unsupported");
    }

    private static EnergyResult energyPerTask(
            EnergyReading before,
            EnergyReading after,
            int successfulTasks) {
        if (successfulTasks <= 0
                || !before.available
                || !after.available
                || !before.source.equals(after.source)
                || before.nanoWattHours <= after.nanoWattHours) {
            String source = before.source.equals(after.source)
                    ? before.source + "-no-positive-delta"
                    : "energy-source-changed";
            return new EnergyResult(Long.MAX_VALUE, source);
        }

        long deltaNWh = before.nanoWattHours - after.nanoWattHours;
        long totalMicrojoules;
        if (deltaNWh > Long.MAX_VALUE / 18L) {
            totalMicrojoules = Long.MAX_VALUE;
        } else {
            totalMicrojoules = (deltaNWh * 18L) / 5L;
        }
        long perTask = Math.max(1L, totalMicrojoules / successfulTasks);
        return new EnergyResult(perTask, before.source);
    }

    private static long percentile95(List<Long> values) {
        if (values.isEmpty()) {
            return Long.MAX_VALUE;
        }
        Collections.sort(values);
        int last = values.size() - 1;
        int index = (last * 95 + 99) / 100;
        return values.get(Math.min(last, index));
    }

    private static int permille(int passed, int attempted) {
        if (attempted <= 0) {
            return 0;
        }
        return Math.max(0, Math.min(1000, (passed * 1000) / attempted));
    }

    private static String sha256Hex(String value) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        byte[] bytes = digest.digest(value.getBytes(StandardCharsets.UTF_8));
        StringBuilder hex = new StringBuilder(bytes.length * 2);
        for (byte item : bytes) {
            hex.append(String.format("%02x", item & 0xff));
        }
        return "sha256:" + hex;
    }

    private static String summary(
            String status,
            String profile,
            String fingerprintHash,
            String buildRevision,
            long totalRamBytes,
            int sampleCount,
            long p95LatencyNanos,
            EnergyResult energy,
            int reliabilityPermille,
            int recoveryPermille) {
        return "status=" + status + "\n"
                + "profile=" + profile + "\n"
                + "fingerprint_hash=" + fingerprintHash + "\n"
                + "build_revision=" + buildRevision + "\n"
                + "total_ram_bytes=" + totalRamBytes + "\n"
                + "sample_count=" + sampleCount + "\n"
                + "p95_latency_nanos=" + p95LatencyNanos + "\n"
                + "energy_per_task_microjoules=" + energy.microjoulesPerTask + "\n"
                + "energy_source=" + energy.source + "\n"
                + "reliability_permille=" + reliabilityPermille + "\n"
                + "recovery_permille=" + recoveryPermille + "\n"
                + "sovereignty_audit_passed="
                + BuildConfig.NTD_SOVEREIGNTY_AUDIT_PASSED
                + "\n";
    }

    private void writeFile(String name, byte[] bytes) {
        File output = new File(getFilesDir(), name);
        try (FileOutputStream stream = new FileOutputStream(output, false)) {
            stream.write(bytes);
            stream.getFD().sync();
        } catch (Exception error) {
            throw new IllegalStateException("failed to write physical evidence", error);
        }
    }

    private static final class EnergyReading {
        final boolean available;
        final String source;
        final long nanoWattHours;

        EnergyReading(boolean available, String source, long nanoWattHours) {
            this.available = available;
            this.source = source;
            this.nanoWattHours = nanoWattHours;
        }

        static EnergyReading unavailable(String source) {
            return new EnergyReading(false, source, 0L);
        }
    }

    private static final class EnergyResult {
        final long microjoulesPerTask;
        final String source;

        EnergyResult(long microjoulesPerTask, String source) {
            this.microjoulesPerTask = microjoulesPerTask;
            this.source = source;
        }
    }

    private static native long nativeValidationWorkload();

    private static native boolean nativeRecoveryProbe();

    private static native byte[] nativeEncodeEvidence(
            String profile,
            String fingerprintHash,
            String buildRevision,
            long totalRamBytes,
            long p95LatencyNanos,
            long energyPerTaskMicrojoules,
            int reliabilityPermille,
            int recoveryPermille,
            boolean sovereigntyAuditPassed,
            int sampleCount,
            String energySource);
}
