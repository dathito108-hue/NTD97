package ai.ntd97.mobile;

import android.app.Activity;
import android.app.NotificationManager;
import android.content.Intent;
import android.os.Bundle;
import android.provider.Settings;

import java.io.File;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

public final class NtdLifecycleProbeActivity extends Activity {
    private static final String RESULT_FILE = "ntd97-lifecycle-probe.txt";
    private static final String EXTRA_SEED_REBOOT = "seed_reboot";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        boolean seedReboot = getIntent().getBooleanExtra(EXTRA_SEED_REBOOT, false);
        if (!seedReboot) {
            new NtdContinuityStore(this).clear();
        }

        List<String> results = new ArrayList<>();
        boolean buildAttestationOk = BuildConfig.NTD_SOVEREIGNTY_AUDIT_PASSED
                && BuildConfig.NTD_GIT_SHA != null
                && BuildConfig.NTD_GIT_SHA.matches("[0-9a-fA-F]{40}");
        results.add(buildAttestationOk
                ? "build_attestation=ok"
                : "build_attestation=failed");

        byte[] accessibilityDefault = NtdDeviceAppPlatform.accessibilityInteract(
                getPackageName(),
                "accessibility.click",
                "view_id\t" + getPackageName() + ":id/ntd_accessibility_probe_button");
        boolean accessibilityDefaultBlocked = accessibilityDefault.length >= 2
                && accessibilityDefault[0] == 1
                && accessibilityDefault[1] == 0;
        results.add(accessibilityDefaultBlocked
                ? "accessibility_default_block=ok"
                : "accessibility_default_block=failed");

        NtdRuntimeHost host = NtdSessionController.runtime();

        if (host == null) {
            results.add("native_host=missing");
        } else {
            results.add("native_host=ok");
            try {
                NtdRuntimeHost.AvatarState state = host.avatarState();
                results.add(state == null ? "avatar=missing" : "avatar=ok");

                host.acceptMicrophonePcm(
                        new short[]{0, 1000, -1000, Short.MAX_VALUE},
                        16_000);
                NtdRuntimeHost.AvatarState audioState = host.avatarState();
                short[] speaker = host.pullSpeakerPcm(32, 24_000);
                boolean audioBridgeOk = audioState != null
                        && audioState.lipAmplitude > 0
                        && speaker != null;
                results.add(audioBridgeOk ? "audio_bridge=ok" : "audio_bridge=failed");
            } catch (RuntimeException error) {
                results.add("native_bridge=failed");
            }
        }

        NotificationManager notifications = getSystemService(NotificationManager.class);
        boolean notificationChannelsOk = notifications != null
                && notifications.getNotificationChannel(
                        NtdNotificationController.EXECUTION_CHANNEL) != null
                && notifications.getNotificationChannel(
                        NtdNotificationController.APPROVAL_CHANNEL) != null;
        results.add(notificationChannelsOk
                ? "notification_channels=ok"
                : "notification_channels=failed");

        try {
            startForegroundService(
                    new Intent(this, NtdForegroundService.class)
                            .putExtra(
                                    NtdForegroundService.EXTRA_WAKE_REASON,
                                    "manual_recovery"));
            results.add("foreground_request=ok");
        } catch (RuntimeException error) {
            results.add("foreground_request=failed");
        }

        if (Settings.canDrawOverlays(this)) {
            try {
                startService(new Intent(this, NtdOverlayService.class));
                results.add("overlay_request=ok");
            } catch (RuntimeException error) {
                results.add("overlay_request=failed");
            }
        } else {
            results.add("overlay_request=permission_missing");
        }

        boolean scheduled = NtdContinuityScheduler.schedule(
                this,
                "diagnostic",
                60_000L,
                false,
                true);
        results.add(scheduled ? "persistent_job=ok" : "persistent_job=failed");

        NtdAudioController audio = new NtdAudioController();
        boolean capture = false;
        boolean playback = false;
        try {
            capture = audio.startCapture(this);
            playback = audio.startPlayback();
        } finally {
            audio.close();
        }
        results.add("audio_device_capture=" + capture);
        results.add("audio_device_playback=" + playback);

        if (seedReboot) {
            try {
                new NtdContinuityStore(this).write(new byte[]{0x4e, 0x54, 0x44, 0x39, 0x37});
                results.add("reboot_marker=ok");
            } catch (Exception error) {
                results.add("reboot_marker=failed");
            }
        }

        writeResults(results);
        finish();
    }

    private void writeResults(List<String> results) {
        File output = new File(getFilesDir(), RESULT_FILE);
        String body = String.join("\n", results) + "\n";
        try (FileOutputStream stream = new FileOutputStream(output, false)) {
            stream.write(body.getBytes(StandardCharsets.UTF_8));
            stream.getFD().sync();
        } catch (Exception error) {
            throw new IllegalStateException("failed to write lifecycle probe", error);
        }
    }
}
