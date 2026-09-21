package ai.ntd97.mobile;

import android.app.Service;
import android.content.Intent;
import android.os.IBinder;

public final class NtdForegroundService extends Service {
    public static final String EXTRA_WAKE_REASON = "wake_reason";

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        String reason = intent == null ? "foreground" : intent.getStringExtra(EXTRA_WAKE_REASON);
        NtdRuntimeHost.ResumeResult result = NtdSessionController.restore(this, reason);

        startForeground(
                NtdNotificationController.EXECUTION_NOTIFICATION_ID,
                NtdNotificationController.foregroundNotification(this, result.status));

        if (!result.restored || !result.hasEligibleWork) {
            stopSelf(startId);
            return START_NOT_STICKY;
        }

        NtdSessionController.checkpoint(this);

        if (!result.requiresForeground) {
            stopSelf(startId);
        }
        return START_REDELIVER_INTENT;
    }

    @Override
    public void onTaskRemoved(Intent rootIntent) {
        NtdSessionController.checkpoint(this);
        super.onTaskRemoved(rootIntent);
    }

    @Override
    public void onDestroy() {
        NtdSessionController.checkpoint(this);
        super.onDestroy();
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
