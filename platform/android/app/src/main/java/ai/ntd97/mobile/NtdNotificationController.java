package ai.ntd97.mobile;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.os.Build;

public final class NtdNotificationController {
    public static final String EXECUTION_CHANNEL = "ntd97.execution";
    public static final String APPROVAL_CHANNEL = "ntd97.approval";
    public static final int EXECUTION_NOTIFICATION_ID = 9701;
    public static final int APPROVAL_NOTIFICATION_ID = 9702;

    private NtdNotificationController() {}

    public static void ensureChannels(Context context) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) {
            return;
        }

        NotificationManager manager =
                (NotificationManager) context.getSystemService(Context.NOTIFICATION_SERVICE);

        NotificationChannel execution = new NotificationChannel(
                EXECUTION_CHANNEL,
                "NTD97 active work",
                NotificationManager.IMPORTANCE_LOW);
        execution.setDescription("Visible foreground execution for NTD97 tasks");

        NotificationChannel approval = new NotificationChannel(
                APPROVAL_CHANNEL,
                "NTD97 approvals",
                NotificationManager.IMPORTANCE_HIGH);
        approval.setDescription("User approval requests for governed actions");

        manager.createNotificationChannel(execution);
        manager.createNotificationChannel(approval);
    }

    public static Notification foregroundNotification(Context context, String status) {
        Intent open = new Intent(context, MainActivity.class);
        PendingIntent pendingOpen = PendingIntent.getActivity(
                context,
                0,
                open,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);

        return new Notification.Builder(context, EXECUTION_CHANNEL)
                .setSmallIcon(android.R.drawable.stat_notify_sync)
                .setContentTitle("NTD97 is working")
                .setContentText(status)
                .setOngoing(true)
                .setContentIntent(pendingOpen)
                .build();
    }

    public static void showApproval(Context context, String capability, String rationale) {
        Intent open = new Intent(context, MainActivity.class)
                .putExtra(MainActivity.EXTRA_OPEN_APPROVAL, true);
        PendingIntent pendingOpen = PendingIntent.getActivity(
                context,
                1,
                open,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);

        Notification notification = new Notification.Builder(context, APPROVAL_CHANNEL)
                .setSmallIcon(android.R.drawable.ic_dialog_alert)
                .setContentTitle("NTD97 needs approval")
                .setContentText(capability + ": " + rationale)
                .setAutoCancel(true)
                .setContentIntent(pendingOpen)
                .build();

        NotificationManager manager =
                (NotificationManager) context.getSystemService(Context.NOTIFICATION_SERVICE);
        manager.notify(APPROVAL_NOTIFICATION_ID, notification);
    }
}
