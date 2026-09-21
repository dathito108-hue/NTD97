package ai.ntd97.mobile;

import android.app.job.JobInfo;
import android.app.job.JobScheduler;
import android.content.ComponentName;
import android.content.Context;
import android.os.PersistableBundle;

public final class NtdContinuityScheduler {
    private static final int JOB_ID = 9700;
    private static final String KEY_REASON = "reason";

    private NtdContinuityScheduler() {}

    public static boolean schedule(
            Context context,
            String reason,
            long minimumLatencyMs,
            boolean requireNetwork,
            boolean persisted) {
        ComponentName component = new ComponentName(context, NtdContinuityJobService.class);
        PersistableBundle extras = new PersistableBundle();
        extras.putString(KEY_REASON, reason);

        JobInfo.Builder builder = new JobInfo.Builder(JOB_ID, component)
                .setMinimumLatency(Math.max(0L, minimumLatencyMs))
                .setBackoffCriteria(30_000L, JobInfo.BACKOFF_POLICY_EXPONENTIAL)
                .setPersisted(persisted)
                .setExtras(extras);

        if (requireNetwork) {
            builder.setRequiredNetworkType(JobInfo.NETWORK_TYPE_ANY);
        }

        JobScheduler scheduler =
                (JobScheduler) context.getSystemService(Context.JOB_SCHEDULER_SERVICE);
        return scheduler.schedule(builder.build()) == JobScheduler.RESULT_SUCCESS;
    }

    static String reason(PersistableBundle extras) {
        String reason = extras.getString(KEY_REASON);
        return reason == null ? "scheduled" : reason;
    }
}
