package ai.ntd97.mobile;

import android.app.job.JobParameters;
import android.app.job.JobService;
import android.content.Intent;
import android.os.PersistableBundle;

public final class NtdContinuityJobService extends JobService {
    @Override
    public boolean onStartJob(JobParameters params) {
        PersistableBundle extras = params.getExtras();
        String reason = NtdContinuityScheduler.reason(extras);
        NtdRuntimeHost.ResumeResult result = NtdSessionController.restore(this, reason);

        if (result.requiresForeground && result.hasEligibleWork) {
            Intent service = new Intent(this, NtdForegroundService.class)
                    .putExtra(NtdForegroundService.EXTRA_WAKE_REASON, reason);
            startForegroundService(service);
        } else {
            NtdSessionController.checkpoint(this);
        }

        jobFinished(params, result.hasEligibleWork && !result.restored);
        return false;
    }

    @Override
    public boolean onStopJob(JobParameters params) {
        NtdSessionController.checkpoint(this);
        return true;
    }
}
