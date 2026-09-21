package ai.ntd97.mobile;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;

public final class NtdBootReceiver extends BroadcastReceiver {
    @Override
    public void onReceive(Context context, Intent intent) {
        String action = intent.getAction();
        if (!Intent.ACTION_BOOT_COMPLETED.equals(action)
                && !Intent.ACTION_LOCKED_BOOT_COMPLETED.equals(action)) {
            return;
        }

        NtdContinuityStore store = new NtdContinuityStore(context);
        if (store.exists()) {
            NtdContinuityScheduler.schedule(
                    context,
                    "reboot",
                    5_000L,
                    false,
                    true);
        }
    }
}
