package ai.ntd97.mobile;

import android.content.Context;

public final class NtdRuntimeBootstrap {
    private NtdRuntimeBootstrap() {}

    public static boolean attachFirstLocalProvider(Context context) {
        NtdNativeRuntimeHost host = NtdNativeRuntimeHost.create(context);
        if (host == null) {
            return false;
        }

        NtdSessionController.attachRuntime(host);
        return true;
    }
}
