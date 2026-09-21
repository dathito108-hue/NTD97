package ai.ntd97.mobile;

import android.content.Context;

import java.util.ServiceLoader;

public final class NtdRuntimeBootstrap {
    private NtdRuntimeBootstrap() {}

    public static boolean attachFirstLocalProvider(Context context) {
        ClassLoader loader = context.getClassLoader();
        ServiceLoader<NtdRuntimeHost> providers =
                ServiceLoader.load(NtdRuntimeHost.class, loader);

        for (NtdRuntimeHost host : providers) {
            NtdSessionController.attachRuntime(host);
            return true;
        }
        return false;
    }
}
