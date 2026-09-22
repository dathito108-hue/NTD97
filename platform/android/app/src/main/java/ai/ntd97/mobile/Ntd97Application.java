package ai.ntd97.mobile;

import android.app.Application;

public final class Ntd97Application extends Application {
    private NtdContinuityStore continuityStore;

    @Override
    public void onCreate() {
        super.onCreate();
        continuityStore = new NtdContinuityStore(this);
        NtdWebPlatform.initialize(this);
        NtdNotificationController.ensureChannels(this);
        NtdRuntimeBootstrap.attachFirstLocalProvider(this);
    }

    public NtdContinuityStore continuityStore() {
        return continuityStore;
    }
}
