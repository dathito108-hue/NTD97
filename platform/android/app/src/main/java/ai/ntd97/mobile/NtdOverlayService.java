package ai.ntd97.mobile;

import android.app.Service;
import android.content.Intent;
import android.graphics.PixelFormat;
import android.os.IBinder;
import android.provider.Settings;
import android.view.Gravity;
import android.view.WindowManager;

public final class NtdOverlayService extends Service {
    private WindowManager windowManager;
    private NtdAvatarGLSurfaceView avatarView;

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (!Settings.canDrawOverlays(this)) {
            stopSelf(startId);
            return START_NOT_STICKY;
        }

        if (avatarView == null) {
            windowManager = (WindowManager) getSystemService(WINDOW_SERVICE);
            avatarView = new NtdAvatarGLSurfaceView(this);

            NtdRuntimeHost host = NtdSessionController.runtime();
            if (host != null) {
                avatarView.setAvatarState(host.avatarState());
            }

            WindowManager.LayoutParams params = new WindowManager.LayoutParams(
                    360,
                    360,
                    WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
                    WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE
                            | WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
                    PixelFormat.TRANSLUCENT);
            params.gravity = Gravity.TOP | Gravity.END;
            params.x = 24;
            params.y = 160;
            windowManager.addView(avatarView, params);
        }

        return START_STICKY;
    }

    @Override
    public void onDestroy() {
        if (avatarView != null && windowManager != null) {
            windowManager.removeView(avatarView);
            avatarView = null;
        }
        super.onDestroy();
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
