package ai.ntd97.mobile;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.view.Gravity;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;

public final class MainActivity extends Activity {
    public static final String EXTRA_OPEN_APPROVAL = "open_approval";
    private static final int NOTIFICATION_PERMISSION_REQUEST = 97;
    private static final int MICROPHONE_PERMISSION_REQUEST = 98;

    private NtdAvatarGLSurfaceView avatarView;
    private TextView statusView;
    private LinearLayout approvalPanel;
    private final NtdAudioController audioController = new NtdAudioController();
    private boolean voiceActive;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        NtdNotificationController.ensureChannels(this);
        requestNotificationPermissionIfNeeded();

        FrameLayout root = new FrameLayout(this);
        root.setBackgroundColor(Color.rgb(10, 10, 15));

        avatarView = new NtdAvatarGLSurfaceView(this);
        root.addView(
                avatarView,
                new FrameLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.MATCH_PARENT));

        LinearLayout controls = new LinearLayout(this);
        controls.setOrientation(LinearLayout.VERTICAL);
        controls.setPadding(24, 24, 24, 24);

        statusView = new TextView(this);
        statusView.setTextColor(Color.WHITE);
        statusView.setTextSize(16.0f);
        controls.addView(statusView);

        Button overlay = new Button(this);
        overlay.setText("Floating assistant");
        overlay.setOnClickListener(view -> openOverlay());
        controls.addView(overlay);

        Button voice = new Button(this);
        voice.setText("Voice");
        voice.setOnClickListener(view -> toggleVoice());
        controls.addView(voice);

        if (BuildConfig.DEBUG) {
            Button physicalValidation = new Button(this);
            physicalValidation.setText("Physical validation");
            physicalValidation.setOnClickListener(view -> openPhysicalValidation());
            controls.addView(physicalValidation);
        }

        approvalPanel = new LinearLayout(this);
        approvalPanel.setOrientation(LinearLayout.HORIZONTAL);
        approvalPanel.setVisibility(LinearLayout.GONE);

        Button approve = new Button(this);
        approve.setText("Approve");
        approve.setOnClickListener(view -> resolveApproval(true));
        approvalPanel.addView(approve);

        Button deny = new Button(this);
        deny.setText("Deny");
        deny.setOnClickListener(view -> resolveApproval(false));
        approvalPanel.addView(deny);
        controls.addView(approvalPanel);

        FrameLayout.LayoutParams controlParams = new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT);
        controlParams.gravity = Gravity.BOTTOM;
        root.addView(controls, controlParams);

        setContentView(root);
        restoreFromUi();
    }

    @Override
    protected void onResume() {
        super.onResume();
        refreshAvatar();
    }

    @Override
    protected void onPause() {
        NtdSessionController.checkpoint(this);
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        audioController.close();
        super.onDestroy();
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode,
            String[] permissions,
            int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == MICROPHONE_PERMISSION_REQUEST
                && grantResults.length > 0
                && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            toggleVoice();
        }
    }

    private void restoreFromUi() {
        NtdRuntimeHost.ResumeResult result =
                NtdSessionController.restore(this, "user_interaction");
        statusView.setText(result.status);
        approvalPanel.setVisibility(
                result.approval == null ? LinearLayout.GONE : LinearLayout.VISIBLE);
        refreshAvatar();
    }

    private void refreshAvatar() {
        NtdRuntimeHost host = NtdSessionController.runtime();
        if (host == null) {
            avatarView.setAvatarState(NtdRuntimeHost.AvatarState.idle());
            return;
        }
        NtdRuntimeHost.AvatarState state = host.avatarState();
        avatarView.setAvatarState(state);
        statusView.setText(state.status);
    }

    private void resolveApproval(boolean approved) {
        NtdRuntimeHost host = NtdSessionController.runtime();
        if (host == null) {
            statusView.setText("Runtime unavailable");
            return;
        }

        boolean accepted = host.resolveApproval(approved);
        if (accepted) {
            NtdSessionController.checkpoint(this);
            approvalPanel.setVisibility(LinearLayout.GONE);
            refreshAvatar();
        } else {
            statusView.setText("Approval state changed; reopen task");
        }
    }

    private void toggleVoice() {
        if (voiceActive) {
            audioController.stopCapture();
            audioController.stopPlayback();
            voiceActive = false;
            statusView.setText("Voice stopped");
            return;
        }

        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(
                    new String[]{Manifest.permission.RECORD_AUDIO},
                    MICROPHONE_PERMISSION_REQUEST);
            return;
        }

        boolean capture = audioController.startCapture(this);
        boolean playback = audioController.startPlayback();
        voiceActive = capture || playback;
        statusView.setText(voiceActive ? "Local voice active" : "Voice unavailable");
    }

    private void openPhysicalValidation() {
        Intent validation = new Intent();
        validation.setClassName(
                this,
                getPackageName() + ".NtdPhysicalEvidenceActivity");
        startActivity(validation);
    }

    private void openOverlay() {
        if (!Settings.canDrawOverlays(this)) {
            Intent permission = new Intent(
                    Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                    Uri.parse("package:" + getPackageName()));
            startActivity(permission);
            return;
        }
        startService(new Intent(this, NtdOverlayService.class));
    }

    private void requestNotificationPermissionIfNeeded() {
        if (Build.VERSION.SDK_INT >= 33
                && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)
                        != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(
                    new String[]{Manifest.permission.POST_NOTIFICATIONS},
                    NOTIFICATION_PERMISSION_REQUEST);
        }
    }
}
