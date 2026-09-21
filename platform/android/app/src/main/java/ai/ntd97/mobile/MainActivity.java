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
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends Activity {
    public static final String EXTRA_OPEN_APPROVAL = "open_approval";
    private static final int NOTIFICATION_PERMISSION_REQUEST = 97;
    private static final int MICROPHONE_PERMISSION_REQUEST = 98;
    private static final int CHAT_MAX_NEW_TOKENS = 64;

    private NtdAvatarGLSurfaceView avatarView;
    private TextView statusView;
    private TextView transcriptView;
    private ScrollView transcriptScroll;
    private EditText composerView;
    private Button sendButton;
    private Button stopButton;
    private LinearLayout approvalPanel;
    private final NtdAudioController audioController = new NtdAudioController();
    private final ExecutorService chatExecutor = Executors.newSingleThreadExecutor();
    private final StringBuilder transcript = new StringBuilder();
    private volatile long activeChatRequestId = -1L;
    private volatile boolean chatCancelRequested;
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
        controls.setPadding(dp(16), dp(16), dp(16), dp(16));
        controls.setBackgroundColor(Color.argb(210, 12, 12, 20));

        statusView = new TextView(this);
        statusView.setTextColor(Color.WHITE);
        statusView.setTextSize(15.0f);
        controls.addView(statusView);

        transcriptView = new TextView(this);
        transcriptView.setTextColor(Color.WHITE);
        transcriptView.setTextSize(15.0f);
        transcriptView.setPadding(dp(8), dp(8), dp(8), dp(8));

        transcriptScroll = new ScrollView(this);
        transcriptScroll.setFillViewport(true);
        transcriptScroll.addView(
                transcriptView,
                new ScrollView.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT));
        controls.addView(
                transcriptScroll,
                new LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        dp(220)));

        LinearLayout composerRow = new LinearLayout(this);
        composerRow.setOrientation(LinearLayout.HORIZONTAL);

        composerView = new EditText(this);
        composerView.setHint("Ask NTD97");
        composerView.setHintTextColor(Color.LTGRAY);
        composerView.setTextColor(Color.WHITE);
        composerView.setSingleLine(false);
        composerView.setMaxLines(4);
        composerRow.addView(
                composerView,
                new LinearLayout.LayoutParams(
                        0,
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        1.0f));

        sendButton = new Button(this);
        sendButton.setText("Send");
        sendButton.setOnClickListener(view -> submitChat());
        composerRow.addView(sendButton);

        stopButton = new Button(this);
        stopButton.setText("Stop");
        stopButton.setEnabled(false);
        stopButton.setOnClickListener(view -> cancelChat());
        composerRow.addView(stopButton);
        controls.addView(composerRow);

        LinearLayout actionRow = new LinearLayout(this);
        actionRow.setOrientation(LinearLayout.HORIZONTAL);

        Button overlay = new Button(this);
        overlay.setText("Floating assistant");
        overlay.setOnClickListener(view -> openOverlay());
        actionRow.addView(
                overlay,
                new LinearLayout.LayoutParams(
                        0,
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        1.0f));

        Button voice = new Button(this);
        voice.setText("Voice");
        voice.setOnClickListener(view -> toggleVoice());
        actionRow.addView(
                voice,
                new LinearLayout.LayoutParams(
                        0,
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        1.0f));
        controls.addView(actionRow);

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
        restoreConversationFromDisk();
    }

    @Override
    protected void onResume() {
        super.onResume();
        refreshAvatar();
    }

    @Override
    protected void onPause() {
        NtdSessionController.checkpoint(this);
        NtdSessionController.checkpointConversation(this);
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        long requestId = activeChatRequestId;
        NtdRuntimeHost host = NtdSessionController.runtime();
        if (host != null && requestId >= 0) {
            host.cancelChat(requestId);
        }
        chatExecutor.shutdownNow();
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

    private void submitChat() {
        String prompt = composerView.getText().toString().trim();
        if (prompt.isEmpty() || activeChatRequestId >= 0) {
            return;
        }

        NtdRuntimeHost host = NtdSessionController.runtime();
        if (host == null) {
            statusView.setText("Native runtime unavailable");
            return;
        }

        composerView.setText("");
        sendButton.setEnabled(false);
        stopButton.setEnabled(true);
        chatCancelRequested = false;
        appendTranscript("\nYou: " + prompt + "\nNTD97: ");
        statusView.setText("Loading native model...");

        chatExecutor.execute(() -> {
            if (!host.chatReady()) {
                finishChatUi("No verified native chat model installed", true);
                return;
            }

            long requestId = host.submitChat(prompt, CHAT_MAX_NEW_TOKENS);
            if (requestId < 0) {
                finishChatUi("Native chat request rejected", true);
                return;
            }
            activeChatRequestId = requestId;
            if (chatCancelRequested) {
                host.cancelChat(requestId);
            }

            runOnUiThread(() -> statusView.setText("Generating locally..."));
            streamChat(host, requestId);
        });
    }

    private void streamChat(NtdRuntimeHost host, long requestId) {
        while (!Thread.currentThread().isInterrupted()) {
            NtdRuntimeHost.ChatEvent event = host.nextChatEvent(requestId);
            if (event.kind == NtdRuntimeHost.ChatEvent.TOKEN) {
                NtdSessionController.checkpointConversation(this);
                if (!event.text.isEmpty()) {
                    runOnUiThread(() -> appendTranscript(event.text));
                }
                continue;
            }
            if (event.kind == NtdRuntimeHost.ChatEvent.COMPLETE) {
                finishChatUi("Native response complete", true);
                return;
            }
            if (event.kind == NtdRuntimeHost.ChatEvent.CANCELLED) {
                finishChatUi("Generation cancelled", true);
                return;
            }
            finishChatUi(
                    event.text.isEmpty() ? "Native generation failed" : event.text,
                    true);
            return;
        }

        host.cancelChat(requestId);
        finishChatUi("Generation interrupted", true);
    }

    private void cancelChat() {
        chatCancelRequested = true;
        long requestId = activeChatRequestId;
        NtdRuntimeHost host = NtdSessionController.runtime();
        if (host != null && requestId >= 0) {
            host.cancelChat(requestId);
        }
        statusView.setText("Cancelling native generation...");
    }

    private void finishChatUi(String status, boolean terminateAssistantLine) {
        NtdSessionController.checkpointConversation(this);
        activeChatRequestId = -1L;
        runOnUiThread(() -> {
            if (terminateAssistantLine) {
                appendTranscript("\n");
            }
            statusView.setText(status);
            sendButton.setEnabled(true);
            stopButton.setEnabled(false);
        });
    }

    private void appendTranscript(String text) {
        transcript.append(text);
        transcriptView.setText(transcript.toString());
        transcriptScroll.post(
                () -> transcriptScroll.fullScroll(ScrollView.FOCUS_DOWN));
    }

    private void restoreConversationFromDisk() {
        NtdRuntimeHost host = NtdSessionController.runtime();
        if (host == null) {
            appendTranscript("NTD97 local chat unavailable.\n");
            return;
        }

        sendButton.setEnabled(false);
        stopButton.setEnabled(false);
        statusView.setText("Restoring sovereign conversation...");

        chatExecutor.execute(() -> {
            long requestId = NtdSessionController.restoreConversation(this);
            String restoredTranscript = host.chatTranscript();
            runOnUiThread(() -> {
                transcript.setLength(0);
                if (restoredTranscript.isEmpty()) {
                    appendTranscript("NTD97 local chat ready.\n");
                } else {
                    appendTranscript(restoredTranscript);
                    if (!restoredTranscript.endsWith("\n")) {
                        appendTranscript("\n");
                    }
                }
            });

            if (requestId < 0) {
                runOnUiThread(() -> {
                    statusView.setText("Conversation restore failed");
                    sendButton.setEnabled(true);
                });
                return;
            }
            if (requestId == 0) {
                runOnUiThread(() -> {
                    statusView.setText("Native runtime ready");
                    sendButton.setEnabled(true);
                });
                return;
            }

            activeChatRequestId = requestId;
            chatCancelRequested = false;
            runOnUiThread(() -> {
                statusView.setText("Restoring native generation...");
                sendButton.setEnabled(false);
                stopButton.setEnabled(true);
            });

            if (!host.chatReady()) {
                runOnUiThread(() -> statusView.setText(
                        "Verified native model required to resume generation"));
                return;
            }

            runOnUiThread(() -> statusView.setText("Resuming local generation..."));
            streamChat(host, requestId);
        });
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
        if (activeChatRequestId < 0) {
            statusView.setText(state.status);
        }
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

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }
}
