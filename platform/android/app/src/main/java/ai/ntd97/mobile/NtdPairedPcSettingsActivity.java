package ai.ntd97.mobile;

import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.graphics.Color;
import android.os.Bundle;
import android.text.InputType;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

public final class NtdPairedPcSettingsActivity extends Activity {
    private EditText peer;
    private EditText address;
    private EditText remotePeerId;
    private EditText remoteVerifyKey;
    private TextView pairedPeers;
    private TextView publicIdentity;
    private TextView status;
    private String copyableReceipt = "";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        ScrollView scroll = new ScrollView(this);
        LinearLayout content = new LinearLayout(this);
        content.setOrientation(LinearLayout.VERTICAL);
        content.setPadding(dp(16), dp(16), dp(16), dp(16));
        content.setBackgroundColor(Color.rgb(12, 12, 18));

        TextView heading = text("Paired PC");
        heading.setTextSize(20.0f);
        content.addView(heading);

        TextView help = text(
                "Pair a desktop PCF97 agent by pinning its public identity. "
                        + "NTD97 generates the phone signing identity locally and never shows its secret seed. "
                        + "Replacing an existing alias requires explicit revoke first.");
        help.setTextSize(14.0f);
        content.addView(help);

        peer = field("Alias, e.g. workstation");
        address = field("PC IP:port, e.g. 192.168.1.20:45970");
        remotePeerId = field("Desktop peer ID (32 hex chars)");
        remoteVerifyKey = field("Desktop Ed25519 verify key (64 hex chars)");
        address.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_URI);
        remotePeerId.setInputType(
                InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS);
        remoteVerifyKey.setInputType(
                InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS);

        content.addView(peer);
        content.addView(address);
        content.addView(remotePeerId);
        content.addView(remoteVerifyKey);

        LinearLayout primary = new LinearLayout(this);
        primary.setOrientation(LinearLayout.HORIZONTAL);

        Button provision = new Button(this);
        provision.setText("Provision");
        provision.setOnClickListener(view -> provision());
        primary.addView(
                provision,
                new LinearLayout.LayoutParams(
                        0,
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        1.0f));

        Button describe = new Button(this);
        describe.setText("Phone identity");
        describe.setOnClickListener(view -> describe());
        primary.addView(
                describe,
                new LinearLayout.LayoutParams(
                        0,
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        1.0f));

        Button revoke = new Button(this);
        revoke.setText("Revoke");
        revoke.setOnClickListener(view -> revoke());
        primary.addView(
                revoke,
                new LinearLayout.LayoutParams(
                        0,
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        1.0f));
        content.addView(primary);

        LinearLayout secondary = new LinearLayout(this);
        secondary.setOrientation(LinearLayout.HORIZONTAL);

        Button refresh = new Button(this);
        refresh.setText("Refresh");
        refresh.setOnClickListener(view -> refreshPairs());
        secondary.addView(
                refresh,
                new LinearLayout.LayoutParams(
                        0,
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        1.0f));

        Button copy = new Button(this);
        copy.setText("Copy public identity");
        copy.setOnClickListener(view -> copyPublicIdentity());
        secondary.addView(
                copy,
                new LinearLayout.LayoutParams(
                        0,
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        1.0f));

        Button close = new Button(this);
        close.setText("Close");
        close.setOnClickListener(view -> finish());
        secondary.addView(
                close,
                new LinearLayout.LayoutParams(
                        0,
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        1.0f));
        content.addView(secondary);

        pairedPeers = text("");
        pairedPeers.setTextSize(14.0f);
        content.addView(pairedPeers);

        publicIdentity = text("");
        publicIdentity.setTextSize(13.0f);
        publicIdentity.setTextIsSelectable(true);
        content.addView(publicIdentity);

        status = text("");
        status.setTextSize(14.0f);
        content.addView(status);

        scroll.addView(
                content,
                new ScrollView.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT));
        setContentView(scroll);
        refreshPairs();
    }

    private void provision() {
        String result = NtdNativeRuntimeHost.provisionPcPair(
                this,
                peer.getText().toString(),
                address.getText().toString(),
                remotePeerId.getText().toString(),
                remoteVerifyKey.getText().toString());
        if (isError(result)) {
            setError(result);
            return;
        }
        showReceipt(result);
        status.setText("Paired-PC profile provisioned. Share only the public phone identity with the desktop.");
        refreshPairs();
    }

    private void describe() {
        String result = NtdNativeRuntimeHost.describePcPair(
                this,
                peer.getText().toString());
        if (isError(result)) {
            setError(result);
            return;
        }
        showReceipt(result);
        status.setText("Public phone identity loaded.");
    }

    private void revoke() {
        String result = NtdNativeRuntimeHost.revokePcPair(
                this,
                peer.getText().toString());
        if (!"OK".equals(result)) {
            setError(result);
            return;
        }
        copyableReceipt = "";
        publicIdentity.setText("");
        status.setText("Paired-PC profile revoked.");
        refreshPairs();
    }

    private void refreshPairs() {
        String result = NtdNativeRuntimeHost.listPcPairs(this);
        if (isError(result)) {
            pairedPeers.setText("Paired aliases: unavailable");
            setError(result);
            return;
        }
        String normalized = result.trim();
        pairedPeers.setText(
                normalized.isEmpty()
                        ? "Paired aliases: none"
                        : "Paired aliases: " + normalized.replace('\n', ','));
    }

    private void showReceipt(String receipt) {
        if (receipt == null || !receipt.startsWith("NTD97_PC_PAIR_RECEIPT_V1\n")) {
            status.setText("Invalid native pairing receipt");
            copyableReceipt = "";
            publicIdentity.setText("");
            return;
        }
        copyableReceipt = receipt;
        publicIdentity.setText(receipt);
    }

    private void copyPublicIdentity() {
        if (copyableReceipt.isEmpty()) {
            status.setText("No public phone identity is loaded");
            return;
        }
        ClipboardManager clipboard =
                (ClipboardManager) getSystemService(CLIPBOARD_SERVICE);
        if (clipboard == null) {
            status.setText("Clipboard unavailable");
            return;
        }
        clipboard.setPrimaryClip(
                ClipData.newPlainText("NTD97 PCF97 public identity", copyableReceipt));
        status.setText("Public phone identity copied.");
    }

    private void setError(String result) {
        String message = result == null ? "unknown error" : result;
        if (message.startsWith("ERROR:")) {
            message = message.substring("ERROR:".length());
        }
        status.setText("Pairing error: " + message);
    }

    private static boolean isError(String result) {
        return result == null || result.startsWith("ERROR:");
    }

    private EditText field(String hint) {
        EditText view = new EditText(this);
        view.setHint(hint);
        view.setHintTextColor(Color.LTGRAY);
        view.setTextColor(Color.WHITE);
        view.setSingleLine(true);
        return view;
    }

    private TextView text(String value) {
        TextView view = new TextView(this);
        view.setText(value);
        view.setTextColor(Color.WHITE);
        view.setPadding(0, dp(6), 0, dp(6));
        return view;
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }
}
