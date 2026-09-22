package ai.ntd97.mobile;

import android.app.Activity;
import android.graphics.Color;
import android.os.Bundle;
import android.text.InputType;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

public final class NtdWebSearchSettingsActivity extends Activity {
    private EditText mode;
    private EditText endpoint;
    private EditText openSearchDescription;
    private EditText resultsPath;
    private EditText titlePath;
    private EditText urlPath;
    private EditText snippetPath;
    private EditText credentialHeader;
    private EditText credentialValue;
    private TextView status;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        ScrollView scroll = new ScrollView(this);
        LinearLayout content = new LinearLayout(this);
        content.setOrientation(LinearLayout.VERTICAL);
        content.setPadding(dp(16), dp(16), dp(16), dp(16));
        content.setBackgroundColor(Color.rgb(12, 12, 18));

        TextView heading = text("Provider-independent WebSearch");
        heading.setTextSize(20.0f);
        content.addView(heading);

        TextView help = text(
                "Choose Generic JSON mapping or standards-based OpenSearch discovery. "
                        + "No provider SDK or hostname is built into NTD97. "
                        + "An optional credential header stays in Android app-private configuration "
                        + "and is never copied into native evidence or cognition.");
        help.setTextSize(14.0f);
        content.addView(help);

        LinearLayout presets = new LinearLayout(this);
        presets.setOrientation(LinearLayout.HORIZONTAL);

        Button jsonPreset = new Button(this);
        jsonPreset.setText("Generic JSON");
        jsonPreset.setOnClickListener(view -> applyJsonPreset());
        presets.addView(
                jsonPreset,
                new LinearLayout.LayoutParams(
                        0,
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        1.0f));

        Button openSearchPreset = new Button(this);
        openSearchPreset.setText("OpenSearch");
        openSearchPreset.setOnClickListener(view -> applyOpenSearchPreset());
        presets.addView(
                openSearchPreset,
                new LinearLayout.LayoutParams(
                        0,
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        1.0f));
        content.addView(presets);

        mode = field("Mode: json or opensearch");
        endpoint = field("JSON endpoint template, e.g. https://host/search?q={query}&n={count}");
        endpoint.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_URI);
        openSearchDescription = field("OpenSearch description URL, e.g. https://host/opensearch.xml");
        openSearchDescription.setInputType(
                InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_URI);
        resultsPath = field("JSON results array path, e.g. items");
        titlePath = field("JSON title path, e.g. title");
        urlPath = field("JSON URL path, e.g. url");
        snippetPath = field("JSON snippet path (optional), e.g. description");
        credentialHeader = field("Credential header (optional), e.g. Authorization");
        credentialValue = field("Credential value (optional)");
        credentialValue.setInputType(
                InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_PASSWORD);

        content.addView(mode);
        content.addView(endpoint);
        content.addView(openSearchDescription);
        content.addView(resultsPath);
        content.addView(titlePath);
        content.addView(urlPath);
        content.addView(snippetPath);
        content.addView(credentialHeader);
        content.addView(credentialValue);

        LinearLayout buttons = new LinearLayout(this);
        buttons.setOrientation(LinearLayout.HORIZONTAL);

        Button save = new Button(this);
        save.setText("Save");
        save.setOnClickListener(view -> saveConfiguration());
        buttons.addView(
                save,
                new LinearLayout.LayoutParams(
                        0,
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        1.0f));

        Button clear = new Button(this);
        clear.setText("Clear");
        clear.setOnClickListener(view -> clearConfiguration());
        buttons.addView(
                clear,
                new LinearLayout.LayoutParams(
                        0,
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        1.0f));

        Button close = new Button(this);
        close.setText("Close");
        close.setOnClickListener(view -> finish());
        buttons.addView(
                close,
                new LinearLayout.LayoutParams(
                        0,
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        1.0f));

        content.addView(buttons);

        status = text("");
        content.addView(status);

        scroll.addView(
                content,
                new ScrollView.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT));
        setContentView(scroll);
        loadConfiguration();
    }

    private void loadConfiguration() {
        NtdWebPlatform.SearchConfiguration configuration =
                NtdWebPlatform.searchConfiguration();
        mode.setText(configuration.mode);
        endpoint.setText(configuration.endpointTemplate);
        openSearchDescription.setText(configuration.openSearchDescription);
        resultsPath.setText(configuration.resultsPath);
        titlePath.setText(configuration.titlePath);
        urlPath.setText(configuration.urlPath);
        snippetPath.setText(configuration.snippetPath);
        credentialHeader.setText(configuration.credentialHeaderName);
        credentialValue.setText(configuration.credentialHeaderValue);
        status.setText(configuration.configured()
                ? "WebSearch configuration loaded"
                : "WebSearch is disabled until configured");
    }

    private void applyJsonPreset() {
        mode.setText(NtdWebPlatform.SEARCH_MODE_JSON);
        openSearchDescription.setText("");
        if (resultsPath.getText().toString().trim().isEmpty()) {
            resultsPath.setText("items");
        }
        if (titlePath.getText().toString().trim().isEmpty()) {
            titlePath.setText("title");
        }
        if (urlPath.getText().toString().trim().isEmpty()) {
            urlPath.setText("url");
        }
        if (snippetPath.getText().toString().trim().isEmpty()) {
            snippetPath.setText("snippet");
        }
        status.setText("Generic JSON preset selected; enter a public HTTPS endpoint.");
    }

    private void applyOpenSearchPreset() {
        mode.setText(NtdWebPlatform.SEARCH_MODE_OPENSEARCH);
        endpoint.setText("");
        resultsPath.setText("");
        titlePath.setText("");
        urlPath.setText("");
        snippetPath.setText("");
        status.setText("OpenSearch preset selected; enter a public HTTPS description URL.");
    }

    private void saveConfiguration() {
        boolean saved = NtdWebPlatform.configureSearchProfile(
                this,
                mode.getText().toString(),
                endpoint.getText().toString(),
                openSearchDescription.getText().toString(),
                resultsPath.getText().toString(),
                titlePath.getText().toString(),
                urlPath.getText().toString(),
                snippetPath.getText().toString(),
                credentialHeader.getText().toString(),
                credentialValue.getText().toString());
        status.setText(saved
                ? "WebSearch profile saved"
                : "Invalid WebSearch mode, HTTPS endpoint, mapping, or credential header");
    }

    private void clearConfiguration() {
        boolean cleared = NtdWebPlatform.configureSearchProfile(
                this,
                NtdWebPlatform.SEARCH_MODE_JSON,
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "");
        if (cleared) {
            mode.setText(NtdWebPlatform.SEARCH_MODE_JSON);
            endpoint.setText("");
            openSearchDescription.setText("");
            resultsPath.setText("");
            titlePath.setText("");
            urlPath.setText("");
            snippetPath.setText("");
            credentialHeader.setText("");
            credentialValue.setText("");
        }
        status.setText(cleared ? "WebSearch disabled and credential cleared" : "Could not clear WebSearch");
    }

    private EditText field(String hint) {
        EditText field = new EditText(this);
        field.setHint(hint);
        field.setHintTextColor(Color.LTGRAY);
        field.setTextColor(Color.WHITE);
        field.setSingleLine(true);
        return field;
    }

    private TextView text(String value) {
        TextView view = new TextView(this);
        view.setText(value);
        view.setTextColor(Color.WHITE);
        return view;
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }
}
