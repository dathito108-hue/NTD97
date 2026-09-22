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
    private EditText endpoint;
    private EditText resultsPath;
    private EditText titlePath;
    private EditText urlPath;
    private EditText snippetPath;
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
                "Configure a public HTTPS JSON endpoint. {query} is required and {count} is optional. "
                        + "Field paths are dot-separated JSON object keys. No provider is built into NTD97.");
        help.setTextSize(14.0f);
        content.addView(help);

        endpoint = field("Endpoint template, e.g. https://host/search?q={query}&n={count}");
        endpoint.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_URI);
        resultsPath = field("Results array path, e.g. items");
        titlePath = field("Title path, e.g. title");
        urlPath = field("URL path, e.g. url");
        snippetPath = field("Snippet path (optional), e.g. description");

        content.addView(endpoint);
        content.addView(resultsPath);
        content.addView(titlePath);
        content.addView(urlPath);
        content.addView(snippetPath);

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
        endpoint.setText(configuration.endpointTemplate);
        resultsPath.setText(configuration.resultsPath);
        titlePath.setText(configuration.titlePath);
        urlPath.setText(configuration.urlPath);
        snippetPath.setText(configuration.snippetPath);
        status.setText(configuration.configured()
                ? "WebSearch configuration loaded"
                : "WebSearch is disabled until configured");
    }

    private void saveConfiguration() {
        boolean saved = NtdWebPlatform.configureSearchProvider(
                this,
                endpoint.getText().toString(),
                resultsPath.getText().toString(),
                titlePath.getText().toString(),
                urlPath.getText().toString(),
                snippetPath.getText().toString());
        status.setText(saved
                ? "WebSearch configuration saved"
                : "Invalid HTTPS template or field mapping");
    }

    private void clearConfiguration() {
        boolean cleared = NtdWebPlatform.configureSearchProvider(
                this, "", "", "", "", "");
        if (cleared) {
            endpoint.setText("");
            resultsPath.setText("");
            titlePath.setText("");
            urlPath.setText("");
            snippetPath.setText("");
        }
        status.setText(cleared ? "WebSearch disabled" : "Could not clear WebSearch");
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
