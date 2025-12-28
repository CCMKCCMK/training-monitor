package com.training.monitor;

import android.app.Activity;
import android.graphics.Color;
import android.os.Bundle;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.util.Log;
import android.view.ViewGroup;

/**
 * Main activity for Training Monitor.
 * Connects to Python WebSocket server for real-time training charts.
 */
public class MainActivity extends Activity {

    private static final String TAG = "MainActivity";

    private LineChartView lossChart;
    private LineChartView energyChart;
    private PredictionView predictionView;
    private WebSocketClient wsClient;
    private TextView statusText;
    private EditText hostInput;
    private EditText portInput;
    private Button connectButton;
    private Button disconnectButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Create layout
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setPadding(16, 16, 16, 16);

        // Connection controls
        LinearLayout connLayout = new LinearLayout(this);
        connLayout.setOrientation(LinearLayout.HORIZONTAL);
        connLayout.setLayoutParams(new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ));

        hostInput = new EditText(this);
        hostInput.setHint("Host");
        hostInput.setText("127.0.0.1");
        hostInput.setLayoutParams(new LinearLayout.LayoutParams(
            0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f
        ));
        connLayout.addView(hostInput);

        portInput = new EditText(this);
        portInput.setHint("Port");
        portInput.setText("8765");
        portInput.setInputType(android.text.InputType.TYPE_CLASS_NUMBER);
        portInput.setLayoutParams(new LinearLayout.LayoutParams(
            150, ViewGroup.LayoutParams.WRAP_CONTENT
        ));
        connLayout.addView(portInput);

        layout.addView(connLayout);

        LinearLayout buttonLayout = new LinearLayout(this);
        buttonLayout.setOrientation(LinearLayout.HORIZONTAL);

        connectButton = new Button(this);
        connectButton.setText("Connect");
        connectButton.setOnClickListener(v -> connectToServer());
        buttonLayout.addView(connectButton);

        disconnectButton = new Button(this);
        disconnectButton.setText("Disconnect");
        disconnectButton.setEnabled(false);
        disconnectButton.setOnClickListener(v -> disconnectFromServer());
        buttonLayout.addView(disconnectButton);

        layout.addView(buttonLayout);

        // Status text
        statusText = new TextView(this);
        statusText.setText("Training Monitor - Disconnected");
        statusText.setTextSize(14);
        statusText.setPadding(0, 16, 0, 16);
        layout.addView(statusText);

        // Loss chart
        lossChart = new LineChartView(this);
        lossChart.setTitle("Loss");
        lossChart.setLayoutParams(new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            300
        ));
        layout.addView(lossChart);

        // Energy chart
        energyChart = new LineChartView(this);
        energyChart.setTitle("Energy");
        energyChart.setLayoutParams(new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            300
        ));
        layout.addView(energyChart);

        // Prediction view
        predictionView = new PredictionView(this);
        predictionView.setLayoutParams(new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            200
        ));
        layout.addView(predictionView);

        setContentView(layout);

        // Setup WebSocket client
        wsClient = new WebSocketClient(new WebSocketClient.DataListener() {
            @Override
            public void onTrainingData(int step, float trainLoss, float valLoss,
                                      float trainEnergy, float valEnergy) {
                runOnUiThread(() -> {
                    lossChart.addPoint(trainLoss, valLoss);
                    energyChart.addPoint(trainEnergy, valEnergy);
                    statusText.setText(String.format("Step: %d | Loss: %.4f / %.4f",
                        step, trainLoss, valLoss));
                });
            }

            @Override
            public void onPredictionData(float[] actual, float[] predicted) {
                runOnUiThread(() -> {
                    predictionView.setData(actual, predicted);
                });
            }

            @Override
            public void onConnected() {
                runOnUiThread(() -> {
                    statusText.setText("Connected to " + wsClient.getServerUrl());
                    connectButton.setEnabled(false);
                    disconnectButton.setEnabled(true);
                    hostInput.setEnabled(false);
                    portInput.setEnabled(false);
                });
            }

            @Override
            public void onDisconnected() {
                runOnUiThread(() -> {
                    statusText.setText("Disconnected");
                    connectButton.setEnabled(true);
                    disconnectButton.setEnabled(false);
                    hostInput.setEnabled(true);
                    portInput.setEnabled(true);
                });
            }

            @Override
            public void onError(String error) {
                runOnUiThread(() -> {
                    statusText.setText("Error: " + error);
                });
            }
        });
    }

    private void connectToServer() {
        String host = hostInput.getText().toString().trim();
        String portStr = portInput.getText().toString().trim();

        if (host.isEmpty()) host = "127.0.0.1";
        int port = portStr.isEmpty() ? 8765 : Integer.parseInt(portStr);

        wsClient.setServerUrl(host, port);
        wsClient.connect();
        statusText.setText("Connecting to " + host + ":" + port + "...");
    }

    private void disconnectFromServer() {
        wsClient.disconnect();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (wsClient != null) {
            wsClient.disconnect();
        }
    }
}
