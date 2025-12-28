package com.training.monitor;

import android.app.Activity;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.util.Log;
import android.view.ViewGroup;
import java.util.ArrayList;
import java.util.List;

/**
 * Main activity for Training Monitor.
 * Connects to Python WebSocket server for real-time training charts.
 *
 * Refactored for simpler, more reliable updates:
 * - Single Handler for all UI updates (batched at 10 FPS)
 * - WebSocket callbacks only update data variables
 * - Periodic update loop applies data to views
 */
public class MainActivity extends Activity {

    private static final String TAG = "MainActivity";
    private static final long UPDATE_INTERVAL_MS = 100; // 10 FPS

    // UI Components
    private LineChartView lossChart;
    private LineChartView energyChart;
    private PredictionView predictionView;
    private PerformanceGauge performanceGauge;
    private WebSocketClient wsClient;
    private TextView statusText;
    private EditText hostInput;
    private EditText portInput;
    private Button connectButton;
    private Button disconnectButton;

    // Data buffer - updated by WebSocket callbacks
    private int currentStep = 0;
    private float currentTrainLoss = 0;
    private float currentValLoss = 0;
    private float currentTrainEnergy = 0;
    private float currentValEnergy = 0;

    // Prediction data buffer
    private List<Float> predictionActual = new ArrayList<>();
    private List<Float> predictionPredicted = new ArrayList<>();

    // Update loop
    private Handler uiHandler;
    private Runnable updateRunnable;
    private boolean updateLoopRunning = false;

    // Flags for new data
    private volatile boolean hasNewTrainingData = false;
    private volatile boolean hasNewPredictionData = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        uiHandler = new Handler(Looper.getMainLooper());

        // Create layout
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setPadding(8, 8, 8, 8);

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
        hostInput.setTextSize(14);
        connLayout.addView(hostInput);

        portInput = new EditText(this);
        portInput.setHint("Port");
        portInput.setText("8766");
        portInput.setInputType(android.text.InputType.TYPE_CLASS_NUMBER);
        portInput.setLayoutParams(new LinearLayout.LayoutParams(
            120, ViewGroup.LayoutParams.WRAP_CONTENT
        ));
        portInput.setTextSize(14);
        connLayout.addView(portInput);

        layout.addView(connLayout);

        LinearLayout buttonLayout = new LinearLayout(this);
        buttonLayout.setOrientation(LinearLayout.HORIZONTAL);
        buttonLayout.setPadding(0, 4, 0, 4);

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
        statusText.setTextSize(12);
        statusText.setPadding(0, 4, 0, 4);
        layout.addView(statusText);

        // Performance gauge (top) - shows current status
        performanceGauge = new PerformanceGauge(this);
        performanceGauge.setLayoutParams(new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            220
        ));
        layout.addView(performanceGauge);

        // Loss chart
        lossChart = new LineChartView(this);
        lossChart.setTitle("Loss");
        lossChart.setLayoutParams(new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            250
        ));
        layout.addView(lossChart);

        // Energy chart
        energyChart = new LineChartView(this);
        energyChart.setTitle("Energy");
        energyChart.setLayoutParams(new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            250
        ));
        layout.addView(energyChart);

        // Prediction view
        predictionView = new PredictionView(this);
        predictionView.setLayoutParams(new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            180
        ));
        layout.addView(predictionView);

        setContentView(layout);

        // Setup periodic update loop
        updateRunnable = new Runnable() {
            @Override
            public void run() {
                updateAllViews();
                if (updateLoopRunning) {
                    uiHandler.postDelayed(this, UPDATE_INTERVAL_MS);
                }
            }
        };

        // Setup WebSocket client
        wsClient = new WebSocketClient(new WebSocketClient.DataListener() {
            @Override
            public void onTrainingData(int step, float trainLoss, float valLoss,
                                      float trainEnergy, float valEnergy) {
                // Just update data buffer - no UI calls here
                currentStep = step;
                currentTrainLoss = trainLoss;
                currentValLoss = valLoss;
                currentTrainEnergy = trainEnergy;
                currentValEnergy = valEnergy;
                hasNewTrainingData = true;
            }

            @Override
            public void onPredictionData(float[] actual, float[] predicted) {
                // Update prediction data buffer
                synchronized (predictionActual) {
                    predictionActual.clear();
                    predictionPredicted.clear();
                    for (float f : actual) predictionActual.add(f);
                    for (float f : predicted) predictionPredicted.add(f);
                    hasNewPredictionData = true;
                }
            }

            @Override
            public void onConnected() {
                runOnUiThread(() -> {
                    statusText.setText("Connected to " + wsClient.getServerUrl());
                    connectButton.setEnabled(false);
                    disconnectButton.setEnabled(true);
                    hostInput.setEnabled(false);
                    portInput.setEnabled(false);

                    // Clear previous data
                    lossChart.clear();
                    energyChart.clear();
                    predictionView.clear();
                    currentStep = 0;
                    hasNewTrainingData = false;
                    hasNewPredictionData = false;

                    // Start update loop
                    startUpdateLoop();
                });
            }

            @Override
            public void onDisconnected() {
                runOnUiThread(() -> {
                    connectButton.setEnabled(true);
                    disconnectButton.setEnabled(false);
                    hostInput.setEnabled(true);
                    portInput.setEnabled(true);
                    stopUpdateLoop();
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

    private void startUpdateLoop() {
        if (!updateLoopRunning) {
            updateLoopRunning = true;
            uiHandler.post(updateRunnable);
            Log.i(TAG, "Update loop started");
        }
    }

    private void stopUpdateLoop() {
        updateLoopRunning = false;
        uiHandler.removeCallbacks(updateRunnable);
        Log.i(TAG, "Update loop stopped");
    }

    private void updateAllViews() {
        // Apply all buffered data to views in a single batch
        if (hasNewTrainingData) {
            lossChart.addPoint(currentTrainLoss, currentValLoss);
            energyChart.addPoint(currentTrainEnergy, currentValEnergy);
            performanceGauge.updateData(currentStep, currentTrainLoss, currentValLoss,
                                       currentTrainEnergy, currentValEnergy);
            statusText.setText(String.format("Step: %d | Loss: %.4f / %.4f",
                currentStep, currentTrainLoss, currentValLoss));
            hasNewTrainingData = false;
        }

        if (hasNewPredictionData) {
            synchronized (predictionActual) {
                predictionView.setData(new ArrayList<>(predictionActual),
                                      new ArrayList<>(predictionPredicted));
            }
            hasNewPredictionData = false;
        }
    }

    private void connectToServer() {
        String host = hostInput.getText().toString().trim();
        String portStr = portInput.getText().toString().trim();

        if (host.isEmpty()) host = "127.0.0.1";
        int port = portStr.isEmpty() ? 8766 : Integer.parseInt(portStr);

        wsClient.setServerUrl(host, port);
        statusText.setText("Connecting to " + host + ":" + port + "...");
        wsClient.connect();
    }

    private void disconnectFromServer() {
        wsClient.disconnect();
        statusText.setText("Disconnected");
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        stopUpdateLoop();
        if (wsClient != null) {
            wsClient.disconnect();
        }
    }
}
