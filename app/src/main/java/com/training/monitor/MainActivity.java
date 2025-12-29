package com.training.monitor;

import android.app.Activity;
import android.graphics.Bitmap;
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
 * - File logging for debugging
 */
public class MainActivity extends Activity {

    private static final String TAG = "MainActivity";
    private static final long UPDATE_INTERVAL_MS = 100; // 10 FPS

    // UI Components
    private LineChartView lossChart;
    private PredictionView predictionView;
    private VideoFrameView videoFrameView;
    private PerformanceGauge performanceGauge;
    private WebSocketClient wsClient;
    private TextView statusText;
    private TextView logPathText;
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

    // Frame data buffer for video understanding
    private Bitmap currentFrameBitmap = null;

    // Update loop
    private Handler uiHandler;
    private Runnable updateRunnable;
    private boolean updateLoopRunning = false;

    // Flags for new data
    private volatile boolean hasNewTrainingData = false;
    private volatile boolean hasNewPredictionData = false;
    private volatile boolean hasNewFrameData = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Initialize FileLogger
        FileLogger.init(this);
        FileLogger.getInstance().i(TAG, "MainActivity onCreate");

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

        // Log path text (small, gray)
        logPathText = new TextView(this);
        logPathText.setText("Log: " + FileLogger.getLogPath());
        logPathText.setTextSize(10);
        logPathText.setTextColor(0xFF888888);
        logPathText.setPadding(0, 0, 0, 4);
        layout.addView(logPathText);

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

        // Prediction view - increased height for time series
        predictionView = new PredictionView(this);
        predictionView.setLayoutParams(new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            280  // Increased from 180
        ));
        layout.addView(predictionView);

        // Video frame view - for video understanding tasks
        videoFrameView = new VideoFrameView(this);
        videoFrameView.setLayoutParams(new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            250
        ));
        layout.addView(videoFrameView);

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
            public void onFrameData(int frame, int total, float conf, String action,
                                   float actionConf, List<float[]> boxes) {
                // Legacy frame data - not used anymore, video frames sent as Bitmap
            }

            @Override
            public void onFrameImage(Bitmap frameImage) {
                // Update frame bitmap buffer
                synchronized (this) {
                    if (currentFrameBitmap != null && !currentFrameBitmap.isRecycled()) {
                        currentFrameBitmap.recycle();
                    }
                    currentFrameBitmap = frameImage;
                    hasNewFrameData = true;
                }
            }

            @Override
            public void onConnected() {
                FileLogger.getInstance().i(TAG, "UI: Connected");
                runOnUiThread(() -> {
                    statusText.setText("Connected to " + wsClient.getServerUrl());
                    connectButton.setEnabled(false);
                    disconnectButton.setEnabled(true);
                    hostInput.setEnabled(false);
                    portInput.setEnabled(false);

                    // Clear previous data
                    lossChart.clear();
                    predictionView.clear();
                    videoFrameView.clear();
                    currentStep = 0;
                    hasNewTrainingData = false;
                    hasNewPredictionData = false;
                    hasNewFrameData = false;

                    // Start update loop
                    startUpdateLoop();
                });
            }

            @Override
            public void onDisconnected() {
                FileLogger.getInstance().w(TAG, "UI: Disconnected");
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
                FileLogger.getInstance().e(TAG, "UI Error: " + error);
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

        if (hasNewFrameData) {
            synchronized (this) {
                if (currentFrameBitmap != null && !currentFrameBitmap.isRecycled()) {
                    videoFrameView.setFrameImage(currentFrameBitmap);
                }
            }
            hasNewFrameData = false;
        }
    }

    private void connectToServer() {
        String host = hostInput.getText().toString().trim();
        String portStr = portInput.getText().toString().trim();

        if (host.isEmpty()) host = "127.0.0.1";
        int port = portStr.isEmpty() ? 8766 : Integer.parseInt(portStr);

        wsClient.setServerUrl(host, port);
        statusText.setText("Connecting to " + host + ":" + port + "...");
        FileLogger.getInstance().i(TAG, "Connecting to " + host + ":" + port);
        wsClient.connect();
    }

    private void disconnectFromServer() {
        FileLogger.getInstance().i(TAG, "Disconnect requested");
        wsClient.disconnect();
        statusText.setText("Disconnected");
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        FileLogger.getInstance().i(TAG, "MainActivity onDestroy");
        stopUpdateLoop();
        if (wsClient != null) {
            wsClient.disconnect();
        }
    }
}
