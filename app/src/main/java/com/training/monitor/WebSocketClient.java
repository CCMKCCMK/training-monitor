package com.training.monitor;

import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import okhttp3.*;
import org.json.JSONObject;
import java.util.concurrent.TimeUnit;

/**
 * WebSocket client for receiving training data.
 * Connects to Python WebSocket server.
 *
 * Refactored with automatic reconnection with exponential backoff.
 */
public class WebSocketClient {

    private static final String TAG = "WebSocketClient";
    private static final String DEFAULT_HOST = "127.0.0.1";
    private static final int DEFAULT_PORT = 8765;
    private static final int MAX_RECONNECT_DELAY_MS = 30000; // 30 seconds max
    private static final int INITIAL_RECONNECT_DELAY_MS = 1000; // 1 second

    private OkHttpClient client;
    private WebSocket webSocket;
    private String serverUrl;
    private DataListener listener;
    private boolean isConnected = false;
    private boolean shouldReconnect = false;

    // Reconnection handling
    private Handler reconnectHandler;
    private int reconnectAttempt = 0;
    private Runnable reconnectRunnable;

    public interface DataListener {
        void onTrainingData(int step, float trainLoss, float valLoss,
                           float trainEnergy, float valEnergy);
        void onPredictionData(float[] actual, float[] predicted);
        void onConnected();
        void onDisconnected();
        void onError(String error);
    }

    public WebSocketClient(DataListener listener) {
        this.listener = listener;
        this.reconnectHandler = new Handler(Looper.getMainLooper());

        this.client = new OkHttpClient.Builder()
            .readTimeout(0, TimeUnit.MILLISECONDS)
            .writeTimeout(10, TimeUnit.SECONDS)
            .pingInterval(30, TimeUnit.SECONDS)
            .retryOnConnectionFailure(true)
            .build();
        this.serverUrl = "ws://" + DEFAULT_HOST + ":" + DEFAULT_PORT;
    }

    public void setServerUrl(String host, int port) {
        this.serverUrl = "ws://" + host + ":" + port;
    }

    public String getServerUrl() {
        return serverUrl;
    }

    public void connect() {
        // Cancel any pending reconnect
        cancelReconnect();

        Request request = new Request.Builder()
            .url(serverUrl)
            .build();

        shouldReconnect = true;
        reconnectAttempt = 0;

        webSocket = client.newWebSocket(request, new WebSocketListener() {
            @Override
            public void onOpen(WebSocket webSocket, Response response) {
                isConnected = true;
                reconnectAttempt = 0; // Reset on successful connection
                Log.i(TAG, "Connected to " + serverUrl);
                if (listener != null) {
                    listener.onConnected();
                }
            }

            @Override
            public void onMessage(WebSocket webSocket, String text) {
                try {
                    JSONObject json = new JSONObject(text);
                    String type = json.optString("type", "training");

                    if (type.equals("training")) {
                        int step = json.getInt("step");
                        float trainLoss = (float) json.getDouble("train_loss");
                        float valLoss = (float) json.getDouble("val_loss");
                        float trainEnergy = (float) json.getDouble("train_energy");
                        float valEnergy = (float) json.getDouble("val_energy");

                        if (listener != null) {
                            listener.onTrainingData(step, trainLoss, valLoss,
                                trainEnergy, valEnergy);
                        }
                    } else if (type.equals("prediction")) {
                        org.json.JSONArray actualArr = json.getJSONArray("actual");
                        org.json.JSONArray predArr = json.getJSONArray("predicted");

                        float[] actual = new float[actualArr.length()];
                        float[] predicted = new float[predArr.length()];

                        for (int i = 0; i < actualArr.length(); i++) {
                            actual[i] = (float) actualArr.getDouble(i);
                            predicted[i] = (float) predArr.getDouble(i);
                        }

                        if (listener != null) {
                            listener.onPredictionData(actual, predicted);
                        }
                    }
                } catch (Exception e) {
                    Log.e(TAG, "Error parsing message: " + e.getMessage());
                }
            }

            @Override
            public void onClosing(WebSocket webSocket, int code, String reason) {
                Log.i(TAG, "Closing: " + reason);
            }

            @Override
            public void onClosed(WebSocket webSocket, int code, String reason) {
                isConnected = false;
                Log.i(TAG, "Connection closed: " + reason);
                if (listener != null) {
                    listener.onDisconnected();
                }
                // Schedule reconnect if appropriate
                scheduleReconnect();
            }

            @Override
            public void onFailure(WebSocket webSocket, Throwable t, Response response) {
                isConnected = false;
                String errorMsg = t.getMessage();
                if (errorMsg == null) errorMsg = "Unknown error";
                Log.e(TAG, "Connection error: " + errorMsg);

                if (listener != null) {
                    listener.onError(errorMsg);
                    listener.onDisconnected();
                }
                // Schedule reconnect
                scheduleReconnect();
            }
        });
    }

    /**
     * Schedule reconnection with exponential backoff.
     * Delay doubles each time up to MAX_RECONNECT_DELAY_MS.
     */
    private void scheduleReconnect() {
        if (!shouldReconnect) {
            Log.i(TAG, "Reconnect disabled, not scheduling");
            return;
        }

        // Calculate delay with exponential backoff
        long delay = Math.min(INITIAL_RECONNECT_DELAY_MS * (1L << reconnectAttempt),
                             MAX_RECONNECT_DELAY_MS);

        Log.i(TAG, String.format("Scheduling reconnect in %d ms (attempt %d)",
                                 delay, reconnectAttempt + 1));

        reconnectRunnable = new Runnable() {
            @Override
            public void run() {
                if (shouldReconnect && !isConnected) {
                    reconnectAttempt++;
                    Log.i(TAG, "Attempting reconnect #" + reconnectAttempt);
                    connect();
                }
            }
        };

        reconnectHandler.postDelayed(reconnectRunnable, delay);
    }

    /**
     * Cancel any pending reconnection attempt.
     */
    private void cancelReconnect() {
        if (reconnectRunnable != null) {
            reconnectHandler.removeCallbacks(reconnectRunnable);
            reconnectRunnable = null;
        }
    }

    public void disconnect() {
        Log.i(TAG, "Disconnect requested");
        shouldReconnect = false;
        cancelReconnect();
        reconnectAttempt = 0;

        if (webSocket != null) {
            try {
                webSocket.close(1000, "User closing");
            } catch (Exception e) {
                Log.e(TAG, "Error closing WebSocket: " + e.getMessage());
            }
            webSocket = null;
        }
        isConnected = false;
    }

    public boolean isConnected() {
        return isConnected && webSocket != null;
    }
}
