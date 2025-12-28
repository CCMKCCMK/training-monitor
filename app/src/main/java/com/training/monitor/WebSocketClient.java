package com.training.monitor;

import android.util.Log;
import okhttp3.*;
import org.json.JSONObject;
import java.util.concurrent.TimeUnit;

/**
 * WebSocket client for receiving training data.
 * Connects to Python WebSocket server.
 */
public class WebSocketClient {

    private static final String TAG = "WebSocketClient";
    private static final String DEFAULT_HOST = "127.0.0.1";
    private static final int DEFAULT_PORT = 8765;

    private OkHttpClient client;
    private WebSocket webSocket;
    private String serverUrl;
    private DataListener listener;
    private boolean isConnected = false;

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
        this.client = new OkHttpClient.Builder()
            .readTimeout(0, TimeUnit.MILLISECONDS)
            .build();
        this.serverUrl = "ws://" + DEFAULT_HOST + ":" + DEFAULT_PORT;
    }

    public void setServerUrl(String host, int port) {
        this.serverUrl = "ws://" + host + ":" + port;
    }

    public void connect() {
        Request request = new Request.Builder()
            .url(serverUrl)
            .build();

        webSocket = client.newWebSocket(request, new WebSocketListener() {
            @Override
            public void onOpen(WebSocket webSocket, Response response) {
                isConnected = true;
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
                isConnected = false;
                webSocket.close(1000, null);
                Log.i(TAG, "Closing: " + reason);
                if (listener != null) {
                    listener.onDisconnected();
                }
            }

            @Override
            public void onFailure(WebSocket webSocket, Throwable t, Response response) {
                isConnected = false;
                Log.e(TAG, "Error: " + t.getMessage());
                if (listener != null) {
                    listener.onError(t.getMessage());
                }
            }
        });
    }

    public void disconnect() {
        if (webSocket != null) {
            webSocket.close(1000, "Client closing");
        }
        isConnected = false;
    }

    public boolean isConnected() {
        return isConnected;
    }

    public String getServerUrl() {
        return serverUrl;
    }
}
