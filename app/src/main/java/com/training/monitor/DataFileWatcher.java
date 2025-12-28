package com.training.monitor;

import android.content.Context;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

/**
 * Watches the training data file and updates the UI.
 * Uses app-specific external storage: /storage/emulated/0/Android/data/com.training.monitor/files/
 * Accessible from Termux via: ~/storage/android/data/com.training.monitor/files/
 */
public class DataFileWatcher {

    private static final String TAG = "DataFileWatcher";
    private static String TRAINING_FILE;
    private static String PREDICTION_FILE;

    public static void initPaths(Context context) {
        File appDir = context.getExternalFilesDir(null);
        if (appDir != null) {
            TRAINING_FILE = new File(appDir, "training_live.json").getAbsolutePath();
            PREDICTION_FILE = new File(appDir, "training_prediction.json").getAbsolutePath();
            Log.i(TAG, "Data path: " + appDir.getAbsolutePath());
        } else {
            TRAINING_FILE = "/data/data/com.training.monitor/files/training_live.json";
            PREDICTION_FILE = "/data/data/com.training.monitor/files/training_prediction.json";
            Log.w(TAG, "Using fallback path");
        }
    }

    private Handler handler;
    private Runnable watcherRunnable;
    private LineChartView lossChart;
    private LineChartView energyChart;
    private PredictionView predictionView;

    private long lastModified = 0;
    private long predLastModified = 0;

    public DataFileWatcher(Context context) {
        this.handler = new Handler(Looper.getMainLooper());
    }

    public void setCharts(LineChartView lossChart, LineChartView energyChart, PredictionView predictionView) {
        this.lossChart = lossChart;
        this.energyChart = energyChart;
        this.predictionView = predictionView;
    }

    public void start() {
        watcherRunnable = new Runnable() {
            @Override
            public void run() {
                checkAndUpdate();
                handler.postDelayed(this, 100);  // Check every 100ms
            }
        };
        handler.post(watcherRunnable);
    }

    public void stop() {
        if (watcherRunnable != null) {
            handler.removeCallbacks(watcherRunnable);
        }
    }

    private void checkAndUpdate() {
        // Check training data file
        File trainingFile = new File(TRAINING_FILE);
        if (trainingFile.exists()) {
            long modified = trainingFile.lastModified();
            if (modified > lastModified) {
                lastModified = modified;
                loadTrainingData(trainingFile);
            }
        }

        // Check prediction file
        File predFile = new File(PREDICTION_FILE);
        if (predFile.exists()) {
            long modified = predFile.lastModified();
            if (modified > predLastModified) {
                predLastModified = modified;
                loadPredictionData(predFile);
            }
        }
    }

    private void loadTrainingData(File file) {
        try {
            StringBuilder content = new StringBuilder();
            char[] buffer = new char[1024];
            FileReader reader = new FileReader(file);
            int n;
            while ((n = reader.read(buffer)) > 0) {
                content.append(buffer, 0, n);
            }
            reader.close();

            JSONObject json = new JSONObject(content.toString());

            int step = json.getInt("step");
            float trainLoss = (float) json.getDouble("train_loss");
            float valLoss = (float) json.getDouble("val_loss");
            float trainEnergy = (float) json.getDouble("train_energy");
            float valEnergy = (float) json.getDouble("val_energy");

            if (lossChart != null) {
                lossChart.addPoint(trainLoss, valLoss);
            }
            if (energyChart != null) {
                energyChart.addPoint(trainEnergy, valEnergy);
            }

        } catch (Exception e) {
            // Ignore parse errors during file writes
        }
    }

    private void loadPredictionData(File file) {
        try {
            StringBuilder content = new StringBuilder();
            char[] buffer = new char[1024];
            FileReader reader = new FileReader(file);
            int n;
            while ((n = reader.read(buffer)) > 0) {
                content.append(buffer, 0, n);
            }
            reader.close();

            JSONObject json = new JSONObject(content.toString());

            JSONArray actualArr = json.getJSONArray("actual");
            JSONArray predArr = json.getJSONArray("predicted");

            List<Float> actual = new ArrayList<>();
            List<Float> predicted = new ArrayList<>();

            for (int i = 0; i < Math.min(actualArr.length(), predArr.length()); i++) {
                actual.add((float) actualArr.getDouble(i));
                predicted.add((float) predArr.getDouble(i));
            }

            if (predictionView != null && !actual.isEmpty()) {
                predictionView.setData(actual, predicted);
            }

        } catch (Exception e) {
            // Ignore
        }
    }
}
