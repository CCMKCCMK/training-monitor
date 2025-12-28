package com.training.monitor;

import android.app.Activity;
import android.os.Bundle;
import android.widget.LinearLayout;
import android.widget.TextView;

/**
 * Main activity for Training Monitor.
 * Displays real-time training charts.
 */
public class MainActivity extends Activity {

    private LineChartView lossChart;
    private LineChartView energyChart;
    private PredictionView predictionView;
    private DataFileWatcher fileWatcher;
    private TextView statusText;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Create layout programmatically (simple)
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setPadding(16, 16, 16, 16);

        // Status text
        statusText = new TextView(this);
        statusText.setText("Training Monitor - Waiting for data...");
        statusText.setTextSize(18);
        layout.addView(statusText);

        // Loss chart
        lossChart = new LineChartView(this);
        lossChart.setTitle("Loss");
        lossChart.setLayoutParams(new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            300
        ));
        layout.addView(lossChart);

        // Energy chart
        energyChart = new LineChartView(this);
        energyChart.setTitle("Energy");
        energyChart.setLayoutParams(new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            300
        ));
        layout.addView(energyChart);

        // Prediction view
        predictionView = new PredictionView(this);
        predictionView.setLayoutParams(new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            200
        ));
        layout.addView(predictionView);

        setContentView(layout);

        // Setup file watcher
        fileWatcher = new DataFileWatcher(this);
        fileWatcher.setCharts(lossChart, energyChart, predictionView);
    }

    @Override
    protected void onResume() {
        super.onResume();
        fileWatcher.start();
        statusText.setText("Training Monitor - Running...");
    }

    @Override
    protected void onPause() {
        super.onPause();
        fileWatcher.stop();
        statusText.setText("Training Monitor - Paused");
    }
}
