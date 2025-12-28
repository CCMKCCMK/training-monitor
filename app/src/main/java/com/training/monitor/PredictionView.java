package com.training.monitor;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.DashPathEffect;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;
import java.util.ArrayList;
import java.util.List;

/**
 * Time series comparison view - shows actual vs predicted as line chart.
 * Ideal for model inference performance visualization.
 */
public class PredictionView extends View {

    // Colors - colorblind friendly
    private static final int COLOR_ACTUAL = Color.parseColor("#0077BB");     // Teal-blue
    private static final int COLOR_PREDICTED = Color.parseColor("#EE7733");  // Orange
    private static final int COLOR_GRID = Color.parseColor("#DDDDDD");       // Light gray
    private static final int COLOR_AXIS = Color.parseColor("#666666");      // Dark gray
    private static final int COLOR_TEXT = Color.parseColor("#333333");       // Near black
    private static final int COLOR_LEGEND_BG = Color.parseColor("#F0F0F0");  // Light background

    private Paint actualPaint;
    private Paint predictedPaint;
    private Paint gridPaint;
    private Paint axisPaint;
    private Paint textPaint;
    private Paint legendBgPaint;
    private Paint dotPaint;
    private Paint hollowCirclePaint;

    private List<Float> actualValues = new ArrayList<>();
    private List<Float> predictedValues = new ArrayList<>();

    private int padding = 50;
    private int chartWidth;
    private int chartHeight;
    private int chartTop;

    public PredictionView(Context context) {
        super(context);
        init();
    }

    public PredictionView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        // Actual line - solid teal-blue
        actualPaint = new Paint();
        actualPaint.setColor(COLOR_ACTUAL);
        actualPaint.setStrokeWidth(4f);
        actualPaint.setAntiAlias(true);
        actualPaint.setStyle(Paint.Style.STROKE);

        // Predicted line - dashed orange
        predictedPaint = new Paint();
        predictedPaint.setColor(COLOR_PREDICTED);
        predictedPaint.setStrokeWidth(4f);
        predictedPaint.setAntiAlias(true);
        predictedPaint.setStyle(Paint.Style.STROKE);
        predictedPaint.setPathEffect(new DashPathEffect(new float[]{10, 5}, 0));

        // Grid lines
        gridPaint = new Paint();
        gridPaint.setColor(COLOR_GRID);
        gridPaint.setStrokeWidth(1f);

        // Axes
        axisPaint = new Paint();
        axisPaint.setColor(COLOR_AXIS);
        axisPaint.setStrokeWidth(2f);

        // Text
        textPaint = new Paint();
        textPaint.setColor(COLOR_TEXT);
        textPaint.setTextSize(24f);
        textPaint.setAntiAlias(true);

        // Legend background
        legendBgPaint = new Paint();
        legendBgPaint.setColor(COLOR_LEGEND_BG);
        legendBgPaint.setStyle(Paint.Style.FILL);
        legendBgPaint.setAlpha(200);

        // Filled dot for actual
        dotPaint = new Paint();
        dotPaint.setColor(COLOR_ACTUAL);
        dotPaint.setAntiAlias(true);

        // Hollow circle for predicted
        hollowCirclePaint = new Paint();
        hollowCirclePaint.setColor(COLOR_PREDICTED);
        hollowCirclePaint.setStrokeWidth(3f);
        hollowCirclePaint.setStyle(Paint.Style.STROKE);
        hollowCirclePaint.setAntiAlias(true);
    }

    public void setData(List<Float> actual, List<Float> predicted) {
        this.actualValues = actual;
        this.predictedValues = predicted;
        postInvalidate();
    }

    public void clear() {
        actualValues.clear();
        predictedValues.clear();
        postInvalidate();
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        chartWidth = w - 2 * padding;
        chartHeight = h - 2 * padding - 40;
        chartTop = padding + 30;
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        int width = getWidth();
        int height = getHeight();

        if (chartWidth == 0) {
            chartWidth = width - 2 * padding;
            chartHeight = height - 2 * padding - 40;
            chartTop = padding + 30;
        }

        // Draw background
        canvas.drawColor(Color.WHITE);

        // Draw title
        Paint titlePaint = new Paint(textPaint);
        titlePaint.setTextSize(28f);
        titlePaint.setFakeBoldText(true);
        canvas.drawText("Prediction vs Ground Truth", padding, 35, titlePaint);

        // Draw legend
        drawLegend(canvas, width, height);

        if (actualValues.isEmpty()) {
            canvas.drawText("No prediction data - Connect to server", width / 2f - 140, height / 2f, textPaint);
            return;
        }

        // Find min/max for scaling
        float minVal = Float.MAX_VALUE;
        float maxVal = -Float.MAX_VALUE;
        int n = Math.min(actualValues.size(), 8); // Show max 8 points
        for (int i = 0; i < n; i++) {
            minVal = Math.min(minVal, actualValues.get(i));
            minVal = Math.min(minVal, predictedValues.get(i));
            maxVal = Math.max(maxVal, actualValues.get(i));
            maxVal = Math.max(maxVal, predictedValues.get(i));
        }

        // Add padding to range
        float range = maxVal - minVal;
        if (range < 0.001f) range = 1f;
        minVal -= range * 0.1f;
        maxVal += range * 0.1f;
        range = maxVal - minVal;

        // Draw grid lines (horizontal)
        for (int i = 0; i <= 4; i++) {
            float y = chartTop + (chartHeight * i / 4f);
            canvas.drawLine(padding, (int) y, width - padding, (int) y, gridPaint);

            // Draw value labels
            float value = maxVal - (range * i / 4f);
            String labelStr = String.format("%.2f", value);
            canvas.drawText(labelStr, 5, y + 8, textPaint);
        }

        // Draw axes
        canvas.drawLine(padding, chartTop, padding, chartTop + chartHeight, axisPaint);
        canvas.drawLine(padding, chartTop + chartHeight, width - padding, chartTop + chartHeight, axisPaint);

        // Draw x-axis labels
        for (int i = 0; i < n; i++) {
            float x = padding + (chartWidth * i / Math.max(n - 1, 1));
            canvas.drawText(String.valueOf(i), (int) x - 5, chartTop + chartHeight + 30, textPaint);
        }

        // Draw lines
        if (n > 1) {
            // Actual line (solid)
            Path actualPath = new Path();
            float firstX = padding;
            float firstY = chartTop + chartHeight * (1 - (actualValues.get(0) - minVal) / range);
            actualPath.moveTo(firstX, firstY);
            for (int i = 1; i < n; i++) {
                float x = padding + (chartWidth * i / (n - 1f));
                float y = chartTop + chartHeight * (1 - (actualValues.get(i) - minVal) / range);
                actualPath.lineTo(x, y);
            }
            canvas.drawPath(actualPath, actualPaint);

            // Predicted line (dashed)
            Path predictedPath = new Path();
            firstY = chartTop + chartHeight * (1 - (predictedValues.get(0) - minVal) / range);
            predictedPath.moveTo(firstX, firstY);
            for (int i = 1; i < n; i++) {
                float x = padding + (chartWidth * i / (n - 1f));
                float y = chartTop + chartHeight * (1 - (predictedValues.get(i) - minVal) / range);
                predictedPath.lineTo(x, y);
            }
            canvas.drawPath(predictedPath, predictedPaint);
        }

        // Draw data points
        for (int i = 0; i < n; i++) {
            float x = padding + (chartWidth * i / Math.max(n - 1, 1));
            float actualY = chartTop + chartHeight * (1 - (actualValues.get(i) - minVal) / range);
            float predictedY = chartTop + chartHeight * (1 - (predictedValues.get(i) - minVal) / range);

            // Actual - filled dot
            canvas.drawCircle(x, actualY, 7, dotPaint);

            // Predicted - hollow circle
            canvas.drawCircle(x, predictedY, 7, hollowCirclePaint);
        }
    }

    private void drawLegend(Canvas canvas, int width, int height) {
        int legendX = width - 200;
        int legendY = 15;
        int legendWidth = 190;
        int legendHeight = 55;

        // Background
        RectF bgRect = new RectF(legendX, legendY, legendX + legendWidth, legendY + legendHeight);
        canvas.drawRoundRect(bgRect, 8, 8, legendBgPaint);

        // Actual - solid line with dot
        canvas.drawLine(legendX + 15, legendY + 20, legendX + 45, legendY + 20, actualPaint);
        canvas.drawCircle(legendX + 30, legendY + 20, 5, dotPaint);
        canvas.drawText("Actual", legendX + 55, legendY + 27, textPaint);

        // Predicted - dashed line with hollow circle
        canvas.drawLine(legendX + 15, legendY + 42, legendX + 45, legendY + 42, predictedPaint);
        canvas.drawCircle(legendX + 30, legendY + 42, 5, hollowCirclePaint);
        canvas.drawText("Predicted", legendX + 55, legendY + 49, textPaint);
    }
}
