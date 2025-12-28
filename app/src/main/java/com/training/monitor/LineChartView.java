package com.training.monitor;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;
import java.util.ArrayList;
import java.util.List;

/**
 * Simple line chart view for training metrics.
 * Draws two lines: train (blue) and validation (red).
 */
public class LineChartView extends View {

    private Paint axisPaint;
    private Paint gridPaint;
    private Paint trainLinePaint;
    private Paint valLinePaint;
    private Paint textPaint;

    private List<Float> trainValues = new ArrayList<>();
    private List<Float> valValues = new ArrayList<>();
    private String title = "";
    private String label = "Loss";

    private int maxPoints = 100;

    public LineChartView(Context context) {
        super(context);
        init();
    }

    public LineChartView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        // Axis paint
        axisPaint = new Paint();
        axisPaint.setColor(Color.GRAY);
        axisPaint.setStrokeWidth(2f);

        // Grid paint
        gridPaint = new Paint();
        gridPaint.setColor(Color.LTGRAY);
        gridPaint.setStrokeWidth(1f);
        gridPaint.setAlpha(100);

        // Train line paint (blue)
        trainLinePaint = new Paint();
        trainLinePaint.setColor(Color.BLUE);
        trainLinePaint.setStrokeWidth(3f);
        trainLinePaint.setAntiAlias(true);

        // Validation line paint (red)
        valLinePaint = new Paint();
        valLinePaint.setColor(Color.RED);
        valLinePaint.setStrokeWidth(3f);
        valLinePaint.setAntiAlias(true);

        // Text paint
        textPaint = new Paint();
        textPaint.setColor(Color.BLACK);
        textPaint.setTextSize(32f);
        textPaint.setAntiAlias(true);
    }

    public void setTitle(String title) {
        this.title = title;
        postInvalidate();
    }

    public void setLabel(String label) {
        this.label = label;
        postInvalidate();
    }

    public void addPoint(float trainValue, float valValue) {
        trainValues.add(trainValue);
        valValues.add(valValue);

        if (trainValues.size() > maxPoints) {
            trainValues.remove(0);
            valValues.remove(0);
        }
        // Use invalidate() since we're called from UI thread
        invalidate();
    }

    public void clear() {
        trainValues.clear();
        valValues.clear();
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        int width = getWidth();
        int height = getHeight();
        int padding = 40;
        int chartWidth = width - 2 * padding;
        int chartHeight = height - 2 * padding - 40;

        // Draw background FIRST (so other layers show on top)
        canvas.drawColor(Color.WHITE);

        // Draw title
        canvas.drawText(title, padding, 35, textPaint);

        if (trainValues.isEmpty()) {
            canvas.drawText("No data", width / 2f - 50, height / 2f, textPaint);
            return;
        }

        // Find min/max for scaling
        float minVal = Float.MAX_VALUE;
        float maxVal = -Float.MAX_VALUE;
        for (float v : trainValues) {
            minVal = Math.min(minVal, v);
            maxVal = Math.max(maxVal, v);
        }
        for (float v : valValues) {
            minVal = Math.min(minVal, v);
            maxVal = Math.max(maxVal, v);
        }

        // Add some padding to range
        float range = maxVal - minVal;
        if (range < 0.001f) range = 1f;
        minVal -= range * 0.1f;
        maxVal += range * 0.1f;
        range = maxVal - minVal;

        // Draw grid lines (horizontal)
        for (int i = 0; i <= 4; i++) {
            float y = padding + 40 + (chartHeight * i / 4f);
            canvas.drawLine(padding, (int) y, width - padding, (int) y, gridPaint);

            // Draw value labels
            float value = maxVal - (range * i / 4f);
            String labelStr = String.format("%.3f", value);
            canvas.drawText(labelStr, 5, y + 10, textPaint);
        }

        // Draw axes
        canvas.drawLine(padding, padding + 40, padding, padding + 40 + chartHeight, axisPaint);
        canvas.drawLine(padding, padding + 40 + chartHeight, width - padding, padding + 40 + chartHeight, axisPaint);

        // Draw label
        canvas.drawText(label, padding, height - 10, textPaint);

        // Draw lines
        if (trainValues.size() > 1) {
            // Train line
            for (int i = 1; i < trainValues.size(); i++) {
                float x1 = padding + (chartWidth * (i - 1) / (maxPoints - 1f));
                float x2 = padding + (chartWidth * i / (maxPoints - 1f));
                float y1 = padding + 40 + chartHeight * (1 - (trainValues.get(i - 1) - minVal) / range);
                float y2 = padding + 40 + chartHeight * (1 - (trainValues.get(i) - minVal) / range);
                canvas.drawLine(x1, y1, x2, y2, trainLinePaint);
            }

            // Validation line
            for (int i = 1; i < valValues.size(); i++) {
                float x1 = padding + (chartWidth * (i - 1) / (maxPoints - 1f));
                float x2 = padding + (chartWidth * i / (maxPoints - 1f));
                float y1 = padding + 40 + chartHeight * (1 - (valValues.get(i - 1) - minVal) / range);
                float y2 = padding + 40 + chartHeight * (1 - (valValues.get(i) - minVal) / range);
                canvas.drawLine(x1, y1, x2, y2, valLinePaint);
            }
        }

        // Draw current value
        if (!trainValues.isEmpty()) {
            String current = String.format("Train: %.4f | Val: %.4f",
                trainValues.get(trainValues.size() - 1),
                valValues.get(valValues.size() - 1));
            canvas.drawText(current, padding, padding + 30, textPaint);
        }
    }
}
