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
 * Prediction comparison view - shows actual vs predicted values as bar chart.
 */
public class PredictionView extends View {

    private Paint actualPaint;
    private Paint predictedPaint;
    private Paint textPaint;
    private Paint gridPaint;

    private List<Float> actualValues = new ArrayList<>();
    private List<Float> predictedValues = new ArrayList<>();

    public PredictionView(Context context) {
        super(context);
        init();
    }

    public PredictionView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        actualPaint = new Paint();
        actualPaint.setColor(Color.BLUE);
        actualPaint.setAlpha(150);

        predictedPaint = new Paint();
        predictedPaint.setColor(Color.RED);
        predictedPaint.setAlpha(150);

        textPaint = new Paint();
        textPaint.setColor(Color.BLACK);
        textPaint.setTextSize(24f);
        textPaint.setAntiAlias(true);

        gridPaint = new Paint();
        gridPaint.setColor(Color.LTGRAY);
        gridPaint.setStrokeWidth(1f);
        gridPaint.setAlpha(100);
    }

    public void setData(List<Float> actual, List<Float> predicted) {
        this.actualValues = actual;
        this.predictedValues = predicted;
        invalidate();
    }

    public void clear() {
        actualValues.clear();
        predictedValues.clear();
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        canvas.drawColor(Color.WHITE);

        int width = getWidth();
        int height = getHeight();
        int padding = 30;

        // Draw title
        canvas.drawText("Prediction vs Actual", padding, 25, textPaint);

        if (actualValues.isEmpty()) {
            canvas.drawText("No prediction data", width / 2f - 100, height / 2f, textPaint);
            return;
        }

        int n = Math.min(actualValues.size(), 16);
        int barWidth = (width - 2 * padding) / (n * 2 + 1);
        int chartHeight = height - padding * 2 - 30;

        // Find min/max
        float minVal = Float.MAX_VALUE;
        float maxVal = -Float.MAX_VALUE;
        for (int i = 0; i < n; i++) {
            minVal = Math.min(minVal, actualValues.get(i));
            minVal = Math.min(minVal, predictedValues.get(i));
            maxVal = Math.max(maxVal, actualValues.get(i));
            maxVal = Math.max(maxVal, predictedValues.get(i));
        }
        float range = maxVal - minVal;
        if (range < 0.001f) range = 1f;

        // Draw bars
        for (int i = 0; i < n; i++) {
            float actualVal = actualValues.get(i);
            float predVal = predictedValues.get(i);

            // Normalize to 0-1
            float actualHeight = chartHeight * (actualVal - minVal) / range;
            float predHeight = chartHeight * (predVal - minVal) / range;

            int x = padding + i * barWidth * 2;

            // Actual bar (blue)
            int actualTop = padding + 30 + (int)(chartHeight - actualHeight);
            canvas.drawRect(x, actualTop, x + barWidth - 2, padding + 30 + chartHeight, actualPaint);

            // Predicted bar (red)
            int predTop = padding + 30 + (int)(chartHeight - predHeight);
            canvas.drawRect(x + barWidth, predTop, x + barWidth * 2 - 2, padding + 30 + chartHeight, predictedPaint);
        }

        // Draw baseline
        canvas.drawLine(padding, padding + 30 + chartHeight, width - padding, padding + 30 + chartHeight, gridPaint);

        // Legend
        canvas.drawRect(width - 150, 10, width - 130, 30, actualPaint);
        canvas.drawText("Actual", width - 125, 28, textPaint);
        canvas.drawRect(width - 70, 10, width - 50, 30, predictedPaint);
        canvas.drawText("Pred", width - 45, 28, textPaint);
    }
}
