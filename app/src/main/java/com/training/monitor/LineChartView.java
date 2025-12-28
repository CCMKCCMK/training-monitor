package com.training.monitor;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;
import java.util.ArrayList;
import java.util.List;

/**
 * Interactive line chart view for training metrics.
 * Touch to see data point values.
 */
public class LineChartView extends View {

    private Paint axisPaint;
    private Paint gridPaint;
    private Paint trainLinePaint;
    private Paint valLinePaint;
    private Paint textPaint;
    private Paint tooltipPaint;
    private Paint tooltipBgPaint;

    private List<Float> trainValues = new ArrayList<>();
    private List<Float> valValues = new ArrayList<>();
    private String title = "";
    private String label = "Loss";

    private int maxPoints = 100;
    private int selectedIndex = -1;  // Currently selected data point

    // Chart dimensions
    private int padding = 40;
    private int chartWidth;
    private int chartHeight;
    private int chartTop;

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
        textPaint.setTextSize(28f);
        textPaint.setAntiAlias(true);

        // Tooltip text paint
        tooltipPaint = new Paint();
        tooltipPaint.setColor(Color.WHITE);
        tooltipPaint.setTextSize(26f);
        tooltipPaint.setAntiAlias(true);

        // Tooltip background paint
        tooltipBgPaint = new Paint();
        tooltipBgPaint.setColor(Color.parseColor("#CC000000"));
    }

    public void setTitle(String title) {
        this.title = title;
        invalidate();
    }

    public void setLabel(String label) {
        this.label = label;
        invalidate();
    }

    public void addPoint(float trainValue, float valValue) {
        trainValues.add(trainValue);
        valValues.add(valValue);

        if (trainValues.size() > maxPoints) {
            trainValues.remove(0);
            valValues.remove(0);
            if (selectedIndex >= 0) selectedIndex--;
        }
        // Use postInvalidate for reliable updates from any thread
        postInvalidate();
    }

    public void clear() {
        trainValues.clear();
        valValues.clear();
        selectedIndex = -1;
        postInvalidate();
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if (event.getAction() == MotionEvent.ACTION_DOWN ||
            event.getAction() == MotionEvent.ACTION_MOVE) {
            // Find closest data point
            float x = event.getX();
            float y = event.getY();

            // Check if touch is within chart area
            if (x >= padding && x <= getWidth() - padding &&
                y >= chartTop && y <= chartTop + chartHeight) {
                // Calculate closest index
                float relativeX = x - padding;
                int index = Math.round(relativeX * (trainValues.size() - 1) / chartWidth);
                index = Math.max(0, Math.min(index, trainValues.size() - 1));
                selectedIndex = index;
                invalidate();
                return true;
            }
        } else if (event.getAction() == MotionEvent.ACTION_UP ||
                   event.getAction() == MotionEvent.ACTION_CANCEL) {
            // Keep selection on tap up, but you could uncomment below to clear it
            // selectedIndex = -1;
            // invalidate();
        }
        return super.onTouchEvent(event);
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        chartWidth = w - 2 * padding;
        chartHeight = h - 2 * padding - 60;
        chartTop = padding + 50;
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        int width = getWidth();
        int height = getHeight();

        if (chartWidth == 0) {
            chartWidth = width - 2 * padding;
            chartHeight = height - 2 * padding - 60;
            chartTop = padding + 50;
        }

        // Draw background
        canvas.drawColor(Color.WHITE);

        // Draw title
        canvas.drawText(title, padding, 35, textPaint);

        if (trainValues.isEmpty()) {
            canvas.drawText("No data - Connect to server", width / 2f - 120, height / 2f, textPaint);
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
            float y = chartTop + (chartHeight * i / 4f);
            canvas.drawLine(padding, (int) y, width - padding, (int) y, gridPaint);

            // Draw value labels
            float value = maxVal - (range * i / 4f);
            String labelStr = String.format("%.3f", value);
            canvas.drawText(labelStr, 5, y + 10, textPaint);
        }

        // Draw axes
        canvas.drawLine(padding, chartTop, padding, chartTop + chartHeight, axisPaint);
        canvas.drawLine(padding, chartTop + chartHeight, width - padding, chartTop + chartHeight, axisPaint);

        // Draw label
        canvas.drawText(label, padding, height - 10, textPaint);

        // Draw lines
        if (trainValues.size() > 1) {
            // Train line (blue)
            for (int i = 1; i < trainValues.size(); i++) {
                float x1 = padding + (chartWidth * (i - 1) / (maxPoints - 1f));
                float x2 = padding + (chartWidth * i / (maxPoints - 1f));
                float y1 = chartTop + chartHeight * (1 - (trainValues.get(i - 1) - minVal) / range);
                float y2 = chartTop + chartHeight * (1 - (trainValues.get(i) - minVal) / range);
                canvas.drawLine(x1, y1, x2, y2, trainLinePaint);
            }

            // Validation line (red)
            for (int i = 1; i < valValues.size(); i++) {
                float x1 = padding + (chartWidth * (i - 1) / (maxPoints - 1f));
                float x2 = padding + (chartWidth * i / (maxPoints - 1f));
                float y1 = chartTop + chartHeight * (1 - (valValues.get(i - 1) - minVal) / range);
                float y2 = chartTop + chartHeight * (1 - (valValues.get(i) - minVal) / range);
                canvas.drawLine(x1, y1, x2, y2, valLinePaint);
            }
        }

        // Draw current value
        if (!trainValues.isEmpty()) {
            String current = String.format("Train: %.4f | Val: %.4f",
                trainValues.get(trainValues.size() - 1),
                valValues.get(valValues.size() - 1));
            canvas.drawText(current, padding, chartTop - 10, textPaint);
        }

        // Draw selection indicator and tooltip
        if (selectedIndex >= 0 && selectedIndex < trainValues.size()) {
            float x = padding + (chartWidth * selectedIndex / (maxPoints - 1f));
            float trainY = chartTop + chartHeight * (1 - (trainValues.get(selectedIndex) - minVal) / range);
            float valY = chartTop + chartHeight * (1 - (valValues.get(selectedIndex) - minVal) / range);

            // Draw vertical line at selection
            Paint selectionPaint = new Paint();
            selectionPaint.setColor(Color.parseColor("#80CCCCCC"));
            canvas.drawLine((int) x, chartTop, (int) x, chartTop + chartHeight, selectionPaint);

            // Draw circles at data points
            Paint circlePaint = new Paint();
            circlePaint.setColor(Color.BLUE);
            canvas.drawCircle(x, trainY, 8, circlePaint);
            circlePaint.setColor(Color.RED);
            canvas.drawCircle(x, valY, 8, circlePaint);

            // Draw tooltip
            String tooltipText = String.format("Step: %d\nTrain: %.4f\nVal: %.4f",
                selectedIndex, trainValues.get(selectedIndex), valValues.get(selectedIndex));
            drawTooltip(canvas, tooltipText, (int) x + 15, (int) Math.min(trainY, valY) - 10);
        }
    }

    private void drawTooltip(Canvas canvas, String text, int x, int y) {
        String[] lines = text.split("\n");
        float maxWidth = 0;
        float lineHeight = 32;
        float totalHeight = lines.length * lineHeight + 20;

        for (String line : lines) {
            float w = tooltipPaint.measureText(line);
            if (w > maxWidth) maxWidth = w;
        }

        // Position tooltip within bounds
        int left = x;
        int top = y;
        if (left + maxWidth + 20 > getWidth()) {
            left = (int) (x - maxWidth - 40);
        }
        if (top + totalHeight > getHeight()) {
            top = getHeight() - (int) totalHeight - 10;
        }
        if (top < 10) top = 10;

        // Draw tooltip background
        RectF bgRect = new RectF(left, top, left + maxWidth + 20, top + totalHeight);
        canvas.drawRoundRect(bgRect, 10, 10, tooltipBgPaint);

        // Draw text
        for (int i = 0; i < lines.length; i++) {
            canvas.drawText(lines[i], left + 10, top + 25 + i * lineHeight, tooltipPaint);
        }
    }
}
