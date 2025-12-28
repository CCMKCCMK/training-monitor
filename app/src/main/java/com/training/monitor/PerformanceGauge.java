package com.training.monitor;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

/**
 * Performance gauge - shows current training status visually.
 * Displays circular progress for loss and energy.
 */
public class PerformanceGauge extends View {

    private Paint bgPaint;
    private Paint trainPaint;
    private Paint valPaint;
    private Paint textPaint;
    private Paint labelPaint;

    private float trainLoss = 0;
    private float valLoss = 0;
    private float trainEnergy = 0;
    private float valEnergy = 0;

    private int step = 0;
    private float maxLoss = 2.0f;
    private float minEnergy = -5.0f;

    public PerformanceGauge(Context context) {
        super(context);
        init();
    }

    public PerformanceGauge(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        bgPaint = new Paint();
        bgPaint.setColor(Color.LTGRAY);
        bgPaint.setStyle(Paint.Style.STROKE);
        bgPaint.setStrokeWidth(20f);
        bgPaint.setAntiAlias(true);

        trainPaint = new Paint();
        trainPaint.setColor(Color.BLUE);
        trainPaint.setStyle(Paint.Style.STROKE);
        trainPaint.setStrokeWidth(20f);
        trainPaint.setAntiAlias(true);

        valPaint = new Paint();
        valPaint.setColor(Color.RED);
        valPaint.setStyle(Paint.Style.STROKE);
        valPaint.setStrokeWidth(20f);
        valPaint.setAntiAlias(true);

        textPaint = new Paint();
        textPaint.setColor(Color.BLACK);
        textPaint.setTextSize(36f);
        textPaint.setAntiAlias(true);
        textPaint.setTextAlign(Paint.Align.CENTER);

        labelPaint = new Paint();
        labelPaint.setColor(Color.DKGRAY);
        labelPaint.setTextSize(22f);
        labelPaint.setAntiAlias(true);
        labelPaint.setTextAlign(Paint.Align.CENTER);
    }

    public void updateData(int step, float trainLoss, float valLoss,
                          float trainEnergy, float valEnergy) {
        this.step = step;
        this.trainLoss = trainLoss;
        this.valLoss = valLoss;
        this.trainEnergy = trainEnergy;
        this.valEnergy = valEnergy;
        postInvalidate();
    }

    public void setMaxLoss(float max) {
        this.maxLoss = max;
    }

    public void setMinEnergy(float min) {
        this.minEnergy = min;
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        int width = getWidth();
        int height = getHeight();
        int centerX = width / 2;
        int centerY = height / 2 - 20;

        // Draw title
        Paint titlePaint = new Paint();
        titlePaint.setColor(Color.BLACK);
        titlePaint.setTextSize(28f);
        titlePaint.setAntiAlias(true);
        titlePaint.setTextAlign(Paint.Align.CENTER);
        canvas.drawText("Current Performance", centerX, 30, titlePaint);

        if (step == 0) {
            canvas.drawText("No data", centerX, centerY, textPaint);
            return;
        }

        // Gauge radius
        float outerRadius = Math.min(width, height) / 2 - 40;
        float innerRadius = outerRadius - 40;

        // Draw loss gauge (outer ring)
        drawGaugeRing(canvas, centerX, centerY, outerRadius,
                     trainLoss, valLoss, maxLoss, true);

        // Draw energy gauge (inner ring)
        drawEnergyRing(canvas, centerX, centerY, innerRadius,
                      trainEnergy, valEnergy);

        // Draw center text
        String stepText = "Step: " + step;
        canvas.drawText(stepText, centerX, centerY + 15, textPaint);

        // Draw legend at bottom
        float legendY = height - 30;
        Paint legendPaint = new Paint();
        legendPaint.setTextSize(24f);
        legendPaint.setTextAlign(Paint.Align.LEFT);

        legendPaint.setColor(Color.BLUE);
        canvas.drawText("● Train", width / 4 - 40, legendY, legendPaint);

        legendPaint.setColor(Color.RED);
        canvas.drawText("● Val", width * 3 / 4 - 20, legendY, legendPaint);
    }

    private void drawGaugeRing(Canvas canvas, int cx, int cy, float radius,
                              float trainVal, float valVal, float max, boolean isLoss) {
        RectF rect = new RectF(cx - radius, cy - radius, cx + radius, cy + radius);

        // Background ring
        canvas.drawArc(rect, 180, 180, false, bgPaint);

        // Calculate angles (lower loss = better, so clockwise from right)
        float trainRatio = Math.min(trainVal / max, 1f);
        float valRatio = Math.min(valVal / max, 1f);

        float trainAngle = 180 * (1 - trainRatio);
        float valAngle = 180 * (1 - valRatio);

        // Train arc (blue)
        canvas.drawArc(rect, 180, trainAngle, false, trainPaint);

        // Val arc (red) - thinner, overlay
        Paint thinValPaint = new Paint(valPaint);
        thinValPaint.setStrokeWidth(12f);
        canvas.drawArc(rect, 180, valAngle, false, thinValPaint);

        // Draw labels
        String label = isLoss ? "Loss" : "";
        if (!label.isEmpty()) {
            labelPaint.setTextSize(20f);
            canvas.drawText(label, cx, cy - radius - 15, labelPaint);
        }
    }

    private void drawEnergyRing(Canvas canvas, int cx, int cy, float radius,
                                float trainEnergy, float valEnergy) {
        RectF rect = new RectF(cx - radius, cy - radius, cx + radius, cy + radius);

        // Background ring
        Paint thinBg = new Paint(bgPaint);
        thinBg.setStrokeWidth(12f);
        canvas.drawArc(rect, 180, 180, false, thinBg);

        // Energy is negative, lower is better (more negative = better)
        // Normalize: map from [minEnergy, 0] to [0, 1]
        float trainRatio = Math.min(Math.max((trainEnergy - minEnergy) / (-minEnergy), 0), 1);
        float valRatio = Math.min(Math.max((valEnergy - minEnergy) / (-minEnergy), 0), 1);

        float trainAngle = 180 * trainRatio;
        float valAngle = 180 * valRatio;

        // Train arc (green tint for energy)
        Paint energyTrainPaint = new Paint(trainPaint);
        energyTrainPaint.setColor(Color.parseColor("#4CAF50"));
        energyTrainPaint.setStrokeWidth(12f);
        canvas.drawArc(rect, 180, trainAngle, false, energyTrainPaint);

        // Val arc (orange tint)
        Paint energyValPaint = new Paint(valPaint);
        energyValPaint.setColor(Color.parseColor("#FF9800"));
        energyValPaint.setStrokeWidth(8f);
        canvas.drawArc(rect, 180, valAngle, false, energyValPaint);

        // Label
        labelPaint.setTextSize(18f);
        labelPaint.setTextAlign(Paint.Align.CENTER);
        canvas.drawText("Energy", cx, cy - radius - 15, labelPaint);
    }
}
