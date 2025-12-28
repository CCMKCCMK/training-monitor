package com.training.monitor;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Video frame view - displays detected objects, actions, and frame-level metrics.
 * Ideal for video understanding, object detection, action recognition tasks.
 */
public class VideoFrameView extends View {

    // Colors
    private static final int COLOR_BG = Color.BLACK;
    private static final int COLOR_GRID = Color.parseColor("#333333");
    private static final int COLOR_BOX_PERSON = Color.parseColor("#00FF00");      // Green
    private static final int COLOR_BOX_CAR = Color.parseColor("#0080FF");        // Blue
    private static final int COLOR_BOX_ANIMAL = Color.parseColor("#FF8000");     // Orange
    private static final int COLOR_BOX_DEFAULT = Color.parseColor("#FF00FF");    // Magenta
    private static final int COLOR_TEXT = Color.WHITE;
    private static final int COLOR_TEXT_BG = Color.parseColor("#AA000000");

    // Paint objects
    private Paint bgPaint;
    private Paint gridPaint;
    private Paint boxPaint;
    private Paint labelPaint;
    private Paint textPaint;
    private Paint confidencePaint;
    private Paint actionPaint;

    // Frame data
    private int currentFrame = 0;
    private int totalFrames = 0;
    private float frameConfidence = 0;
    private String actionLabel = "";
    private float actionConfidence = 0;

    // Detected objects
    private static class DetectedObject {
        String label;
        float confidence;
        float x, y, width, height;
        int color;

        DetectedObject(String label, float confidence, float x, float y, float w, float h, int color) {
            this.label = label;
            this.confidence = confidence;
            this.x = x;
            this.y = y;
            this.width = w;
            this.height = h;
            this.color = color;
        }
    }

    private List<DetectedObject> objects = new ArrayList<>();

    // Tracking info
    private Map<Integer, Integer> objectTracks = new HashMap<>(); // trackId -> color
    private int nextTrackId = 1;
    private int[] trackColors = {
        Color.parseColor("#FF0000"), Color.parseColor("#00FF00"),
        Color.parseColor("#0000FF"), Color.parseColor("#FFFF00"),
        Color.parseColor("#FF00FF"), Color.parseColor("#00FFFF"),
        Color.parseColor("#FFA500"), Color.parseColor("#FFC0CB")
    };

    public VideoFrameView(Context context) {
        super(context);
        init();
    }

    public VideoFrameView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        bgPaint = new Paint();
        bgPaint.setColor(COLOR_BG);

        gridPaint = new Paint();
        gridPaint.setColor(COLOR_GRID);
        gridPaint.setStrokeWidth(1f);

        boxPaint = new Paint();
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(3f);

        labelPaint = new Paint();
        labelPaint.setColor(COLOR_TEXT);
        labelPaint.setTextSize(24f);
        labelPaint.setAntiAlias(true);

        textPaint = new Paint();
        textPaint.setColor(COLOR_TEXT);
        textPaint.setTextSize(20f);
        textPaint.setAntiAlias(true);

        confidencePaint = new Paint();
        confidencePaint.setTextSize(32f);
        confidencePaint.setAntiAlias(true);

        actionPaint = new Paint();
        actionPaint.setTextSize(28f);
        actionPaint.setFakeBoldText(true);
        actionPaint.setAntiAlias(true);
    }

    public void setFrameData(int frame, int total, float conf, String action, float actionConf,
                            List<DetectedObject> objs) {
        this.currentFrame = frame;
        this.totalFrames = total;
        this.frameConfidence = conf;
        this.actionLabel = action;
        this.actionConfidence = actionConf;
        this.objects = objs != null ? objs : new ArrayList<>();
        postInvalidate();
    }

    /**
     * Simple frame data for video understanding simulation.
     * frame: current frame number
     * conf: overall frame confidence
     * action: predicted action label
     * actionConf: action confidence
     * boxes: array of [label, confidence, x, y, w, h] for each detection
     */
    public void setFrameDataSimple(int frame, int total, float conf, String action, float actionConf,
                                   List<float[]> boxes) {
        this.currentFrame = frame;
        this.totalFrames = total;
        this.frameConfidence = conf;
        this.actionLabel = action;
        this.actionConfidence = actionConf;
        this.objects = new ArrayList<>();

        if (boxes != null) {
            for (float[] box : boxes) {
                String label = String.valueOf((int) box[0]);
                if (box.length >= 6) {
                    label = getLabelForClass((int) box[0]);
                }
                int color = getColorForClass((int) box[0]);
                this.objects.add(new DetectedObject(label, box[1], box[2], box[3], box[4], box[5], color));
            }
        }
        postInvalidate();
    }

    private String getLabelForClass(int classId) {
        String[] labels = {"person", "car", "animal", "object"};
        return labels[classId % labels.length];
    }

    private int getColorForClass(int classId) {
        int[] colors = {COLOR_BOX_PERSON, COLOR_BOX_CAR, COLOR_BOX_ANIMAL, COLOR_BOX_DEFAULT};
        return colors[classId % colors.length];
    }

    public void clear() {
        this.objects.clear();
        this.currentFrame = 0;
        this.actionLabel = "";
        postInvalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        int width = getWidth();
        int height = getHeight();

        // Background
        canvas.drawColor(COLOR_BG);

        // Draw grid
        for (int i = 1; i < 4; i++) {
            float x = width * i / 4f;
            canvas.drawLine((int) x, 0, (int) x, height, gridPaint);
        }
        for (int i = 1; i < 3; i++) {
            float y = height * i / 3f;
            canvas.drawLine(0, (int) y, width, (int) y, gridPaint);
        }

        // Draw frame info
        String frameInfo = String.format("Frame: %d / %d", currentFrame, totalFrames);
        canvas.drawText(frameInfo, 15, 30, labelPaint);

        // Draw confidence bar
        float confWidth = width * 0.3f;
        float confHeight = 20f;
        float confX = (width - confWidth) / 2f;
        float confY = 50f;

        // Confidence bar background
        Paint confBgPaint = new Paint();
        confBgPaint.setColor(Color.parseColor("#444444"));
        RectF confBgRect = new RectF(confX, confY, confX + confWidth, confY + confHeight);
        canvas.drawRoundRect(confBgRect, 5f, 5f, confBgPaint);

        // Confidence bar fill
        int confColor = Color.GREEN;
        if (frameConfidence < 0.5f) confColor = Color.RED;
        else if (frameConfidence < 0.7f) confColor = Color.YELLOW;

        Paint confFillPaint = new Paint();
        confFillPaint.setColor(confColor);
        RectF confFillRect = new RectF(confX, confY, confX + confWidth * frameConfidence, confY + confHeight);
        canvas.drawRoundRect(confFillRect, 5f, 5f, confFillPaint);

        // Confidence text
        canvas.drawText(String.format("Confidence: %.2f", frameConfidence), confX, confY - 10, textPaint);

        // Draw action label
        if (!actionLabel.isEmpty()) {
            actionPaint.setColor(Color.CYAN);
            canvas.drawText(actionLabel, width / 2f - 50, 100, actionPaint);

            Paint actionConfPaint = new Paint();
            actionConfPaint.setColor(Color.parseColor("#AAAAAA"));
            actionConfPaint.setTextSize(22f);
            canvas.drawText(String.format("(%.2f)", actionConfidence), width / 2f + 80, 100, actionConfPaint);
        }

        // Draw detected objects
        float boxScaleX = width / 100f;  // Coordinates are 0-100
        float boxScaleY = (height - 150) / 100f;  // Leave space at top
        float offsetY = 120f;

        for (DetectedObject obj : objects) {
            float left = obj.x * boxScaleX;
            float top = obj.y * boxScaleY + offsetY;
            float right = (obj.x + obj.width) * boxScaleX;
            float bottom = (obj.y + obj.height) * boxScaleY + offsetY;

            // Draw bounding box
            boxPaint.setColor(obj.color);
            canvas.drawRect((int) left, (int) top, (int) right, (int) bottom, boxPaint);

            // Draw label
            String labelText = String.format("%s %.2f", obj.label, obj.confidence);
            labelPaint.setColor(obj.color);
            canvas.drawText(labelText, left + 5, top - 5, labelPaint);
        }

        // Draw legend at bottom
        float legendY = height - 30;
        float legendX = 20;
        String[] legendLabels = {"Person", "Car", "Animal", "Other"};
        int[] legendColors = {COLOR_BOX_PERSON, COLOR_BOX_CAR, COLOR_BOX_ANIMAL, COLOR_BOX_DEFAULT};

        for (int i = 0; i < legendLabels.length; i++) {
            // Color box
            Paint legendBoxPaint = new Paint();
            legendBoxPaint.setColor(legendColors[i]);
            canvas.drawRect((int) legendX, (int) legendY, (int) legendX + 25, (int) legendY + 25, legendBoxPaint);

            // Label
            textPaint.setColor(COLOR_TEXT);
            canvas.drawText(legendLabels[i], legendX + 35, legendY + 18, textPaint);

            legendX += 120;
        }

        // Draw object count
        textPaint.setColor(Color.WHITE);
        canvas.drawText(String.format("Objects: %d", objects.size()), width - 120, height - 15, textPaint);
    }
}
