package com.training.monitor;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.util.AttributeSet;
import android.view.View;

/**
 * Video frame view - displays actual video frames/images.
 * Receives Bitmap frames from training server and displays them.
 */
public class VideoFrameView extends View {

    private static final int COLOR_BG = Color.BLACK;
    private static final int COLOR_TEXT = Color.GRAY;

    private Bitmap currentFrame = null;
    private Paint bitmapPaint;
    private Paint textPaint;

    public VideoFrameView(Context context) {
        super(context);
        init();
    }

    public VideoFrameView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        bitmapPaint = new Paint();
        bitmapPaint.setFilterBitmap(true);
        bitmapPaint.setAntiAlias(true);

        textPaint = new Paint();
        textPaint.setColor(COLOR_TEXT);
        textPaint.setTextSize(24f);
        textPaint.setAntiAlias(true);
    }

    /**
     * Set the video frame image to display.
     * Takes ownership of the bitmap - will recycle when replaced.
     */
    public void setFrameImage(Bitmap bitmap) {
        // Recycle old bitmap to prevent memory leaks
        if (currentFrame != null && !currentFrame.isRecycled()) {
            currentFrame.recycle();
        }
        currentFrame = bitmap;
        postInvalidate();
    }

    /**
     * Clear the current frame.
     */
    @Override
    public void clear() {
        if (currentFrame != null && !currentFrame.isRecycled()) {
            currentFrame.recycle();
            currentFrame = null;
        }
        postInvalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        int width = getWidth();
        int height = getHeight();

        // Background
        canvas.drawColor(COLOR_BG);

        // Draw frame if available
        if (currentFrame != null && !currentFrame.isRecycled()) {
            // Calculate scale to fit image while maintaining aspect ratio
            float scale = Math.min(
                (float) width / currentFrame.getWidth(),
                (float) height / currentFrame.getHeight()
            );

            int scaledWidth = (int) (currentFrame.getWidth() * scale);
            int scaledHeight = (int) (currentFrame.getHeight() * scale);

            // Center the image
            int x = (width - scaledWidth) / 2;
            int y = (height - scaledHeight) / 2;

            Rect destRect = new Rect(x, y, x + scaledWidth, y + scaledHeight);
            canvas.drawBitmap(currentFrame, null, destRect, bitmapPaint);
        } else {
            // No frame - draw placeholder text
            String placeholder = "No video frame";
            float textWidth = textPaint.measureText(placeholder);
            canvas.drawText(placeholder, (width - textWidth) / 2, height / 2, textPaint);
        }
    }

    @Override
    protected void onDetachedFromWindow() {
        super.onDetachedFromWindow();
        // Clean up bitmap when view is detached
        if (currentFrame != null && !currentFrame.isRecycled()) {
            currentFrame.recycle();
            currentFrame = null;
        }
    }
}
