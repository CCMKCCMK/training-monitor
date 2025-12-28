package com.training.monitor;

import android.content.Context;
import android.os.Environment;
import android.util.Log;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/**
 * File-based logger for debugging.
 * Logs to app's external storage directory.
 */
public class FileLogger {

    private static final String TAG = "FileLogger";
    private static final long MAX_LOG_SIZE = 1024 * 1024; // 1MB
    private static final String LOG_FILE_NAME = "training_monitor.log";

    private static FileLogger instance;
    private File logFile;
    private SimpleDateFormat dateFormat;

    private FileLogger(Context context) {
        // Use app-specific external storage
        File logDir = context.getExternalFilesDir(null);
        if (logDir == null) {
            logDir = context.getFilesDir();
        }
        this.logFile = new File(logDir, LOG_FILE_NAME);
        this.dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US);
    }

    public static synchronized void init(Context context) {
        if (instance == null) {
            instance = new FileLogger(context.getApplicationContext());
        }
    }

    public static FileLogger getInstance() {
        if (instance == null) {
            Log.e(TAG, "FileLogger not initialized! Call init() first.");
        }
        return instance;
    }

    public static String getLogPath() {
        if (instance != null && instance.logFile != null) {
            return instance.logFile.getAbsolutePath();
        }
        return "Not initialized";
    }

    private void writeLog(String level, String tag, String message) {
        if (logFile == null) return;

        // Check file size and rotate if needed
        if (logFile.exists() && logFile.length() > MAX_LOG_SIZE) {
            rotateLog();
        }

        String timestamp = dateFormat.format(new Date());
        String logLine = String.format("%s [%s/%s] %s\n", timestamp, level, tag, message);

        synchronized (this) {
            try {
                FileWriter writer = new FileWriter(logFile, true);
                writer.append(logLine);
                writer.flush();
                writer.close();
            } catch (IOException e) {
                Log.e(TAG, "Failed to write to log file: " + e.getMessage());
            }
        }
    }

    private void rotateLog() {
        synchronized (this) {
            try {
                File backupFile = new File(logFile.getParent(), LOG_FILE_NAME + ".old");
                if (backupFile.exists()) {
                    backupFile.delete();
                }
                if (logFile.exists()) {
                    logFile.renameTo(backupFile);
                }
            } catch (Exception e) {
                Log.e(TAG, "Failed to rotate log: " + e.getMessage());
            }
        }
    }

    public void d(String tag, String message) {
        Log.d(tag, message);
        writeLog("D", tag, message);
    }

    public void i(String tag, String message) {
        Log.i(tag, message);
        writeLog("I", tag, message);
    }

    public void w(String tag, String message) {
        Log.w(tag, message);
        writeLog("W", tag, message);
    }

    public void e(String tag, String message) {
        Log.e(tag, message);
        writeLog("E", tag, message);
    }

    public void e(String tag, String message, Throwable throwable) {
        Log.e(tag, message, throwable);
        String fullMessage = message + "\n" + Log.getStackTraceString(throwable);
        writeLog("E", tag, fullMessage);
    }
}
