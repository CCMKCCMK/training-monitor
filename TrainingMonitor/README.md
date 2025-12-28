# Training Monitor - Android App

Real-time training visualization app for entropy-based learning models.

## Features
- Real-time Loss & Energy charts
- Prediction vs Actual comparison
- Reads data from `~/.training_live.json` created by Python training script

## Building APK with GitHub Actions

### 1. Create GitHub Repository
```bash
cd /data/data/com.termux/files/home/chat/app
git add .
git commit -m "Add TrainingMonitor app"
git remote add origin https://github.com/YOUR_USERNAME/training-monitor.git
git push -u origin main
```

### 2. Enable GitHub Actions
- Go to https://github.com/YOUR_USERNAME/training-monitor/actions
- Click "I understand my workflows, go ahead and enable them"

### 3. Build APK
The APK will be automatically built on each push to `main`.

To manually trigger a build:
- Go to Actions tab
- Select "Build APK" workflow
- Click "Run workflow"

### 4. Download APK
- Go to Actions tab
- Select a completed workflow run
- Scroll to "Artifacts" section
- Download `TrainingMonitor-APK`

## Usage

### 1. Install APK on Android
Download the APK from GitHub Actions artifacts and install it.

### 2. Run Python Training
```python
from entropy_agent import train

train(data="lorenz", epochs=20, android_viz=True)
```

### 3. Open the App
The app will automatically show live training charts.

## Data Format
The app reads `~/.training_live.json` with format:
```json
{
  "step": 100,
  "train_loss": 0.5,
  "val_loss": 0.4,
  "train_energy": 0.3,
  "val_energy": 0.25
}
```

## Permissions
- Read external storage (to access Termux home directory)
