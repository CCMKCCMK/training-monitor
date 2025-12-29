#!/usr/bin/env python3
"""
Real Video Action Recognition Training
Uses PyTorch with 3D CNN for video classification
Sends training metrics to Android app via WebSocket
"""

import asyncio
import websockets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import time
from datetime import datetime
import socket

# ==================== MODEL ====================
class Simple3DCNN(nn.Module):
    """3D CNN for video action recognition"""
    def __init__(self, num_classes=8, seq_len=16):
        super().__init__()
        self.seq_len = seq_len

        # 3D Conv layers: (Batch, Time, Channel, Height, Width)
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        # Calculate flattened size
        # Input: (B, 3, 16, 64, 64)
        # After conv1+pool1: (B, 32, 16, 32, 32)
        # After conv2+pool2: (B, 64, 8, 16, 16)
        # After conv3+pool3: (B, 128, 4, 8, 8) = 32768
        self.fc1 = nn.Linear(32768, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (B, T, C, H, W) -> (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))

        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ==================== DATASET ====================
class SyntheticVideoDataset(Dataset):
    """Synthetic video dataset for training"""
    def __init__(self, num_samples=1000, seq_len=16, img_size=64, num_classes=8):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.img_size = img_size
        self.num_classes = num_classes
        self.actions = [
            "walking", "running", "sitting", "standing",
            "crossing", "shopping", "meeting", "exercise"
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate synthetic video frames with motion patterns
        label = np.random.randint(0, self.num_classes)

        # Create motion pattern based on action
        video = np.zeros((self.seq_len, 3, self.img_size, self.img_size), dtype=np.float32)

        # Add motion patterns (simulating different actions)
        for t in range(self.seq_len):
            offset_x = int((label * 5 + t * 2) % (self.img_size - 16))
            offset_y = int((label * 3 + t) % (self.img_size - 16))

            # Draw moving rectangle (simulating a person)
            for c in range(3):
                video[t, c, offset_y:offset_y+12, offset_x:offset_x+8] = 0.8 + label * 0.02

            # Add noise
            video[t] += np.random.randn(3, self.img_size, self.img_size) * 0.1

        video = np.clip(video, 0, 1)
        return torch.FloatTensor(video), label

# ==================== TRAINING ====================
class VideoTrainer:
    def __init__(self, websocket=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Model
        self.model = Simple3DCNN(num_classes=8, seq_len=16).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)

        # Dataset
        self.train_dataset = SyntheticVideoDataset(num_samples=1000)
        self.val_dataset = SyntheticVideoDataset(num_samples=200)
        self.train_loader = DataLoader(self.train_dataset, batch_size=8, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=8)

        self.websocket = websocket
        self.step = 0
        self.best_val_loss = float('inf')

    async def send_training_data(self, train_loss, val_loss, train_acc, val_acc):
        """Send training metrics to Android app"""
        if self.websocket:
            data = {
                "type": "training",
                "step": self.step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_energy": train_acc,  # Using energy field for accuracy
                "val_energy": val_acc
            }
            try:
                await self.websocket.send(json.dumps(data))
            except:
                pass

    async def send_frame_prediction(self, frame_num, total, predicted_action, confidence, probs):
        """Send frame prediction to Android app"""
        if self.websocket:
            # Create fake bounding box for visualization
            boxes = [[0, confidence, 0.3, 0.3, 0.4, 0.5]]
            data = {
                "type": "frame",
                "frame": frame_num,
                "total": total,
                "confidence": confidence,
                "action": predicted_action,
                "action_conf": confidence,
                "boxes": boxes
            }
            try:
                await self.websocket.send(json.dumps(data))
            except:
                pass

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (videos, labels) in enumerate(self.train_loader):
            videos, labels = videos.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(videos)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return total_loss / len(self.train_loader), 100. * correct / total

    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for videos, labels in self.val_loader:
                videos, labels = videos.to(self.device), labels.to(self.device)
                outputs = self.model(videos)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return total_loss / len(self.val_loader), 100. * correct / total

    async def train(self, num_epochs=50):
        actions = ["walking", "running", "sitting", "standing",
                  "crossing", "shopping", "meeting", "exercise"]

        print(f"\n{'='*60}")
        print(f"Video Action Recognition Training")
        print(f"{'='*60}")
        print(f"Epochs: {num_epochs}")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        print(f"Actions: {', '.join(actions)}")
        print(f"{'='*60}\n")

        for epoch in range(num_epochs):
            start_time = time.time()

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate()

            self.scheduler.step()
            self.step += 1

            epoch_time = time.time() - start_time

            # Print progress
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s) | "
                  f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                  f"Acc: {train_acc:.1f}%/{val_acc:.1f}%")

            # Send to Android
            await self.send_training_data(train_loss, val_loss, train_acc/100, val_acc/100)

            # Send frame prediction every 10 steps for visualization
            if epoch % 10 == 0:
                # Get a sample prediction
                self.model.eval()
                with torch.no_grad():
                    sample_video, _ = self.val_dataset[0]
                    sample_video = sample_video.unsqueeze(0).to(self.device)
                    outputs = self.model(sample_video)
                    probs = torch.softmax(outputs, dim=1)[0]
                    confidence, pred = probs.max(0)
                    predicted_action = actions[pred.item()]
                    await self.send_frame_prediction(
                        epoch, num_epochs, predicted_action, confidence.item(), probs.tolist()
                    )
                self.model.train()

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_video_model.pth')

            # Small delay for visualization
            await asyncio.sleep(0.5)

        print(f"\nTraining complete! Best val loss: {self.best_val_loss:.4f}")

# ==================== WEBSOCKET SERVER ====================
clients = set()

async def handler(websocket):
    """Handle WebSocket connection"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Client connected")
    clients.add(websocket)

    try:
        # Start training when client connects
        trainer = VideoTrainer(websocket)
        await trainer.train(num_epochs=100)

    except websockets.exceptions.ConnectionClosed:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Client disconnected")
    finally:
        clients.remove(websocket)

async def main(port):
    """Start WebSocket server"""
    print(f"\n{'='*60}")
    print(f"Video Training Server")
    print(f"{'='*60}")
    print(f"Port: {port}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"{'='*60}\n")

    async with websockets.serve(handler, "0.0.0.0", port, ping_interval=None):
        print(f"Server started on ws://0.0.0.0:{port}")
        print("Waiting for Android app connection...\n")
        await asyncio.Future()

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8766
    try:
        asyncio.run(main(port))
    except KeyboardInterrupt:
        print("\nServer stopped")
