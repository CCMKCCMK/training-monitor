#!/usr/bin/env python3
"""
Video Understanding Test Server for Android Training Monitor
Simulates object detection and action recognition on video frames
"""

import asyncio
import websockets
import json
import random
import math
import time
from datetime import datetime

# Video scenes configuration
SCENES = [
    {"name": "Street Crossing", "action": "crossing", "persons": 3, "duration": 100},
    {"name": "Park Walking", "action": "walking", "persons": 5, "duration": 100},
    {"name": "Store Shopping", "action": "shopping", "persons": 4, "duration": 100},
    {"name": "Office Meeting", "action": "meeting", "persons": 6, "duration": 100},
    {"name": "Gym Exercise", "action": "exercise", "persons": 4, "duration": 100},
]

# Action labels
ACTIONS = ["walking", "running", "sitting", "standing", "crossing", "shopping", "meeting", "exercise"]

# Object classes
CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]

class VideoSimulator:
    def __init__(self, total_frames=500):
        self.total_frames = total_frames
        self.current_frame = 0
        self.scene_index = 0
        self.persons = {}  # Track persons across frames

    def get_frame_data(self):
        """Generate frame-level detection and action data"""
        self.current_frame += 1

        # Change scene every 100 frames
        scene_idx = min(self.current_frame // 100, len(SCENES) - 1)
        scene = SCENES[scene_idx]
        scene_action = scene["action"]
        num_persons = scene["persons"]

        # Update person positions (simulate tracking)
        self._update_persons(num_persons, scene_action)

        # Generate bounding boxes
        boxes = []
        frame_conf = random.uniform(0.85, 0.98)
        action_conf = random.uniform(0.75, 0.95)

        for pid, person in self.persons.items():
            if person["visible"]:
                boxes.append([
                    0,  # classId (person)
                    max(0, min(1, person["conf"])),  # confidence
                    person["x"],  # x
                    person["y"],  # y
                    person["w"],  # width
                    person["h"],  # height
                ])

        return {
            "type": "frame",
            "frame": self.current_frame,
            "total": self.total_frames,
            "confidence": frame_conf,
            "action": scene_action,
            "action_conf": action_conf,
            "boxes": boxes
        }

    def _update_persons(self, num_persons, action):
        """Update person positions for tracking simulation"""
        # Initialize new persons
        for i in range(num_persons):
            if i not in self.persons:
                self.persons[i] = {
                    "x": random.uniform(0.1, 0.7),
                    "y": random.uniform(0.3, 0.6),
                    "w": random.uniform(0.08, 0.15),
                    "h": random.uniform(0.25, 0.45),
                    "conf": random.uniform(0.85, 0.98),
                    "vx": random.uniform(-0.01, 0.01),  # velocity x
                    "vy": random.uniform(-0.005, 0.005),  # velocity y
                    "visible": True
                }

        # Update positions
        for pid, person in list(self.persons.items()):
            person["x"] += person["vx"]
            person["y"] += person["vy"]

            # Keep in bounds
            if person["x"] < 0: person["x"] = 0.8
            if person["x"] > 0.85: person["x"] = 0.05
            if person["y"] < 0.2: person["y"] = 0.6
            if person["y"] > 0.7: person["y"] = 0.3

            # Random visibility changes
            if random.random() < 0.02:
                person["visible"] = not person["visible"]

            # Update confidence
            person["conf"] = max(0.7, min(0.99, person["conf"] + random.uniform(-0.05, 0.05)))

    def reset(self):
        self.current_frame = 0
        self.persons = {}

async def video_handler(websocket):
    """Handle WebSocket connection for video understanding test"""
    remote_addr = str(getattr(websocket, 'remote_address', 'unknown'))
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Client connected: {remote_addr}")

    simulator = VideoSimulator(total_frames=500)

    try:
        while True:
            frame_data = simulator.get_frame_data()

            # Send frame data
            await websocket.send(json.dumps(frame_data))

            # Print progress every 50 frames
            if simulator.current_frame % 50 == 0:
                scene = SCENES[min(simulator.current_frame // 100, len(SCENES) - 1)]
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Frame {simulator.current_frame}/500 "
                      f"| Scene: {scene['name']} | Action: {scene['action']} | "
                      f"Persons: {len([p for p in simulator.persons.values() if p['visible']])}")

            # Reset after completion
            if simulator.current_frame >= simulator.total_frames:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Completed 500 frames! Restarting...")
                simulator.reset()
                await asyncio.sleep(2)

            # Frame rate ~10 fps
            await asyncio.sleep(0.1)

    except websockets.exceptions.ConnectionClosed:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Client disconnected")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Error: {e}")

async def main(port):
    """Start the WebSocket server"""
    print(f"\n{'='*60}")
    print(f"Video Understanding Test Server")
    print(f"{'='*60}")
    print(f"Port: {port}")
    print(f"Scenes: {len(SCENES)}")
    print(f"Actions: {', '.join(ACTIONS)}")
    print(f"Total frames: 500")
    print(f"{'='*60}\n")

    async with websockets.serve(video_handler, "0.0.0.0", port, ping_interval=None):
        print(f"Server started on ws://0.0.0.0:{port}")
        print("Waiting for Android app connection...")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8766

    try:
        asyncio.run(main(port))
    except KeyboardInterrupt:
        print("\nServer stopped")
