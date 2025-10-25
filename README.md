# Project: Personalized AI Home Security System

Version: 2.0.0
Last Updated: October 24, 2025

## 1. Project Overview

This project is an intelligent, edge-based home security and greeting system. Think, Ring Doorbell Camera on crack... but not always the good stuff. It runs all AI processing locally on an NVIDIA Jetson Xavier NX for maximum performance and privacy.

The system uses the RealSense camera to monitor a video feed in real-time. It performs high-speed, GPU-accelerated face detection and recognition to identify individuals. It captures images and sends them to a private Discord server for real-time alerts.

Core Features:

GPU-Accelerated AI: Uses PyTorch and facenet-pytorch's MTCNN model to run face detection on the Jetson's GPU (cuda:0), ensuring a fast and responsive framerate.

Real-time Recognition: Identifies known users and greets them by name (e.g., "Welcome, master Tyler and master Anna Elise.").

Instant Discord Alerts: Sends an immediate push notification to a private Discord server when any person (known or unknown) is detected, complete with a snapshot, name, and timestamp.

Autonomous Enrollment: If an unknown person is detected alone for 3 seconds, the system automatically starts a voice-based registration process.

Self-Updating AI: After any new face is registered, the system automatically re-trains its recognition model (known_faces.pkl) and restarts itself to immediately learn the new person.

## 2. Technical Architecture

Hardware

Compute: NVIDIA Jetson Xavier NX

Camera: Intel RealSense D455

Audio Input: USB Microphone

Audio Output: USB Speaker

Display: 7" HDMI Screen (for live feed)

Core Software & Libraries

Environment: Python 3.8 (via venv)

Core AI Framework: PyTorch (NVIDIA-built version for JetPack)

GPU Face Detector: facenet-pytorch (MTCNN)

Face Recognition: face_recognition (for encoding & comparison)

Camera Interface: pyrealsense2

Display: OpenCV (for drawing the feed, boxes, and status bar)

Voice Input: SpeechRecognition

Voice Output: gTTS & mpg123

Alerts: requests (for Discord Webhooks)

Configuration: python-dotenv (for secure API key management)

## How It Works: The main.py Loop

The main application operates in a continuous loop, performing these steps on every frame:

Initialize: Loads the known_faces.pkl file. Initializes the MTCNN face detector on the GPU (cuda:0).

Capture: Grabs a 640x480 frame from the RealSense camera.

Fast Detection (MTCNN): The full-resolution frame is passed to the MTCNN model. This model runs on the GPU, rapidly detecting all faces and returning their bounding boxes.

Recognition (face_recognition): The frame and the bounding boxes are passed to the face_recognition library to generate facial encodings (the unique "fingerprint" for each face).

Compare: These live encodings are compared against the loaded database (known_faces.pkl) to find a match.

Display: The system draws bounding boxes and names ("master Tyler", "Unknown") on the video feed. A black status bar at the bottom shows the current system message (e.g., "Monitoring...", "Welcome, master Tyler.").

Event Logic:

If a Known Person is seen: The system checks if they are new to this "session." If so, it speaks a "Welcome, master..." greeting (handling single or multiple names) and sends a Discord alert with their name, timestamp, and snapshot.

If an Unknown Person is seen (and is alone):

A 3-second timer (UNKNOWN_FACE_TIMER_SECONDS) begins.

If the person remains for 3 seconds, a Discord alert is sent for the "Unrecognized person."

The enroll_new_person() function is triggered.

Auto-Enrollment:

The system speaks to the person, asking them to register and say their name.

The name is captured via the microphone.

The current frame is saved as an image in dataset/<Person's Name>/.

The script then runs encode_faces.py as a subprocess.

Finally, it calls os.execv to restart main.py, forcing it to load the new, updated recognition model.

## 4. The MTCNN Advantage (Performance)

A key part of this project is its real-time performance.

The Problem: The face_recognition library's built-in cnn model is accurate but runs on the CPU. On the Jetson, this is extremely slow, often resulting in 1-2 frames per second (FPS), making real-time application impossible.

The Solution: We replace only the detection part. We use facenet-pytorch, which provides a pre-trained MTCNN (Multi-task Cascaded Convolutional Network) model.

How it Works: By initializing MTCNN(device='cuda:0'), we tell the library to use the PyTorch backend to run all its calculations on the Jetson's GPU.

The Result: The AI workload is offloaded from the CPU to the specialized AI hardware. This provides a massive 10x-30x speedup, allowing us to run detection on the full 640x480 video feed while maintaining a fast, responsive framerate.

## 5. User Management (manage_users.py)

A separate script is provided to list and delete users from your database.

To List All Users:
(Make sure your venv is active)
```
python manage_users.py list
```

Output:
```
[INFO] Listing all registered users:
- person1 
- person2
```

To Delete a User:
This command will delete the user's image folder and automatically re-run encode_faces.py to update the AI model.

```python manage_users.py delete --name "Tyler"
```

Output:

```[WARNING] You are about to permanently delete all data for 'Tyler'.
Are you sure? (y/n): y
[SUCCESS] Successfully deleted user 'Tyler'.
[INFO] Re-running face encoding to update the model...
[SUCCESS] Encodings file 'known_faces.pkl' has been updated.
```

7. Future Plans

[ ] System Automation (Deployment): Create a systemd service file to ensure main.py launches automatically when the Jetson boots up.

[ ] Remote Management UI: Build a simple web interface (using Flask) to view the live feed and manage enrolled users from a phone or browser.

[ ] Cloud Backup: Modify encode_faces.py to also back up the entire dataset folder to a secure cloud drive.

[ ] Physical Enclosure: Design and 3D-print a custom case to house all components (Jetson, screen, camera, mic) in a single unit.