Project: Personalized AI Home Security System

Version: 1.1.0
Last Updated: October 20, 2025

This document outlines the development status, operational instructions, and future plans for an intelligent, edge-based home security system. The project's goal is to create a self-contained security solution that runs all AI processing locally on an NVIDIA Jetson Xavier NX for enhanced privacy and real-time performance.

## Core Technologies

Hardware: NVIDIA Jetson Xavier NX, Intel RealSense D455 Camera, USB Microphone, USB Speaker, Arduino Uno, 16x2 I2C LCD.

Software & AI: Python, PyTorch, Torchvision, OpenCV, face_recognition, pyrealsense2, pyserial, gtts, SpeechRecognition.

Current Progress & Implemented Features

The core application is feature-complete and operational. The system can successfully identify known individuals, greet them, and automatically enroll unknown individuals.

### [✅] System & AI Setup:

Jetson development environment is fully configured with all necessary drivers and dependencies.

Python virtual environment (venv) ensures dependency isolation.

PyTorch and a source-built Torchvision are installed and verified.

### [✅] Hardware Integration:

RealSense D455 Camera: Fully integrated for live, high-framerate video capture.

USB Microphone: Integrated for voice-based user enrollment.

USB Speaker: Integrated for spoken greetings and system prompts.

### [✅] Core Application Logic (main.py):

Real-time Face Recognition: The system can detect and identify multiple faces in the camera's view against a trained dataset.

Dynamic Verbal Greetings: Issues a custom "Welcome master..." greeting, correctly formatted for one or multiple recognized individuals.

Intelligent Auto-Enrollment:

If an unknown person is detected alone for 3 seconds, the system automatically starts the registration process.

It uses text-to-speech to ask the person for their name and captures their image.

Autonomous Database Update & Restart:

After a new person is enrolled for the first time, the system automatically runs the encode_faces.py script in the background.

It then gracefully restarts the main application to load the newly trained face data.

### [✅] User Management Scripts:

add_face_voice.py: A utility script for manually enrolling users via camera and microphone.

encode_faces.py: A script to process all images in the dataset folder and generate the known_faces.pkl recognition file.

How to Use the System

Manual Enrollment (Optional First Step): To add initial users, run the manual enrollment script. For each person, take 10-15 photos.

# Activate the virtual environment first
source venv/bin/activate
python add_face_voice.py


Initial Face Encoding: After enrolling the first users manually, you must create the initial recognition database.

python encode_faces.py


Launch the System: Run the main application.

python main.py


The system is now fully active. It will greet known users and automatically handle the registration and re-training for any new, unknown individuals it encounters.

## Next Steps & Future Plans

The final phase of the project involves integrating the physical display and deploying the application as an automated service.

[ ] Arduino & LCD Integration:

Modify main.py to connect to the Arduino via pyserial.

Send messages to the LCD to display system status (e.g., "Welcome, Tyler", "ALERT: Unrecognized Person").

[ ] System Automation (Deployment):

Create a systemd service file to ensure the main.py application launches automatically when the Jetson boots up, turning it into a true appliance.

[ ] Optimization and Enhancements:

Convert the final AI models to an optimized format using NVIDIA TensorRT to maximize inference speed.

Integrate a cloud service like Twilio to send SMS alerts when an unrecognized person is detected.

Build a simple web interface to view a live feed and manage enrolled users remotely.
