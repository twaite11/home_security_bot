import face_recognition
import pickle
import cv2
import pyrealsense2 as rs
import numpy as np
from gtts import gTTS
import os
import time
import speech_recognition as sr
import subprocess
import sys
import requests
from dotenv import load_dotenv
# --- TensorRT/GPU-accelerated detector ---
import torch
from facenet_pytorch import MTCNN

load_dotenv()

# --- System Configuration ---
ENCODINGS_PATH = "known_faces.pkl"
DATASET_PATH = "dataset"
SNAPSHOT_PATH = "snapshots"
GREETING_COOLDOWN_SECONDS = 20
FACE_RECOGNITION_TOLERANCE = 0.6
GREETING_SOUND_FILE = "greeting.mp3"
ENROLLMENT_SOUND_FILE = "enroll.mp3"
UNKNOWN_FACE_TIMER_SECONDS = 3.0
DISCORD_WEBHOOK_URL = os.environ.get('DISCORD_WEBHOOK_URL')

# --- State Variables ---
last_greeting_time = 0
last_unknown_alert_time = 0
greeted_persons_this_session = set()
auto_enrollment_in_progress = False
unknown_face_start_time = None
display_message = "System Initializing..."

# --- NEW: Initialize GPU-accelerated face detector ---
print("[INFO] Loading GPU-accelerated face detector (MTCNN)...")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Running on device: {device}")
mtcnn = MTCNN(
    keep_all=True,        # Find all faces, not just the one with highest probability
    min_face_size=20,     # Don't find tiny faces
    thresholds=[0.6, 0.7, 0.7], # Confidence thresholds
    post_process=False,   # We'll do our own post-processing
    device=device
)

# --- Discord Alert Function ---
def send_alert(message_body, frame_to_send):
    """Saves a snapshot and sends an alert to Discord."""
    if not DISCORD_WEBHOOK_URL or "your_webhook_url" in DISCORD_WEBHOOK_URL:
        print("[WARNING] Discord Webhook URL not set in .env file. Skipping alert.")
        return
    try:
        snapshot_filename = "latest_alert.jpg"
        snapshot_path = os.path.join(SNAPSHOT_PATH, snapshot_filename)
        cv2.imwrite(snapshot_path, frame_to_send)
        print(f"[INFO] Sending Discord alert...")
        with open(snapshot_path, 'rb') as f:
            files = {'file': (snapshot_filename, f, 'image/jpeg')}
            response = requests.post(
                DISCORD_WEBHOOK_URL,
                data={'content': message_body},
                files=files
            )
        if 200 <= response.status_code < 300: print(f"[SUCCESS] Discord alert sent.")
        else: print(f"[ERROR] Failed to send Discord alert: {response.status_code} {response.text}")
    except Exception as e: print(f"[ERROR] Failed to send Discord alert: {e}")

# --- Helper Functions (No changes here) ---
def speak(text, sound_file=GREETING_SOUND_FILE):
    global display_message
    display_message = text
    print(f"[AI-VOICE] {text}")
    try:
        tts = gTTS(text=text, lang='en')
        tts.save(sound_file)
        subprocess.run(["mpg123", "-q", sound_file], check=True)
    except Exception as e: print(f"Error in text-to-speech: {e}")

def listen_for_name():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n[INFO] Calibrating microphone for ambient noise...")
        r.adjust_for_ambient_noise(source, duration=1)
        speak("Please say your name now.", ENROLLMENT_SOUND_FILE)
        time.sleep(1.0)
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=4)
            print("[INFO] Recognizing name...")
            name = r.recognize_google(audio).capitalize()
            print(f"[SUCCESS] I heard the name: {name}")
            return name
        except (sr.WaitTimeoutError, sr.UnknownValueError, sr.RequestError) as e:
            print(f"[ERROR] Could not recognize name: {e}")
            speak("I'm sorry, I could not understand that. Walk out of frame, then walk back in to restart.", ENROLLMENT_SOUND_FILE)
            return None

def enroll_new_person(frame, pipeline):
    global auto_enrollment_in_progress, display_message
    auto_enrollment_in_progress = True
    speak("Hello, I don't recognize you. Please register.", ENROLLMENT_SOUND_FILE)
    time.sleep(1)
    person_name = listen_for_name()
    if person_name:
        if "please say your name now" in person_name.lower():
            speak("Registration failed. Please try again.", ENROLLMENT_SOUND_FILE)
            auto_enrollment_in_progress = False
            return
        person_path = os.path.join(DATASET_PATH, person_name)
        if not os.path.exists(person_path): os.makedirs(person_path)
        img_counter = len([name for name in os.listdir(person_path)])
        img_name = f"{person_name}_{img_counter:04d}.jpg"
        img_path = os.path.join(person_path, img_name)
        cv2.imwrite(img_path, frame)
        speak(f"Thank you, {person_name}. I have saved your profile.", ENROLLMENT_SOUND_FILE)
        speak("Updating database and restarting. One moment.", ENROLLMENT_SOUND_FILE)
        try:
            subprocess.run([sys.executable, "encode_faces.py"], check=True)
            print("[AUTO-SYSTEM] Face encoding complete.")
        except subprocess.CalledProcessError as e:
            speak("Error updating database. Please restart manually.", ENROLLMENT_SOUND_FILE)
            auto_enrollment_in_progress = False
            return
        print("[AUTO-SYSTEM] Restarting the main application...")
        print("[INFO] Releasing camera hardware...")
        pipeline.stop()
        os.execv(sys.executable, ['python'] + sys.argv)
    auto_enrollment_in_progress = False
    display_message = "Monitoring..."

# --- Main Application ---
if __name__ == "__main__":
    if not os.path.exists(SNAPSHOT_PATH): os.makedirs(SNAPSHOT_PATH)
    print("[INFO] Loading face encodings...")
    if not os.path.exists(ENCODINGS_PATH): data = {"encodings": [], "names": []}
    else: data = pickle.loads(open(ENCODINGS_PATH, "rb").read())

    print("[INFO] Starting RealSense camera...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    cv2.namedWindow("Security Feed", cv2.WINDOW_AUTOSIZE)
    print("[INFO] System is running.")
    display_message = "Monitoring..."
    try:
        while True:
            if auto_enrollment_in_progress:
                time.sleep(0.5)
                continue

            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame: continue

            frame = np.asanyarray(color_frame.get_data())
            h, w, _ = frame.shape
            status_bar_height = 60
            cv2.rectangle(frame, (0, h - status_bar_height), (w, h), (0,0,0), -1)

            # --- MODIFIED: Use new MTCNN detector. No resize needed! ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces with MTCNN
            mtcnn_boxes, _ = mtcnn.detect(rgb_frame)

            # Convert boxes from MTCNN format [l, t, r, b] to face_recognition format (t, r, b, l)
            boxes = []
            if mtcnn_boxes is not None:
                for box in mtcnn_boxes:
                    l, t, r, b = [int(val) for val in box]
                    boxes.append((t, r, b, l))


            encodings = face_recognition.face_encodings(rgb_frame, boxes)


            names_for_display = []
            unknown_person_present = False
            current_persons_in_frame = set()

            for encoding in encodings:
                name = "Unknown"
                if data["encodings"]:
                    matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=FACE_RECOGNITION_TOLERANCE)
                    if True in matches:
                        matched_idxs = [i for (i, b) in enumerate(matches) if b]
                        counts = {}
                        for i in matched_idxs:
                            name = data["names"][i]
                            counts[name] = counts.get(name, 0) + 1
                        name = max(counts, key=counts.get)

                names_for_display.append(name)
                if name == "Unknown": unknown_person_present = True
                current_persons_in_frame.add(name)

            time_since_last_greeting = time.time() - last_greeting_time

            if unknown_person_present and len(current_persons_in_frame) == 1:
                if unknown_face_start_time is None:
                    print("[INFO] Unknown person detected. Starting 3-second timer...")
                    unknown_face_start_time = time.time()
                    display_message = "Unrecognized person detected. Please approach to register."

                elif time.time() - unknown_face_start_time > UNKNOWN_FACE_TIMER_SECONDS:
                    if time.time() - last_unknown_alert_time > GREETING_COOLDOWN_SECONDS:
                        print("[INFO] 3-second timer complete. Sending unknown person alert.")
                        alert_time = time.strftime("%I:%M %p")
                        message = f"Unrecognized person detected at {alert_time}."
                        send_alert(message, frame)
                        last_unknown_alert_time = time.time()
                        last_greeting_time = time.time()
                    if not auto_enrollment_in_progress:
                        print("[INFO] Initiating enrollment.")
                        enroll_new_person(frame, pipeline)
                        unknown_face_start_time = None
                        continue
            else:
                if unknown_face_start_time is not None:
                    print("[INFO] Unknown person no longer detected or is not alone. Resetting timer.")
                unknown_face_start_time = None

            if not current_persons_in_frame:
                if greeted_persons_this_session: greeted_persons_this_session.clear()
                display_message = "Monitoring..."

            if current_persons_in_frame and time_since_last_greeting > GREETING_COOLDOWN_SECONDS:
                newly_seen_persons = current_persons_in_frame - greeted_persons_this_session
                recognized_names = [name for name in newly_seen_persons if name != "Unknown"]

                if recognized_names:
                    prefixed_names = [f"master {name}" for name in recognized_names]

                    # --- BUG FIX: Ensure names_str is defined for both cases ---
                    if len(prefixed_names) > 1:
                        names_str = ", ".join(prefixed_names[:-1]) + f" and {prefixed_names[-1]}"
                    else:
                        names_str = prefixed_names[0] # Define names_str for a single person

                    greeting = f"Welcome, {names_str}."


                    speak(greeting)
                    last_greeting_time = time.time()
                    greeted_persons_this_session.update(recognized_names)
                    alert_time = time.strftime("%I:%M %p")
                    message = f"Recognized: {names_str} at {alert_time}."
                    send_alert(message, frame)
                    last_unknown_alert_time = time.time()


            for box_to_display, name_to_display in zip(boxes, names_for_display):
                # Boxes are already (t, r, b, l)
                t, r, b, l = box_to_display
                cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
                cv2.putText(frame, name_to_display, (l, t - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            cv2.putText(frame, display_message, (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Security Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break

    finally:
        print("[INFO] Shutting down...")
        pipeline.stop()
        cv2.destroyAllWindows()

