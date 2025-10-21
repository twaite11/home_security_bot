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

# --- System Configuration ---
ENCODINGS_PATH = "known_faces.pkl"
DATASET_PATH = "dataset"
GREETING_COOLDOWN_SECONDS = 20
FACE_RECOGNITION_TOLERANCE = 0.6
GREETING_SOUND_FILE = "greeting.mp3"
ENROLLMENT_SOUND_FILE = "enroll.mp3"
UNKNOWN_FACE_TIMER_SECONDS = 3.0

# --- State Variables ---
last_greeting_time = 0
greeted_persons_this_session = set()
auto_enrollment_in_progress = False
unknown_face_start_time = None
# --- NEW: Variable to hold the current status message for the display ---
display_message = "System Initializing..."

# --- Helper Functions ---
def speak(text, sound_file=GREETING_SOUND_FILE):
    """Converts text to speech and plays it, waiting for it to finish."""
    global display_message
    display_message = text # Update the display message when the system speaks
    print(f"[AI-VOICE] {text}")
    try:
        tts = gTTS(text=text, lang='en')
        tts.save(sound_file)
        subprocess.run(["mpg123", "-q", sound_file], check=True)
    except Exception as e:
        print(f"Error in text-to-speech: {e}")

def listen_for_name():
    """Listens for a name via the microphone and returns it as capitalized text."""
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
    """Handles the process of enrolling an unknown person."""
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
        is_new_person = not os.path.exists(person_path)

        if is_new_person:
            os.makedirs(person_path)

        img_counter = len([name for name in os.listdir(person_path)])
        img_name = f"{person_name}_{img_counter:04d}.jpg"
        img_path = os.path.join(person_path, img_name)

        cv2.imwrite(img_path, frame)
        speak(f"Thank you, {person_name}. I have saved your profile.", ENROLLMENT_SOUND_FILE)

        if is_new_person:
            speak("Updating database and restarting. One moment.", ENROLLMENT_SOUND_FILE)
            try:
                subprocess.run([sys.executable, "encode_faces.py"], check=True)
            except subprocess.CalledProcessError as e:
                speak("Error updating database. Please restart manually.", ENROLLMENT_SOUND_FILE)
                auto_enrollment_in_progress = False
                return

            pipeline.stop()
            os.execv(sys.executable, ['python'] + sys.argv)
        else:
            # No restart needed if just adding a photo to an existing person
            time.sleep(2)


    auto_enrollment_in_progress = False
    display_message = "Monitoring..." # Reset message after enrollment attempt


# --- Main Application ---
print("[INFO] Loading face encodings...")
if not os.path.exists(ENCODINGS_PATH):
    data = {"encodings": [], "names": []}
else:
    data = pickle.loads(open(ENCODINGS_PATH, "rb").read())

print("[INFO] Starting RealSense camera...")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30) # Increased resolution for 7" screen
pipeline.start(config)

cv2.namedWindow("Security Feed", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Security Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


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
        # --- NEW: Create a black bar at the bottom for text ---
        h, w, _ = frame.shape
        status_bar_height = 60
        cv2.rectangle(frame, (0, h - status_bar_height), (w, h), (0,0,0), -1)

        rgb_small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (0, 0), fx=0.25, fy=0.25)

        boxes = face_recognition.face_locations(rgb_small, model="cnn")
        encodings = face_recognition.face_encodings(rgb_small, boxes)

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

        if unknown_person_present and len(current_persons_in_frame) == 1:
            if unknown_face_start_time is None:
                unknown_face_start_time = time.time()
                display_message = "Unrecognized person detected. Please approach to register."

            elif time.time() - unknown_face_start_time > UNKNOWN_FACE_TIMER_SECONDS:
                enroll_new_person(frame, pipeline)
                unknown_face_start_time = None
                continue
        else:
            unknown_face_start_time = None

        if not current_persons_in_frame:
            if greeted_persons_this_session: greeted_persons_this_session.clear()
            display_message = "Monitoring..."

        time_since_last_greeting = time.time() - last_greeting_time
        if current_persons_in_frame and time_since_last_greeting > GREETING_COOLDOWN_SECONDS:
            newly_seen_persons = current_persons_in_frame - greeted_persons_this_session
            recognized_names = [name for name in newly_seen_persons if name != "Unknown"]

            if recognized_names:
                if len(recognized_names) > 1:
                    names_str = ", ".join(recognized_names[:-1]) + f" and {recognized_names[-1]}"
                    greeting = f"Welcome, {names_str}."
                else:
                    greeting = f"Welcome, {recognized_names[0]}."

                speak(greeting) # This will also update the display_message
                last_greeting_time = time.time()
                greeted_persons_this_session.update(recognized_names)

        for (top, right, bottom, left), name_to_display in zip(boxes, names_for_display):
            top *= 4; right *= 4; bottom *= 4; left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name_to_display, (left, top - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # --- NEW: Draw the status message on the black bar ---
        cv2.putText(frame, display_message, (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Security Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

finally:
    print("[INFO] Shutting down...")
    pipeline.stop()
    cv2.destroyAllWindows()

