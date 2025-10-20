import cv2
import os
import speech_recognition as sr

def listen_for_name():
    """
    Listens for a name via the microphone and returns it as capitalized text.
    """
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n[INFO] Calibrating microphone for ambient noise...")
        r.adjust_for_ambient_noise(source, duration=1)
        print("[SPEECH] Please say your name now.")
        
        try:
            # Listen for audio with a 5-second timeout
            audio = r.listen(source, timeout=5, phrase_time_limit=3)
            print("[INFO] Recognizing...")
            # Use Google's free web speech recognition
            name = r.recognize_google(audio)
            # Capitalize the first letter for consistent folder names
            name = name.capitalize() 
            print(f"[SUCCESS] I heard: {name}")
            return name
        except sr.WaitTimeoutError:
            print("[ERROR] No speech detected. Please try again.")
            return None
        except sr.UnknownValueError:
            print("[ERROR] Google Speech Recognition could not understand audio.")
            return None
        except sr.RequestError as e:
            print(f"[ERROR] Could not request results from Google; {e}")
            return None

# --- Main Script ---
dataset_path = "dataset"
# Use 0 for a standard USB webcam. You may need to change this on the Jetson
# if the RealSense camera is not the default.
cam = cv2.VideoCapture(2) 
cv2.namedWindow("Press Space to Register, ESC to exit", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Press Space to Register, ESC to exit", 800, 600)

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Flip frame horizontally for a more natural mirror-like view
    frame = cv2.flip(frame, 1)
    
    # Display instructions on the frame
    cv2.putText(frame, "Position face and press SPACE", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Press Space to Register, ESC to exit", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        print("\n--- Starting Registration ---")
        
        # 1. Listen for the person's name
        person_name = listen_for_name()
        
        # 2. If a name was successfully heard, proceed to save the image
        if person_name:
            person_path = os.path.join(dataset_path, person_name)
            if not os.path.exists(person_path):
                print(f"[INFO] New person detected. Creating directory '{person_name}'.")
                os.makedirs(person_path)
            
            # Find the next available image number to avoid overwriting
            existing_files = len([name for name in os.listdir(person_path) if os.path.isfile(os.path.join(person_path, name))])
            img_name = f"{person_name}_{existing_files:04d}.jpg"
            img_path = os.path.join(person_path, img_name)
            
            # Save the original, un-flipped frame for accurate recognition
            cv2.imwrite(img_path, cv2.flip(frame, 1))
            print(f"[SAVED] {img_name} has been saved!")
        else:
            print("[INFO] Registration failed. Please try again.")

cam.release()
cv2.destroyAllWindows()

