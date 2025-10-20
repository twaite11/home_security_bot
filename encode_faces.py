import face_recognition
import pickle
import cv2
import os

print("[INFO] Starting face encoding...")
dataset_path = "dataset"

known_encodings = []
known_names = []

# Loop over the folders in the dataset (each folder is one person)
for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_path):
        continue

    # Loop over the images of the person
    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        
        print(f"[INFO] Processing {image_name} for {person_name}...")
        
        # Load image and convert it from BGR (OpenCV default) to RGB
        image = cv2.imread(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect the (x, y)-coordinates of the bounding boxes for each face
        # Using "cnn" is more accurate but slower. "hog" is faster but less accurate.
        boxes = face_recognition.face_locations(rgb, model="cnn") 
        
        # Compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)
        
        # Add each encoding + name to our list of known faces
        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(person_name)

# Save the encodings and names to disk for later use
print("[INFO] Serializing encodings...")
data = {"encodings": known_encodings, "names": known_names}
with open("known_faces.pkl", "wb") as f:
    f.write(pickle.dumps(data))

print("[INFO] Encoding complete. The 'known_faces.pkl' file has been created.")

