import os
import pickle
import cv2
import face_recognition

# Define paths
DATASET_DIR = "dataset"
ENCODINGS_FILE = "encodings.pickle"

known_encodings = []
known_names = []

print("[INFO] Processing dataset images...")

# Loop through each person's folder in dataset
for person_name in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person_name)

    if not os.path.isdir(person_dir):
        continue

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)

        # Load image and convert BGR (OpenCV) to RGB (face_recognition)
        image = cv2.imread(img_path)
        if image is None:
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect face locations and generate encodings
        boxes = face_recognition.face_locations(rgb_image, model="hog")
        encodings = face_recognition.face_encodings(rgb_image, boxes)

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(person_name)

print(f"[INFO] Processed {len(known_encodings)} face encodings.")

# Save encodings to file
data = {"encodings": known_encodings, "names": known_names}
with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump(data, f)

print(f"✅ Encodings successfully saved to '{ENCODINGS_FILE}'!")