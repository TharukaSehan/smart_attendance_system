import cv2
import face_recognition
import pickle
import numpy as np
import csv
import os
from datetime import datetime

# 1. Load trained encodings
print("[INFO] Loading face encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.load(f)

# 2. Setup CSV Attendance File
csv_file = "attendance.csv"
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Timestamp"])

logged_names = set()

# 3. Start Webcam
cap = cv2.VideoCapture(0)
print("[INFO] Starting camera. Press 'Q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    # Downscale frame to 1/4 size for faster face recognition performance
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect face locations and encodings
    boxes = face_recognition.face_locations(rgb_small_frame)
    encodings = face_recognition.face_encodings(rgb_small_frame, boxes)

    for (top, right, bottom, left), encoding in zip(boxes, encodings):
        # Compare with known encodings
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(data["encodings"], encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = data["names"][best_match_index]

        # Log attendance to CSV if not already logged in this session
        if name != "Unknown" and name not in logged_names:
            logged_names.add(name)
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([name, timestamp])
            print(f"✅ Marked Attendance for: {name} at {timestamp}")

        # Scale coordinates back up to original frame size (4x)
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw box and label on camera feed
        box_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Smart Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n✅ Session ended. Attendance saved to '{csv_file}'.")