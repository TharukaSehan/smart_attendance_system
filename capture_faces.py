import cv2
import os

# Get person's name to create folder
person_name = input("Enter the name of the person: ").strip()

if not person_name:
    print("[ERROR] Name cannot be empty!")
    exit()

# Create dataset directory structure
dataset_dir = os.path.join("dataset", person_name)
os.makedirs(dataset_dir, exist_ok=True)

# Open laptop camera (0 is default webcam)
cap = cv2.VideoCapture(0)
img_count = 0

print(f"\n[INFO] Camera started for '{person_name}'.")
print("[INFO] Press 'SPACEBAR' to capture photo | Press 'Q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Couldn't access webcam.")
        break

    # Show status on video feed
    display_frame = frame.copy()
    cv2.putText(display_frame, f"Saved: {img_count} | [SPACE]: Snap | [Q]: Exit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Face Capture Tool", display_frame)

    key = cv2.waitKey(1) & 0xFF

    # SPACEBAR key triggers photo capture
    if key == ord(' '):
        img_name = f"{person_name}_{img_count + 1}.jpg"
        img_path = os.path.join(dataset_dir, img_name)
        cv2.imwrite(img_path, frame)
        img_count += 1
        print(f"📸 Saved: {img_path}")

    # Press 'q' to stop
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n✅ Finished! {img_count} photos saved inside '{dataset_dir}'.")