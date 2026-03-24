import os
import sys
import cv2
import face_recognition
import pickle

# Resolve path relative to exe directory (for PyInstaller builds)
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FACE_DATA_PATH = os.path.join(BASE_DIR, "face_data.pkl")


def register_face():
    video = cv2.VideoCapture(0)

    if not video.isOpened():
        print("Camera not accessible.")
        return

    # Load existing encodings so new faces are added, not overwritten
    existing_encodings = []
    if os.path.exists(FACE_DATA_PATH):
        try:
            with open(FACE_DATA_PATH, "rb") as f:
                existing_encodings = pickle.load(f)
            print(f"Loaded {len(existing_encodings)} existing face encodings.")
        except Exception:
            pass

    print("Press S to capture samples. Need 3 samples.")
    print("Press Q to quit.")

    encodings = []
    sample_count = 0
    required_samples = 3

    while True:
        ret, frame = video.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.putText(frame, f"Samples: {sample_count}/{required_samples}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 255), 2)

        cv2.imshow("Register Face", frame)

        key = cv2.waitKey(1)

        if key == ord('s'):
            if len(face_locations) != 1:
                print("Ensure only one face is visible.")
                continue

            encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
            encodings.append(encoding)
            sample_count += 1

            print(f"Sample {sample_count} captured.")

            if sample_count >= required_samples:
                all_encodings = existing_encodings + encodings
                with open(FACE_DATA_PATH, "wb") as f:
                    pickle.dump(all_encodings, f)

                print("Face registration completed.")
                break

        elif key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    register_face()