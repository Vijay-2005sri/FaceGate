import os
import sys
import cv2
import face_recognition
import pickle
import numpy as np

# Resolve path relative to exe directory (for PyInstaller builds)
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FACE_DATA_PATH = os.path.join(BASE_DIR, "face_data.pkl")


def verify_face():
    try:
        with open(FACE_DATA_PATH, "rb") as f:
            known_encodings = pickle.load(f)
    except FileNotFoundError:
        print("No registered face found.")
        return

    video = cv2.VideoCapture(0)

    if not video.isOpened():
        print("Camera not accessible.")
        return

    print("Press Q to quit.")

    threshold = 0.45
    consecutive_matches = 0
    required_matches = 1

    process_this_frame = True

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Resize frame to 1/4 size for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        process_this_frame = not process_this_frame

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

            distances = face_recognition.face_distance(known_encodings, face_encoding)
            min_distance = np.min(distances)

            # Scale face locations back to original frame size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            if min_distance < threshold:
                consecutive_matches += 1
            else:
                consecutive_matches = 0

            color = (0, 255, 0) if min_distance < threshold else (0, 0, 255)
            label = f"Distance: {min_distance:.3f}"

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if consecutive_matches >= required_matches:
            print("ACCESS GRANTED")
            break

        cv2.imshow("Verify Face - Optimized", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    verify_face()