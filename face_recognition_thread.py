import cv2
import numpy as np
import face_recognition
from PyQt6.QtCore import QThread, pyqtSignal


class FaceRecognitionThread(QThread):

    result = pyqtSignal(object, bool)

    def __init__(self, known_encodings):
        super().__init__()
        self.known_encodings = known_encodings
        self.frame = None
        self.running = True

    def update_frame(self, frame):
        self.frame = frame

    def run(self):

        frame_count = 0

        while self.running:

            if self.frame is None:
                continue

            frame_count += 1

            # skip frames for smoother UI
            if frame_count % 10 != 0:
                continue

            frame = self.frame

            # resize for faster processing
            small = cv2.resize(frame, (320, 240))

            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            locations = face_recognition.face_locations(rgb, model="hog")

            if len(locations) == 0:
                self.result.emit(None, False)
                continue

            encodings = face_recognition.face_encodings(rgb, locations)

            face_box = None
            access = False

            for (top, right, bottom, left), enc in zip(locations, encodings):

                matches = face_recognition.compare_faces(
                    self.known_encodings,
                    enc,
                    tolerance=0.45
                )

                distances = face_recognition.face_distance(
                    self.known_encodings,
                    enc
                )

                if len(distances) > 0:

                    best = np.argmin(distances)

                    top *= 2
                    right *= 2
                    bottom *= 2
                    left *= 2

                    face_box = (top, right, bottom, left)

                    if matches[best]:
                        access = True

            self.result.emit(face_box, access)