import cv2
import mediapipe as mp
import json
import os
import numpy as np
from collections import deque

# ---------------- CONFIG ----------------
MODEL_PATH = "face_landmarker.task"
DATABASE_FOLDER = "database"

MATCH_THRESHOLD = 0.96
FRAME_BUFFER = 10

LEFT_EYE = 33
RIGHT_EYE = 263

# ---------------- MEDIAPIPE ----------------
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)

# ---------------- NORMALIZATION ----------------
def normalize_landmarks(landmarks):

    landmarks = np.array(landmarks)

    left_eye = landmarks[LEFT_EYE]
    right_eye = landmarks[RIGHT_EYE]

    eye_distance = np.linalg.norm(
        left_eye[:2] - right_eye[:2]
    )

    if eye_distance < 1e-6:
        return None

    center = (left_eye[:2] + right_eye[:2]) / 2

    normalized_xy = (
        landmarks[:, :2] - center
    ) / eye_distance

    normalized_z = (
        landmarks[:, 2:]
    ) / eye_distance

    return np.hstack(
        [normalized_xy, normalized_z]
    )

# ---------------- COSINE SIMILARITY ----------------
def cosine_similarity(a, b):

    a = a.flatten()
    b = b.flatten()

    denom = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denom == 0:
        return 0

    return np.dot(a, b) / denom

# ---------------- LOAD DATABASE ----------------
database = {}

for file in os.listdir(DATABASE_FOLDER):

    if not file.endswith(".json"):
        continue

    path = os.path.join(
        DATABASE_FOLDER,
        file
    )

    with open(path, "r") as f:

        data = json.load(f)

        database[
            data["name"]
        ] = np.array(
            data["landmarks"]
        )

print("\nLoaded database:")
print(list(database.keys()))

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

timestamp = 0

name_buffer = deque(maxlen=FRAME_BUFFER)
score_buffer = deque(maxlen=FRAME_BUFFER)

print("\nPress Q to quit\n")

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )

    timestamp += 33

    person_name = "UNKNOWN"
    confidence = 0

    if result.face_landmarks:

        face_landmarks = result.face_landmarks[0]

        current_landmarks = []

        h, w, _ = frame.shape

        for lm in face_landmarks:

            x = int(lm.x * w)
            y = int(lm.y * h)

            cv2.circle(
                frame,
                (x, y),
                1,
                (0, 255, 0),
                -1
            )

            current_landmarks.append(
                [lm.x, lm.y, lm.z]
            )

        current = normalize_landmarks(
            current_landmarks
        )

        if current is not None:

            num_points = len(current)

            weights = np.ones(num_points)

            important_points = [
                1, 6, 168, 197,
                33, 133,
                263, 362,
                70, 63, 105, 66,
                336, 296, 334, 300
            ]

            for idx in important_points:
                if idx < num_points:
                    weights[idx] = 3.0

            weighted_current = (
                current *
                weights[:, None]
            )

            best_name = "UNKNOWN"
            best_score = -1

            for name, stored in database.items():

                if len(stored) != len(current):
                    continue

                weighted_stored = (
                    stored *
                    weights[:, None]
                )

                score = cosine_similarity(
                    weighted_current,
                    weighted_stored
                )

                if score > best_score:

                    best_score = score
                    best_name = name

            print(
                f"{best_name}: {best_score:.4f}"
            )

            score_buffer.append(
                best_score
            )

            name_buffer.append(
                best_name
            )

            confidence = np.mean(
                score_buffer
            )

            if confidence > MATCH_THRESHOLD:

                counts = {}

                for n in name_buffer:
                    counts[n] = (
                        counts.get(n, 0) + 1
                    )

                person_name = max(
                    counts,
                    key=counts.get
                )

    cv2.putText(
        frame,
        f"Person: {person_name}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.4f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "Face Recognition",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
