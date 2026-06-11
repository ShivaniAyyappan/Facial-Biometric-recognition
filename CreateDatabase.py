import cv2
import mediapipe as mp
import json
import os
import numpy as np

MODEL_PATH = "face_landmarker.task"
DATABASE_FOLDER = "database"

os.makedirs(DATABASE_FOLDER, exist_ok=True)

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)

LEFT_EYE = 33
RIGHT_EYE = 263

def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks)

    left_eye = landmarks[LEFT_EYE]
    right_eye = landmarks[RIGHT_EYE]

    eye_distance = np.linalg.norm(left_eye[:2] - right_eye[:2])

    if eye_distance == 0:
        eye_distance = 1

    center_x = (left_eye[0] + right_eye[0]) / 2
    center_y = (left_eye[1] + right_eye[1]) / 2

    normalized = []

    for x, y, z in landmarks:
        normalized.append([
            (x - center_x) / eye_distance,
            (y - center_y) / eye_distance,
            z / eye_distance
        ])

    return normalized

images = {
    "amma": "faceamma.jpg",
    "anish": "faceanish.jpg",
    "daddy": "facedaddy.jpg",
    "shivani": "faceshivani.jpg"
}

for name, filename in images.items():

    image = cv2.imread(filename)

    if image is None:
        print(f"Could not load {filename}")
        continue

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        print(f"No face found in {filename}")
        continue

    landmarks = []

    for lm in result.face_landmarks[0]:
        landmarks.append([lm.x, lm.y, lm.z])

    landmarks = normalize_landmarks(landmarks)

    output = {
        "name": name,
        "landmarks": landmarks
    }

    with open(
        os.path.join(DATABASE_FOLDER, f"{name}.json"),
        "w"
    ) as f:
        json.dump(output, f)

    print(f"Saved {name}.json")

print("Database creation complete.")
