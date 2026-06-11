import cv2
import mediapipe as mp
import json
import os
import numpy as np

# ---------------- CONFIG ----------------
MODEL_PATH = "face_landmarker.task"
DATABASE_FOLDER = "database"

NUM_SAMPLES = 50

os.makedirs(DATABASE_FOLDER, exist_ok=True)

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
LEFT_EYE = 33
RIGHT_EYE = 263


def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks)

    left_eye = landmarks[LEFT_EYE][:2]
    right_eye = landmarks[RIGHT_EYE][:2]

    eye_distance = np.linalg.norm(left_eye - right_eye)

    if eye_distance < 1e-6:
        return None

    center = (left_eye + right_eye) / 2

    normalized = (landmarks[:, :2] - center) / eye_distance
    z = landmarks[:, 2:] / eye_distance

    return np.hstack([normalized, z]).tolist()


# ---------------- INPUT ----------------
person_name = input("Enter person name: ").strip().lower()

cap = cv2.VideoCapture(0)

timestamp = 0
samples = []

print("\nPress 'S' to start capturing")
print("Move face slightly while recording")
print("Press 'Q' to quit\n")

started = False

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect_for_video(mp_image, timestamp)
    timestamp += 33

    # ❗ FIX 1: skip invalid detection
    if not result.face_landmarks:
        cv2.imshow("Register Face", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    face_landmarks = result.face_landmarks[0]

    current_landmarks = []

    h, w, _ = frame.shape

    for landmark in face_landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        current_landmarks.append([landmark.x, landmark.y, landmark.z])

    # ❗ FIX 2: only start after pressing S AND stable frames
    if started and len(samples) < NUM_SAMPLES:

        normalized = normalize_landmarks(current_landmarks)

        # ❗ FIX 3: skip bad samples
        if normalized is not None:
            samples.append(normalized)

    cv2.putText(
        frame,
        f"Samples: {len(samples)}/{NUM_SAMPLES}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Register Face", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        started = True

    elif key == ord("q"):
        break

    if len(samples) >= NUM_SAMPLES:
        break

cap.release()
cv2.destroyAllWindows()

# ---------------- SAVE DATA ----------------
if len(samples) == 0:
    print("No valid samples collected. Try again.")
    exit()

avg_landmarks = np.mean(samples, axis=0).tolist()

save_path = os.path.join(DATABASE_FOLDER, f"{person_name}.json")

with open(save_path, "w") as f:
    json.dump(
        {
            "name": person_name,
            "landmarks": avg_landmarks
        },
        f,
        indent=4
    )

print(f"\nSaved {person_name}.json successfully!")
