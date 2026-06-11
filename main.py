import cv2
import mediapipe as mp
# MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
# Downloaded model path
MODEL_PATH = "face_landmarker.task"
# Configure face landmarker
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)
landmarker = FaceLandmarker.create_from_options(options)
# Webcam
cap = cv2.VideoCapture(0)
frame_timestamp_ms = 0
while True:
    success, frame = cap.read()
    if not success:
        print("Camera failed")
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert OpenCV image -> MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )
    # Detect landmarks
    result = landmarker.detect_for_video(
        mp_image,
        frame_timestamp_ms
    )
    frame_timestamp_ms += 33
    h, w, _ = frame.shape
    # Draw landmarks
    if result.face_landmarks:
        for face_landmarks in result.face_landmarks:
            for idx, landmark in enumerate(face_landmarks):
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(
                    frame,
                    str(idx),
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.25,
                    (255, 255, 255),
                    1
                )
    cv2.imshow("Face Landmarks", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
