# Face Recognition using MediaPipe

## About the Project

This project is a real-time face recognition system built using Python, OpenCV, and MediaPipe. The goal was to recognize registered users through a webcam, even when parts of the face are covered, such as with a mask.

Instead of using traditional face recognition methods, the project uses MediaPipe facial landmarks and compares them using cosine similarity.

## Features

* Register new users through a webcam
* Real-time face recognition
* Facial landmark extraction using MediaPipe
* Landmark normalization for better accuracy
* Confidence score displayed during recognition
* Works with partial face occlusion better than basic face detection methods

## Technologies Used

* Python
* OpenCV
* MediaPipe
* NumPy
* JSON

## Files

* `Register.py` – captures face data and creates a template for a user
* `Recognize_face.py` – recognizes registered users in real time
* `face_landmarker.task` – MediaPipe face landmark model
* `main.py` – additional testing and project code

## How It Works

1. A user is registered through the webcam.
2. Multiple facial landmark samples are collected.
3. The samples are averaged and stored as a face template.
4. During recognition, landmarks from the live camera feed are extracted.
5. Cosine similarity is used to compare the live face with stored templates.
6. The closest match is displayed along with a confidence score.

## What I Learned

Through this project, I learned:

* How facial landmark detection works
* Using MediaPipe for computer vision tasks
* Feature normalization techniques
* Similarity metrics such as cosine similarity
* Working with real-time video streams using OpenCV

## Future Improvements

* Support for multiple faces at once
* Better user interface
* Liveness detection to prevent spoofing
* Storing data in a database instead of JSON files
* Web deployment
