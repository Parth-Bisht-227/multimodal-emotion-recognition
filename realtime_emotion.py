# Final realtime emotion detection model

# realtime_emotion.py

import cv2
import numpy as np
import tensorflow as tf
import time

# === Load Trained Model ===
model = tf.keras.models.load_model("emotion_model.h5")

# === Load Haar Cascade for face detection ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# === Start Webcam ===
cap = cv2.VideoCapture(0)

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48)).reshape(1, 48, 48, 1) / 255.0
        pred = np.argmax(model.predict(face, verbose=0))
        label = emotions[pred]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Real-time Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
