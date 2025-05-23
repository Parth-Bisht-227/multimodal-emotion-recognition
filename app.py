import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import librosa
import os
from PIL import Image

# Load both models
face_model_path = "D:\Mental health ai\emotion_recognition_model.h5"
face_model = tf.keras.models.load_model(face_model_path)

# Define emotion labels
emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Load OpenCV face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load voice model
voice_model_path = "D:/Mental health ai/emotion_recognition_model.h5"
voice_model = tf.keras.models.load_model(voice_model_path)

# Streamlit UI
st.title("🎭 Live Emotion Recognition App")
st.write("Detect emotions from images, videos, and audio!")

# Upload Image
st.subheader("📷 Upload an Image for Facial Emotion Recognition")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48)).reshape(1, 48, 48, 1) / 255.0

        # Predict emotion
        emotion_pred = np.argmax(face_model.predict(face))
        emotion_text = emotions[emotion_pred]

        # Draw rectangle and label
        cv2.rectangle(img_array, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img_array, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display result
    st.image(img_array, caption=f"Detected Emotion: {emotion_text}", use_column_width=True)

# Upload Audio
st.subheader("🎤 Upload an Audio File for Voice Emotion Recognition")
audio_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3", "m4a"])

if audio_file:
    st.audio(audio_file, format="audio/wav")

    # Convert audio file to numpy array for processing
    y, sr = librosa.load(audio_file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1).reshape(1, -1)

    # Predict emotion
    voice_emotion_pred = np.argmax(voice_model.predict(mfcc_mean))
    voice_emotions = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
    voice_emotion_text = voice_emotions[voice_emotion_pred]

    st.write(f"**Predicted Voice Emotion:** {voice_emotion_text}")

# Webcam for Live Detection
st.subheader("📷 Live Webcam Facial Emotion Recognition")
if st.button("Start Webcam"):
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48)).reshape(1, 48, 48, 1) / 255.0

            emotion_pred = np.argmax(face_model.predict(face))
            emotion_text = emotions[emotion_pred]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        stframe.image(frame, channels="BGR")

    cap.release()
