import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import tensorflow as tf
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("emotion_model.h5", compile=False)
    # Recompile with same settings used during training
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

model = load_model()
emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_emotion(frame):
    img = frame.to_ndarray(format="bgr24")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_processed = cv2.resize(face_roi, (48, 48)).reshape(1, 48, 48, 1) / 255.0
        
        # Debug: Check input range
        # print("Min:", face_processed.min(), "Max:", face_processed.max())  # Should be 0-1
        
        pred = model.predict(face_processed, verbose=0)
        emotion_idx = np.argmax(pred)
        emotion_text = emotions[emotion_idx]
        confidence = np.max(pred)
        
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, f"{emotion_text} ({confidence:.2f})", 
                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("🎥 Real-Time Facial Emotion Detection")
webrtc_streamer(key="example", video_frame_callback=detect_emotion)
