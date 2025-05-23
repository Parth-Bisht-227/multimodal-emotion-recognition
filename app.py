# # import streamlit as st
# # import cv2
# # import numpy as np
# # import tensorflow as tf
# # import librosa
# # import os
# # from PIL import Image

# # # Load both models
# # face_model_path = "D:\Mental health ai\emotion_recognition_model.h5"
# # face_model = tf.keras.models.load_model(face_model_path)

# # # Define emotion labels
# # emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# # # Load OpenCV face detection
# # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # # Load voice model
# # voice_model_path = "D:/Mental health ai/emotion_recognition_model.h5"
# # voice_model = tf.keras.models.load_model(voice_model_path)

# # # Streamlit UI
# # st.title("🎭 Live Emotion Recognition App")
# # st.write("Detect emotions from images, videos, and audio!")

# # # Upload Image
# # st.subheader("📷 Upload an Image for Facial Emotion Recognition")
# # uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# # if uploaded_file:
# #     image = Image.open(uploaded_file)
# #     img_array = np.array(image)

# #     # Convert to grayscale
# #     gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

# #     # Detect faces
# #     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# #     for (x, y, w, h) in faces:
# #         face = gray[y:y+h, x:x+w]
# #         face = cv2.resize(face, (48, 48)).reshape(1, 48, 48, 1) / 255.0

# #         # Predict emotion
# #         emotion_pred = np.argmax(face_model.predict(face))
# #         emotion_text = emotions[emotion_pred]

# #         # Draw rectangle and label
# #         cv2.rectangle(img_array, (x, y), (x+w, y+h), (255, 0, 0), 2)
# #         cv2.putText(img_array, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# #     # Display result
# #     st.image(img_array, caption=f"Detected Emotion: {emotion_text}", use_column_width=True)

# # # Upload Audio
# # st.subheader("🎤 Upload an Audio File for Voice Emotion Recognition")
# # audio_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3", "m4a"])

# # if audio_file:
# #     st.audio(audio_file, format="audio/wav")

# #     # Convert audio file to numpy array for processing
# #     y, sr = librosa.load(audio_file, sr=None)
# #     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
# #     mfcc_mean = np.mean(mfcc, axis=1).reshape(1, -1)

# #     # Predict emotion
# #     voice_emotion_pred = np.argmax(voice_model.predict(mfcc_mean))
# #     voice_emotions = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
# #     voice_emotion_text = voice_emotions[voice_emotion_pred]

# #     st.write(f"**Predicted Voice Emotion:** {voice_emotion_text}")

# # # Webcam for Live Detection
# # st.subheader("📷 Live Webcam Facial Emotion Recognition")
# # if st.button("Start Webcam"):
# #     cap = cv2.VideoCapture(0)
# #     stframe = st.empty()

# #     while True:
# #         ret, frame = cap.read()
# #         if not ret:
# #             break

# #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #         faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# #         for (x, y, w, h) in faces:
# #             face = gray[y:y+h, x:x+w]
# #             face = cv2.resize(face, (48, 48)).reshape(1, 48, 48, 1) / 255.0

# #             emotion_pred = np.argmax(face_model.predict(face))
# #             emotion_text = emotions[emotion_pred]

# #             cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
# #             cv2.putText(frame, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# #         stframe.image(frame, channels="BGR")

# #     cap.release()


# import streamlit as st
# import cv2
# import numpy as np
# import tensorflow as tf
# import librosa
# import os
# from PIL import Image

# # ✅ FIXED: Separate model paths and error handling
# @st.cache_resource
# def load_models():
#     try:
#         face_model = tf.keras.models.load_model("emotion_model.h5")  # From facetrain.py
#         # voice_model = tf.keras.models.load_model("voice_emotion_model.h5")  # Separate model
#         return face_model, None  # Return None for voice until model is fixed
#     except Exception as e:
#         st.error(f"Error loading models: {e}")
#         return None, None

# face_model, voice_model = load_models()

# # Define emotion labels
# emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# # Load OpenCV face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# st.title("🎭 Live Emotion Recognition App")
# st.write("Detect emotions from images, videos, and audio!")

# # ✅ FIXED: Image Upload Section
# st.subheader("📷 Upload an Image for Facial Emotion Recognition")
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# if uploaded_file and face_model:
#     image = Image.open(uploaded_file)
#     img_array = np.array(image)
    
#     # Convert to RGB if needed
#     if len(img_array.shape) == 3 and img_array.shape[2] == 3:
#         rgb_array = img_array
#     else:
#         rgb_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

#     # Convert to grayscale for face detection
#     gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)

#     # Detect faces
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
#     if len(faces) > 0:
#         for (x, y, w, h) in faces:
#             face = gray[y:y+h, x:x+w]
#             face = cv2.resize(face, (48, 48))
#             face = face.reshape(1, 48, 48, 1) / 255.0  # Keep grayscale for custom CNN

#             # Predict emotion
#             emotion_pred = np.argmax(face_model.predict(face, verbose=0))
#             emotion_text = emotions[emotion_pred]

#             # Draw rectangle and label
#             cv2.rectangle(rgb_array, (x, y), (x+w, y+h), (255, 0, 0), 2)
#             cv2.putText(rgb_array, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

#         # Display result
#         st.image(rgb_array, caption=f"Detected Emotion: {emotion_text}", use_column_width=True)
#     else:
#         st.warning("No faces detected in the image.")

# # ✅ FIXED: Webcam Section (removed infinite loop)
# st.subheader("📷 Live Webcam Facial Emotion Recognition")
# if st.button("Start Webcam") and face_model:
#     st.warning("⚠️ Webcam functionality requires local OpenCV setup. Use image upload instead.")
#     # Note: Real webcam streaming in Streamlit requires different approach
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import librosa
import os
import joblib
from PIL import Image

# ✅ FIXED: Model loading with proper error handling
@st.cache_resource
def load_models():
    try:
        # Load facial emotion model
        if os.path.exists("emotion_model.h5"):
            face_model = tf.keras.models.load_model("emotion_model.h5")
            st.success("✅ Facial emotion model loaded successfully!")
        else:
            st.error("❌ emotion_model.h5 not found in current directory")
            face_model = None
            
        # Load voice emotion model and preprocessing objects
        voice_model = None
        scaler = None
        label_encoder = None
        
        if os.path.exists("voice_emotion_model.h5"):
            voice_model = tf.keras.models.load_model("voice_emotion_model.h5")
            if os.path.exists("voice_scaler.pkl"):
                scaler = joblib.load("voice_scaler.pkl")
            if os.path.exists("voice_label_encoder.pkl"):
                label_encoder = joblib.load("voice_label_encoder.pkl")
            st.success("✅ Voice emotion model loaded successfully!")
        else:
            st.warning("⚠️ Voice model files not found")
            
        return face_model, voice_model, scaler, label_encoder
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

# Load models
face_model, voice_model, voice_scaler, voice_label_encoder = load_models()

# Define emotion labels
face_emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Load OpenCV face detection
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        st.error("❌ Could not load face cascade classifier")
except:
    st.error("❌ OpenCV face detection not available")
    face_cascade = None

# Streamlit UI
st.title("🎭 Live Emotion Recognition App")
st.write("Detect emotions from images and audio!")

# Show current directory and files for debugging
with st.expander("🔍 Debug Info"):
    st.write("**Current Directory:**", os.getcwd())
    st.write("**Files in Directory:**")
    files = [f for f in os.listdir('.') if f.endswith(('.h5', '.pkl'))]
    for file in files:
        st.write(f"- {file}")

# ✅ Image Upload Section
st.subheader("📷 Upload an Image for Facial Emotion Recognition")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file and face_model and face_cascade:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Convert to RGB if needed
    if len(img_array.shape) == 3:
        rgb_array = img_array
    else:
        rgb_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) > 0:
        emotion_text = "Unknown"
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face.reshape(1, 48, 48, 1) / 255.0

            # Predict emotion
            try:
                emotion_pred = np.argmax(face_model.predict(face, verbose=0))
                emotion_text = face_emotions[emotion_pred]
                confidence = np.max(face_model.predict(face, verbose=0))
                
                # Draw rectangle and label
                cv2.rectangle(rgb_array, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(rgb_array, f"{emotion_text} ({confidence:.2f})", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            except Exception as e:
                st.error(f"Prediction error: {e}")

        # Display result
        st.image(rgb_array, caption=f"Detected Emotion: {emotion_text}", use_column_width=True)
    else:
        st.warning("No faces detected in the image.")
elif uploaded_file and not face_model:
    st.error("❌ Facial emotion model not loaded. Cannot process image.")

# ✅ Audio Upload Section
st.subheader("🎤 Upload an Audio File for Voice Emotion Recognition")
audio_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3", "m4a"])

if audio_file and voice_model and voice_scaler and voice_label_encoder:
    st.audio(audio_file, format="audio/wav")
    
    try:
        # Process audio
        y, sr = librosa.load(audio_file, sr=22050, duration=3)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
        
        # Ensure consistent shape
        if mfcc.shape[1] < 130:
            mfcc = np.pad(mfcc, ((0, 0), (0, 130 - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :130]
            
        mfcc_mean = np.mean(mfcc, axis=1).reshape(1, -1)
        
        # Scale features
        mfcc_scaled = voice_scaler.transform(mfcc_mean)
        
        # Predict emotion
        voice_pred = np.argmax(voice_model.predict(mfcc_scaled, verbose=0))
        voice_confidence = np.max(voice_model.predict(mfcc_scaled, verbose=0))
        voice_emotion = voice_label_encoder.inverse_transform([voice_pred])[0]
        
        st.success(f"**Predicted Voice Emotion:** {voice_emotion.title()} (Confidence: {voice_confidence:.2f})")
        
    except Exception as e:
        st.error(f"Audio processing error: {e}")
        
elif audio_file and not voice_model:
    st.warning("⚠️ Voice emotion model not loaded. Cannot process audio.")

st.subheader("📷 Real-Time Webcam Facial Emotion Detection")
st.info("""
To use real-time webcam emotion detection, please run the following command in your terminal:

    streamlit run realtime_webcam.py

This will open a new browser window with live webcam emotion detection.
""")


# Add usage instructions
st.subheader("📋 How to Use")
st.write("""
1. **For Facial Emotion Recognition:** Upload an image using the file uploader above
2. **For Voice Emotion Recognition:** Upload an audio file (WAV, MP3, M4A)
3. The app will process and display the predicted emotions with confidence scores
""")
