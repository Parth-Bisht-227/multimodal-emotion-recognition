import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import librosa
import os
import tempfile
import joblib
from PIL import Image
import ffmpeg

st.set_page_config(page_title="Emotion Recognition", page_icon="🎭", layout="wide")

# 🧠 Load models with caching
@st.cache_resource
def load_models():
    try:
        face_model = tf.keras.models.load_model("emotion_model.h5") if os.path.exists("emotion_model.h5") else None
        voice_model = tf.keras.models.load_model("voice_emotion_model.h5") if os.path.exists("voice_emotion_model.h5") else None
        scaler = joblib.load("voice_scaler.pkl") if os.path.exists("voice_scaler.pkl") else None
        label_encoder = joblib.load("voice_label_encoder.pkl") if os.path.exists("voice_label_encoder.pkl") else None
        return face_model, voice_model, scaler, label_encoder
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None, None, None, None

# Load HAAR face detector
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
except:
    face_cascade = None

# Load models
face_model, voice_model, voice_scaler, voice_label_encoder = load_models()
face_emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# 🌟 App Layout
st.title("🎭 Multimodal Emotion Recognition")
st.markdown("Detect emotions from **faces** and **voices** using deep learning models. Upload your files below to begin!")

# === FACE EMOTION SECTION ===
st.header("📸 Facial Emotion Detection")
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "png", "jpeg"], key="face")
use_detection = st.checkbox("🔍 Detect faces (use for full-face photos)", value=False)

if uploaded_file and face_model:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    if use_detection and face_cascade is not None:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        if len(faces) == 0:
            st.warning("⚠️ No faces detected.")
        else:
            st.success(f"✅ Detected {len(faces)} face(s). Showing predictions below...")
            cols = st.columns(len(faces))  # Batch display
            for i, (x, y, w, h) in enumerate(faces):
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (48, 48))
                face = face.reshape(1, 48, 48, 1) / 255.0
                pred = face_model.predict(face, verbose=0)
                label = face_emotions[np.argmax(pred)]
                conf = np.max(pred)

                # Draw on original image
                cv2.rectangle(img_array, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(img_array, f"{label} ({conf:.2f})", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                with cols[i]:
                    st.image(gray[y:y+h, x:x+w], caption=f"{label} ({conf:.2f})", width=150)
            st.image(img_array, caption="📌 Image with Detected Faces", use_container_width=True)

    else:
        face_img = gray
        if face_img.shape != (48, 48):
            face_img = cv2.resize(face_img, (48, 48))
        face_img = face_img.reshape(1, 48, 48, 1) / 255.0
        pred = face_model.predict(face_img, verbose=0)
        label = face_emotions[np.argmax(pred)]
        conf = np.max(pred)

        st.image(image, caption=f"🧠 Predicted Emotion: {label} ({conf:.2f})", width=300)
        st.success(f"🧠 Emotion: {label} ({conf:.2f})")

# === VOICE EMOTION SECTION ===
st.header("🎤 Voice Emotion Detection")
audio_file = st.file_uploader("Upload a short audio clip (≤3s)", type=["wav", "mp3", "m4a"], key="audio")

if audio_file and voice_model and voice_scaler and voice_label_encoder:
    st.audio(audio_file)
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_input:
            tmp_input.write(audio_file.read())
            tmp_input_path = tmp_input.name

        tmp_output_path = tmp_input_path + ".wav" if not audio_file.name.endswith(".wav") else tmp_input_path
        if tmp_output_path != tmp_input_path:
            ffmpeg.input(tmp_input_path).output(tmp_output_path).run(quiet=True, overwrite_output=True)

        y, sr = librosa.load(tmp_output_path, sr=22050, duration=3)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc = mfcc[:, :130] if mfcc.shape[1] >= 130 else np.pad(mfcc, ((0, 0), (0, 130 - mfcc.shape[1])), mode='constant')
        mfcc_scaled = voice_scaler.transform(np.mean(mfcc, axis=1).reshape(1, -1))
        pred = voice_model.predict(mfcc_scaled, verbose=0)
        label = voice_label_encoder.inverse_transform([np.argmax(pred)])[0]
        conf = np.max(pred)
        st.success(f"🎧 Predicted Voice Emotion: **{label.title()}** ({conf:.2f})")

        os.remove(tmp_input_path)
        if tmp_output_path != tmp_input_path:
            os.remove(tmp_output_path)

    except Exception as e:
        st.error(f"❌ Audio processing error: {e}")

# === INSTRUCTIONS ===
st.header("🧾 How to Use")
st.markdown("""
- ✅ **Facial Emotion**:
  - If your image contains a full face, enable **"Detect faces"**.
  - If you're using already cropped grayscale 48×48 images, **disable it**.
- ✅ **Voice Emotion**:
  - Upload a short voice clip (≤ 3 seconds). Supported: WAV, MP3, M4A.
""")
st.caption("Built with ❤️ by Team PPM")
