import cv2
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import joblib
import time
from tensorflow.keras.preprocessing.image import img_to_array

# === Load Models ===
print("Loading models...")
face_model = tf.keras.models.load_model("emotion_model_2.h5")
voice_model = tf.keras.models.load_model("voice_emotion_model_W1.h5")
voice_scaler = joblib.load("voice_scaler.pkl")
voice_label_encoder = joblib.load("voice_label_encoder.pkl")

# === Emotion Labels ===
face_emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# === Load Haar Cascade for face detection ===
frontal_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# === Voice Emotion Prediction Function ===
def get_voice_emotion():
    duration = 2  # seconds
    fs = 22050

    print("🎤 Listening for voice input...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()

    y = np.squeeze(audio)
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=fs, n_mfcc=40)
        mfcc = np.mean(mfcc, axis=1).reshape(1, -1)
        mfcc_scaled = voice_scaler.transform(mfcc)
        prediction = voice_model.predict(mfcc_scaled, verbose=0)[0]
        label = voice_label_encoder.inverse_transform([np.argmax(prediction)])[0]
        confidence = np.max(prediction)
        return prediction, label, confidence
    except Exception as e:
        print("❌ Voice processing error:", e)
        return None, None, None

# === Start Webcam ===
cap = cv2.VideoCapture(0)
frame_count = 0
voice_emotion = "Listening..."
voice_probs = None
voice_label = None
voice_conf = 0.0

print("🟢 Real-Time Emotion Detection Started (Press 'q' to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = frontal_face.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        faces = profile_face.detectMultiScale(gray, 1.3, 5)

    face_label = None
    face_probs = None
    face_conf = 0.0

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum(roi_gray) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = face_model.predict(roi, verbose=0)[0]
            face_label = face_emotions[np.argmax(prediction)]
            face_probs = prediction
            face_conf = np.max(prediction)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'Face: {face_label} ({face_conf:.2f})', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Every 60 frames (~2 seconds), update voice emotion
    if frame_count % 60 == 0:
        voice_probs, voice_label, voice_conf = get_voice_emotion()
        if voice_label:
            voice_emotion = f"{voice_label} ({voice_conf:.2f})"

    # === Soft Voting (if possible) ===
    final_emotion = "Analyzing..."
    final_confidence = 0.0

    if face_probs is not None and voice_probs is not None and face_probs.shape == voice_probs.shape:
        w1, w2 = 0.6, 0.4  # weights: prioritize face slightly more
        combined_probs = (w1 * face_probs + w2 * voice_probs) / (w1 + w2)
        final_emotion = face_emotions[np.argmax(combined_probs)]
        final_confidence = np.max(combined_probs)
    elif face_probs is not None:
        final_emotion = face_label
        final_confidence = face_conf
    elif voice_probs is not None:
        final_emotion = voice_label
        final_confidence = voice_conf

    # === Display Final Emotion on Frame ===
    cv2.putText(frame, f'Voice: {voice_emotion}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f'Final: {final_emotion} ({final_confidence:.2f})', (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 0), 2)

    cv2.imshow('Real-Time Emotion Detection (Face + Voice)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
