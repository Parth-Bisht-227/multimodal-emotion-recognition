# import os
# import librosa
# import numpy as np
# import tensorflow as tf
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split

# # Define the path to your dataset
# # data_path = "C:\Users\bisht\ProjectS DTU\Multimodal-Emotion-Recognition-System\data\Audio_Speech_Actors_01-24"
# data_path = r"C:\Users\bisht\ProjectS DTU\Multimodal-Emotion-Recognition-System\data\Audio_Speech_Actors_01-24"

# # Function to extract features (MFCC) from audio
# # def extract_features(audio_path):
# #     y, sr = librosa.load(audio_path, sr=None)
# #     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # Increased MFCC features to 40
# #     mfcc_mean = np.mean(mfcc, axis=1)  # Take the mean of the MFCCs
# #     return mfcc_mean
# def extract_features(audio_path):
#     try:
#         y, sr = librosa.load(audio_path, sr=22050, duration=3)  # Limit duration
#         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
#         mfcc_mean = np.mean(mfcc, axis=1)
#         return mfcc_mean
#     except Exception as e:
#         print(f"Error processing {audio_path}: {e}")
#         return None
# # Function to load the dataset and labels
# # def load_data(data_path):
# #     features = []
# #     labels = []
    
# #     # Loop over each folder (actor's data)
# #     for actor_folder in os.listdir(data_path):
# #         actor_folder_path = os.path.join(data_path, actor_folder)
        
# #         # Check if it's a directory (actor folder)
# #         if os.path.isdir(actor_folder_path):
# #             # Loop over each audio file in the actor's folder
# #             for audio_file in os.listdir(actor_folder_path):
# #                 audio_file_path = os.path.join(actor_folder_path, audio_file)
                
# #                 # Extract features
# #                 mfcc_features = extract_features(audio_file_path)
# #                 features.append(mfcc_features)
                
# #                 # Get label (emotion) from the audio filename or folder name
# #                 emotion_label = actor_folder  # Example: actor folder name as the label
# #                 labels.append(emotion_label)
    
# #     return np.array(features), np.array(labels)
# def load_ravdess_data(data_path):
#     features = []
#     labels = []
    
#     # RAVDESS emotion mapping
#     emotion_map = {
#         '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
#         '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
#     }
    
#     for actor_folder in os.listdir(data_path):
#         actor_path = os.path.join(data_path, actor_folder)
        
#         if os.path.isdir(actor_path):
#             for audio_file in os.listdir(actor_path):
#                 if audio_file.endswith('.wav'):
#                     # Extract emotion from filename (RAVDESS format)
#                     # Format: modality-vocal_channel-emotion-emotional_intensity-statement-repetition-actor.wav
#                     parts = audio_file.split('-')
#                     if len(parts) >= 3:
#                         emotion_code = parts[2]
#                         if emotion_code in emotion_map:
#                             audio_path = os.path.join(actor_path, audio_file)
#                             features_extracted = extract_features(audio_path)
                            
#                             if features_extracted is not None:
#                                 features.append(features_extracted)
#                                 labels.append(emotion_map[emotion_code])
    
#     return np.array(features), np.array(labels)

# # Load data
# X, y = load_data(data_path)

# # Encode labels into numeric form
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# # Standardize features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Build a more complex neural network model
# model = tf.keras.Sequential([
#     tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),  # Input layer
#     tf.keras.layers.Dense(256, activation='relu'),  # Increased size of hidden layers
#     tf.keras.layers.Dropout(0.5),  # Dropout for regularization
#     tf.keras.layers.Dense(128, activation='relu'),  # Another hidden layer
#     tf.keras.layers.Dropout(0.5),  # Dropout for regularization
#     tf.keras.layers.Dense(64, activation='relu'),  # Another hidden layer
#     tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')  # Output layer
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# # Evaluate the model
# test_loss, test_accuracy = model.evaluate(X_test, y_test)
# print(f"Test Accuracy: {test_accuracy:.2f}")

# # Save the model if you want to use it later
# model.save('emotion_recognition_model.h5')

import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib  # For saving scaler and encoder

# Define the path to your dataset
data_path = r"C:\Users\bisht\ProjectS DTU\Multimodal-Emotion-Recognition-System\data\Audio_Speech_Actors_01-24"

def extract_features(audio_path):
    try:
        # Load audio with fixed sample rate and duration
        y, sr = librosa.load(audio_path, sr=22050, duration=3)
        
        # Extract MFCC features with padding if needed
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
        
        # Pad/Cut to ensure consistent shape
        if mfcc.shape[1] < 130:
            mfcc = np.pad(mfcc, ((0, 0), (0, 130 - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :130]
            
        return np.mean(mfcc, axis=1)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def load_ravdess_data(data_path):
    features = []
    labels = []
    
    # RAVDESS emotion mapping (simplified to 5 classes)
    emotion_map = {
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fear',
        '08': 'surprise',
        # Combine neutral and calm into neutral
        '01': 'neutral',
        '02': 'neutral',
        # Exclude disgust (limited samples)
        '07': None  
    }
    
    for actor_folder in os.listdir(data_path):
        actor_path = os.path.join(data_path, actor_folder)
        
        if os.path.isdir(actor_path):
            for audio_file in os.listdir(actor_path):
                if audio_file.endswith('.wav'):
                    parts = audio_file.split('-')
                    if len(parts) >= 3:
                        emotion_code = parts[2]
                        mapped_emotion = emotion_map.get(emotion_code)
                        
                        if mapped_emotion is not None:
                            audio_path = os.path.join(actor_path, audio_file)
                            features_extracted = extract_features(audio_path)
                            
                            if features_extracted is not None:
                                features.append(features_extracted)
                                labels.append(mapped_emotion)
    
    return np.array(features), np.array(labels)

# Load and preprocess data
X, y = load_ravdess_data(data_path)

# Filter out None values (if any)
valid_indices = [i for i, label in enumerate(y) if label is not None]
X = X[valid_indices]
y = y[valid_indices]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.2, 
    random_state=42,
    stratify=y_encoded
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save preprocessing objects
joblib.dump(scaler, 'voice_scaler.pkl')
joblib.dump(label_encoder, 'voice_label_encoder.pkl')

# Build improved model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer='l2'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer='l2'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer='l2'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile with adjusted learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Add callbacks
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6
)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nFinal Test Accuracy: {test_accuracy:.2%}")

# Save model
model.save('voice_emotion_model.h5')
print("Model saved successfully")
