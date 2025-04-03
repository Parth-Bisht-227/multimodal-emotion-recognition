import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Define the path to your dataset
data_path = "D:\\Mental health ai\\Audio_Speech_Actors_01-24"

# Function to extract features (MFCC) from audio
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # Increased MFCC features to 40
    mfcc_mean = np.mean(mfcc, axis=1)  # Take the mean of the MFCCs
    return mfcc_mean

# Function to load the dataset and labels
def load_data(data_path):
    features = []
    labels = []
    
    # Loop over each folder (actor's data)
    for actor_folder in os.listdir(data_path):
        actor_folder_path = os.path.join(data_path, actor_folder)
        
        # Check if it's a directory (actor folder)
        if os.path.isdir(actor_folder_path):
            # Loop over each audio file in the actor's folder
            for audio_file in os.listdir(actor_folder_path):
                audio_file_path = os.path.join(actor_folder_path, audio_file)
                
                # Extract features
                mfcc_features = extract_features(audio_file_path)
                features.append(mfcc_features)
                
                # Get label (emotion) from the audio filename or folder name
                emotion_label = actor_folder  # Example: actor folder name as the label
                labels.append(emotion_label)
    
    return np.array(features), np.array(labels)

# Load data
X, y = load_data(data_path)

# Encode labels into numeric form
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build a more complex neural network model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),  # Input layer
    tf.keras.layers.Dense(256, activation='relu'),  # Increased size of hidden layers
    tf.keras.layers.Dropout(0.5),  # Dropout for regularization
    tf.keras.layers.Dense(128, activation='relu'),  # Another hidden layer
    tf.keras.layers.Dropout(0.5),  # Dropout for regularization
    tf.keras.layers.Dense(64, activation='relu'),  # Another hidden layer
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Save the model if you want to use it later
model.save('emotion_recognition_model.h5')
