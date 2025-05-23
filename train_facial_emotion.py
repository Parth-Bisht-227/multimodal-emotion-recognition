import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define dataset path
dataset_path = r"C:\Users\bisht\ProjectS DTU\Multimodal-Emotion-Recognition-System\data\fer2013.csv"

# Emotion categories (modify if needed)
emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Lists to store images and labels
X_train = []
y_train = []

# Load images from dataset directory
for emotion_idx, emotion in enumerate(emotions):
    emotion_folder = os.path.join(dataset_path, emotion)
    
    if not os.path.exists(emotion_folder):
        print(f"Warning: Folder not found - {emotion_folder}")
        continue  # Skip if folder does not exist

    for image_name in os.listdir(emotion_folder):
        image_path = os.path.join(emotion_folder, image_name)
        
        # Read the image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"Skipping corrupt file: {image_path}")
            continue
        
        # Resize image to 48x48 (standard FER2013 size)
        image = cv2.resize(image, (48, 48))

        # Normalize pixel values (0-1 range)
        image = image.astype("float32") / 255.0

        # Append image and corresponding label
        X_train.append(image)
        y_train.append(emotion_idx)  # Convert emotion to numeric label

# Convert lists to NumPy arrays
X_train = np.array(X_train).reshape(-1, 48, 48, 1)  # Add channel dimension
y_train = np.array(y_train)

# ✅ Print dataset details
print(f"Dataset loaded successfully: {X_train.shape[0]} images")
print(f"X_train shape: {X_train.shape}")  # Expected: (28709, 48, 48, 1)
print(f"y_train shape: {y_train.shape}")  # Expected: (28709,)
print(f"Unique labels: {np.unique(y_train)}")  # Check if all emotions exist

# ✅ Preview Sample Images
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i].reshape(48, 48), cmap="gray")
    ax.set_title(f"Label: {y_train[i]}")
    ax.axis("off")

plt.tight_layout()
plt.show()
