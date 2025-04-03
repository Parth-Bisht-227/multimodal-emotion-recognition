import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Define dataset path
dataset_path = r"D:\Mental health ai\archive\train"

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

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=len(emotions))

# ✅ Print dataset details
print(f"Dataset loaded successfully: {X_train.shape[0]} images")
print(f"X_train shape: {X_train.shape}")  # Expected: (28709, 48, 48, 1)
print(f"y_train shape: {y_train.shape}")  # Expected: (28709, 7)
print(f"Unique labels: {np.unique(y_train)}")  # Check if all emotions exist

# ✅ Preview Sample Images
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i].reshape(48, 48), cmap="gray")
    ax.set_title(f"Label: {y_train[i].argmax()}")
    ax.axis("off")

plt.tight_layout()
plt.show()

# Split the data manually for validation (80% training, 20% validation)
split_ratio = 0.2
split_index = int(len(X_train) * (1 - split_ratio))

X_train_split = X_train[:split_index]
y_train_split = y_train[:split_index]

X_val_split = X_train[split_index:]
y_val_split = y_train[split_index:]

# Data Augmentation to improve generalization
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Fit the generator on the training data
datagen.fit(X_train_split)

# Build the CNN model
model = Sequential()

# Conv1 -> Conv2 -> Conv3 with batch normalization, dropout
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Fully connected layer with L2 regularization
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(len(emotions), activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Callbacks for early stopping and learning rate adjustment
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

# Train the model using the data augmentation generator
history = model.fit(datagen.flow(X_train_split, y_train_split, batch_size=64),
                    epochs=25, validation_data=(X_val_split, y_val_split),
                    callbacks=[early_stop, lr_reduction])

# ✅ Evaluate the model (optional, after training)
loss, accuracy = model.evaluate(X_val_split, y_val_split)
print(f"Final model accuracy: {accuracy * 100:.2f}%")

# ✅ Save the model
model.save("emotion_model.h5")
print("Model saved successfully.")
