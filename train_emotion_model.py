# Final script for training the model oon facial data for identifying emotions

# train_emotion_model.py

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense

# === Step 1: Load Data ===
def load_data(train_dir="fer2013_images/train", val_dir="fer2013_images/val"):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(48, 48), batch_size=64, color_mode='grayscale', class_mode='categorical'
    )
    val_generator = val_datagen.flow_from_directory(
        val_dir, target_size=(48, 48), batch_size=64, color_mode='grayscale', class_mode='categorical'
    )

    return train_generator, val_generator

# === Step 2: Build Model ===
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# === Step 3: Train and Save ===
def train_and_save():
    train_gen, val_gen = load_data()
    model = build_model()
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.2, min_lr=1e-6)

    model.fit(train_gen, epochs=50, validation_data=val_gen, callbacks=[early_stop, reduce_lr])
    model.save("emotion_model.h5")
    print("✅ Model saved as 'emotion_model.h5'")

if __name__ == "__main__":
    train_and_save()
