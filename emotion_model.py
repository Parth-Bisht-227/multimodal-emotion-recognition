# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.models import Model
# import time

# # # Step 1: Load and Preprocess Data
# # def load_data():
# #     # Replace with your dataset path
# #     # Before
# #     # train_data_dir = r"D:\Mental health ai\archive\train"
# #     # val_data_dir = r"D:\Mental health ai\archive\test"

# #     # After
# #     train_data_dir = 'fer2013_images/train'
# #     val_data_dir = 'fer2013_images/val'
    
# #     # Data Augmentation for Training
# #     train_datagen = ImageDataGenerator(
# #         rescale=1.0 / 255.0,
# #         rotation_range=20,
# #         width_shift_range=0.2,
# #         height_shift_range=0.2,
# #         shear_range=0.2,
# #         zoom_range=0.2,
# #         horizontal_flip=True,
# #         fill_mode='nearest'
# #     )

# #     # Validation Data (No Augmentation)
# #     val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# #     train_generator = train_datagen.flow_from_directory(
# #         train_data_dir,
# #         target_size=(48, 48),
# #         batch_size=32,
# #         color_mode='rgb',
# #         class_mode='categorical'
# #     )

# #     val_generator = val_datagen.flow_from_directory(
# #         val_data_dir,
# #         target_size=(48, 48),
# #         batch_size=32,
# #         color_mode='rgb',
# #         class_mode='categorical'
# #     )

# #     return train_generator, val_generator

# # # Step 2: Build the Model
# # def build_model():
# #     base_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
# #     x = base_model.output
# #     x = GlobalAveragePooling2D()(x)
# #     x = Dense(1024, activation='relu')(x)
# #     predictions = Dense(7, activation='softmax')(x)  # 7 emotions

# #     model = Model(inputs=base_model.input, outputs=predictions)

# #     # Freeze initial layers
# #     for layer in base_model.layers:
# #         layer.trainable = False

# #     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# #     return model

# # # Step 3: Train the Model
# # def train_model(model, train_generator, val_generator):
# #     history = model.fit(
# #         train_generator,
# #         epochs=50,
# #         validation_data=val_generator
# #     )
# #     model.save("emotion_recognition_model.h5")
# #     return history

# # # Step 4: Real-Time Emotion Detection
# # def emotion_detection():
# #     # Load trained model
# #     try:
# #         face_model = tf.keras.models.load_model("emotion_recognition_model.h5")
# #     except Exception as e:
# #         print(f"Error loading model: {e}")
# #         exit()

# #     # Load OpenCV face detection
# #     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# #     cap = cv2.VideoCapture(0)  # Open webcam

# #     while True:
# #         start_time = time.time()  # Track frame time

# #         ret, frame = cap.read()
# #         if not ret:
# #             print("Error: Could not read frame.")
# #             break

# #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# #         for (x, y, w, h) in faces:
# #             face = gray[y:y+h, x:x+w]
# #             face = cv2.resize(face, (48, 48))
# #             face = np.expand_dims(face, axis=-1)  # Add channel dimension
# #             face = np.repeat(face, 3, axis=-1)  # Convert to 3 channels for VGG16
# #             face = face.reshape(1, 48, 48, 3) / 255.0  # Normalize

# #             emotion_pred = np.argmax(face_model.predict(face))
# #             emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
# #             emotion_text = emotions[emotion_pred]

# #             # Draw rectangle & label
# #             cv2.putText(frame, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
# #             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

# #         # Calculate FPS
# #         fps = 1.0 / (time.time() - start_time)
# #         cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# #         # Display window
# #         cv2.imshow("Emotion Detection", frame)

# #         if cv2.waitKey(1) & 0xFF == ord('q'):
# #             break

# #     cap.release()
# #     cv2.destroyAllWindows()

# # # Main Execution
# # if __name__ == "__main__":
# #     # Step 1: Load Data
# #     train_generator, val_generator = load_data()

# #     # Step 2: Build Model
# #     model = build_model()

# #     # Step 3: Train Model
# #     train_model(model, train_generator, val_generator)

# #     # Step 4: Run Real-Time Emotion Detection
# #     emotion_detection()



# # ✅ FIXED: Use smaller CNN instead of VGG16 for 48x48 images
# def build_model():
#     model = Sequential([
#         Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
#         BatchNormalization(),
#         MaxPooling2D(2, 2),
#         Dropout(0.25),
        
#         Conv2D(64, (3, 3), activation='relu'),
#         BatchNormalization(),
#         MaxPooling2D(2, 2),
#         Dropout(0.25),
        
#         Conv2D(128, (3, 3), activation='relu'),
#         BatchNormalization(),
#         MaxPooling2D(2, 2),
#         Dropout(0.25),
        
#         Flatten(),
#         Dense(512, activation='relu'),
#         Dropout(0.5),
#         Dense(7, activation='softmax')
#     ])
    
#     model.compile(optimizer='adam', 
#                   loss='categorical_crossentropy', 
#                   metrics=['accuracy'])
#     return model


import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, BatchNormalization, 
                                     Dropout, Flatten, Dense)
import time

# Step 1: Load and Preprocess Data
def load_data():
    train_data_dir = 'fer2013_images/train'
    val_data_dir = 'fer2013_images/val'
    
    # Data Augmentation for Training (grayscale)
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

    # Validation Data (No Augmentation)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(48, 48),
        batch_size=64,
        color_mode='grayscale',
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(48, 48),
        batch_size=64,
        color_mode='grayscale',
        class_mode='categorical'
    )

    return train_generator, val_generator

# Step 2: Build the Model (Optimized for 48x48 grayscale)
def build_model():
    model = Sequential()
    
    # Block 1
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', 
                    input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Block 2
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Block 3
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Classification Head
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Step 3: Train the Model
def train_model(model, train_generator, val_generator):
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    )

    history = model.fit(
        train_generator,
        epochs=100,
        validation_data=val_generator,
        callbacks=[early_stop, reduce_lr]
    )
    
    model.save("emotion_model.h5")
    return history

# Step 4: Real-Time Emotion Detection
def emotion_detection():
    try:
        face_model = tf.keras.models.load_model("emotion_model.h5")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    cap = cv2.VideoCapture(0)
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face.reshape(1, 48, 48, 1) / 255.0  # Grayscale processing

            emotion_pred = np.argmax(face_model.predict(face, verbose=0))
            emotion_text = emotions[emotion_pred]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion_text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Calculate FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main Execution
if __name__ == "__main__":
    # Load data
    train_gen, val_gen = load_data()
    
    # Build model
    model = build_model()
    model.summary()
    
    # Train model
    train_model(model, train_gen, val_gen)
    
    # Run real-time detection
    emotion_detection()
