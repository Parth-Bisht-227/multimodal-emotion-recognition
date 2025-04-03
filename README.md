# Multimodal-Emotion-Recognition-System
A deep learning-based multimodal emotion recognition system that detects human emotions from facial expressions and voice signals. It includes CNN and VGG16 models for image input, and MFCC-based DNN for audio. Integrated into a real-time Streamlit app for interactive use.

# Multimodal Emotion Recognition System

This repository contains a complete emotion recognition system using deep learning models trained on both facial image data and speech audio. The project is designed for applications in affective computing, mental health monitoring, and human-computer interaction.

The system supports:
- Facial emotion recognition via webcam or static image input.
- Speech emotion recognition from uploaded audio files (MP3, WAV, M4A).
- A unified Streamlit app interface for real-time interaction.

## Table of Contents

- [Project Overview](#project-overview)
- [Model Architectures](#model-architectures)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Models](#training-the-models)
- [Datasets](#datasets)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)
- [Author](#author)

## Project Overview

This project integrates two modalities for emotion detection:

1. **Facial Emotion Recognition**: Uses a Convolutional Neural Network (CNN) or a pre-trained VGG16 model to classify facial expressions into one of seven basic emotions.
2. **Voice Emotion Recognition**: Uses MFCC features extracted from audio samples and a deep neural network to classify vocal tone into emotional categories.

A simple and interactive Streamlit application allows users to upload images or audio and receive real-time emotion predictions.

## Model Architectures

### Facial Emotion Model (CNN)

- Input: 48x48 grayscale facial images
- Layers:
  - Conv2D → BatchNormalization → MaxPooling → Dropout
  - Dense → Dropout → Output (Softmax)
- Output: 7 classes (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)

### Facial Emotion Model (VGG16 Transfer Learning)

- Base Model: VGG16 (pretrained on ImageNet)
- Custom head: GlobalAveragePooling2D → Dense → Dense(7)
- Training: Final dense layers trainable, base frozen initially

### Voice Emotion Model

- Feature extraction: MFCC (40 features)
- Architecture:
  - Dense(256) → Dropout → Dense(128) → Dropout → Dense(64) → Softmax
- Output: Emotion labels from speaker folders

## Repository Structure

```
emotion-recognition-system/
│
├── app.py                          # Streamlit application for live emotion detection
│
├── emotion_model.py                # VGG16-based facial emotion model training
├── facetrain.py                    # CNN-based facial emotion training
├── train_facial_emotion.py         # Preprocessing + visualization of image dataset
├── voice_emotion_model.py          # Voice emotion model training with MFCC
│
├── models/
│   ├── face_emotion_model.h5       # Trained face model (CNN or VGG16)
│   ├── emotion_recognition_model.h5# Trained audio model
│
├── data/                           # Data folders (not included in repo)
│   ├── archive/                    # FER2013 or similar dataset
│   └── Audio_Speech_Actors_01-24/  # Audio dataset
│
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
```

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/emotion-recognition-system.git
cd emotion-recognition-system
pip install -r requirements.txt
```

Make sure you download the required datasets separately and place them in the `data/` folder.

## Usage

To run the Streamlit application:

```bash
streamlit run app.py
```

The app supports:

- Uploading an image for facial emotion detection
- Uploading an audio file for voice emotion detection
- Live detection using your webcam

Ensure that the model files (`.h5`) are placed correctly under `models/`.

## Training the Models

### Train Facial Emotion Model (CNN)

```bash
python facetrain.py
```

### Train Facial Emotion Model (VGG16 Transfer Learning)

```bash
python emotion_model.py
```

### Train Voice Emotion Model

```bash
python voice_emotion_model.py
```

The models will be saved automatically after training.

## Datasets

1. **Facial Emotion Dataset**:
   - Format: Images sorted into folders by emotion label
   - Size: 48x48 grayscale or RGB images
   - Suggestion: Use FER2013 or similar open datasets

2. **Speech Emotion Dataset**:
   - Used: Custom dataset organized by actor folders
   - Each folder contains emotional speech samples (WAV/MP3)

Note: Due to size, datasets are not included in this repository.

## Results

| Model                | Accuracy |
|---------------------|----------|
| Facial CNN          | ~91%     |
| Facial VGG16        | ~94%     |
| Voice DNN (MFCC)    | ~85%     |

## Future Work

- Integrate facial landmarks and temporal features
- Add multilingual voice emotion classification
- Real-time deployment with edge devices or APIs
- Convert to Docker and host via Streamlit Sharing or Hugging Face Spaces

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Author

**Varun Kasa**  
Master's in Information Systems, Northeastern University  
Email: varunkasa8@gmail.com  
GitHub: https://github.com/KasaVarun
