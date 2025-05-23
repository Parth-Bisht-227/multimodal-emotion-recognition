# csv_to_images.py
import os
import csv
import cv2
import numpy as np

def convert_csv_to_images(csv_path, output_dir):
    # Create directories
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    
    # Create emotion subdirectories
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    for split in ['train', 'val', 'test']:
        for emotion in emotions:
            os.makedirs(os.path.join(output_dir, split, emotion), exist_ok=True)

    # Read CSV
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            pixels = np.array(list(map(int, row['pixels'].split()))).reshape(48, 48)
            emotion = emotions[int(row['emotion'])]
            usage = row['Usage'].lower()
            
            filename = f"{emotion}_{i}.jpg"
            save_path = os.path.join(output_dir, usage, emotion, filename)
            cv2.imwrite(save_path, pixels)

# Usage
convert_csv_to_images('./data/fer2013.csv', 'fer2013_images')
