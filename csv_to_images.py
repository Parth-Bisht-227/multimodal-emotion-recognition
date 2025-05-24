import os
import csv
import cv2
import numpy as np

def convert_csv_to_images(csv_path, output_dir):
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    usage_map = {
        'Training': 'train',
        'PublicTest': 'val',
        'PrivateTest': 'test'
    }

    # Create output directories
    for split in ['train', 'val', 'test']:
        for emotion in emotions:
            dir_path = os.path.join(output_dir, split, emotion)
            os.makedirs(dir_path, exist_ok=True)

    total_saved = 0

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                emotion_idx = int(row['emotion'])

                if emotion_idx >= len(emotions):
                    print(f"[WARN] Invalid emotion index at row {i}: {emotion_idx}")
                    continue

                emotion = emotions[emotion_idx]
                pixels = row['pixels'].split()
                if len(pixels) != 48 * 48:
                    print(f"[WARN] Incorrect pixel length at row {i}")
                    continue

                usage = usage_map.get(row['Usage'].strip())
                if not usage:
                    print(f"[WARN] Unknown usage at row {i}: {row['Usage']}")
                    continue

                # Create image
                pixels = np.array(list(map(int, pixels)), dtype=np.uint8).reshape(48, 48)
                filename = f"{emotion}_{i}.jpg"
                save_path = os.path.join(output_dir, usage, emotion, filename)

                saved = cv2.imwrite(save_path, pixels)
                if not saved:
                    print(f"[ERROR] Failed to write image at {save_path}")
                else:
                    total_saved += 1

                if i % 1000 == 0:
                    print(f"[INFO] Processed row {i}, saved: {filename}")

            except Exception as e:
                print(f"[ERROR] Failed to process row {i}: {e}")

    print(f"[DONE] Finished converting CSV to images.")
    print(f"Total images saved: {total_saved}")


# Run when executed directly
if __name__ == "__main__":
    csv_path = "./data/fer2013.csv"  # Change if path is different
    output_dir = "./fer2013_images"
    convert_csv_to_images(csv_path, output_dir)
