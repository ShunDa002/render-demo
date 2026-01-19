# classifier.py
import cv2
from ultralytics import YOLO

# Load your YOLOv8 model
model = YOLO("models/train1_last.pt")  # path to your exported model

def extract_frames(video_path: str):
    """
    Extract frames from video every `frame_rate` frames
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        count += 1
    cap.release()
    return frames

def classify_video(video_path: str):
    frames = extract_frames(video_path)
    predictions = []

    for frame in frames:
        results = model.predict(frame)
        probs = results[0].probs

        if probs is not None:
            class_id = int(probs.top1)
            class_name = results[0].names[class_id]
            predictions.append(class_name)

    if not predictions:
        return "unknown"

    return max(set(predictions), key=predictions.count)

