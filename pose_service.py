import cv2
from movenet import detect_pose
import math
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import tempfile

app = FastAPI()


# -----------------------
# Helper: calculate angle between 3 points
# -----------------------
def angle(a, b, c):
    ba = (a['x'] - b['x'], a['y'] - b['y'])
    bc = (c['x'] - b['x'], c['y'] - b['y'])

    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag1 = math.sqrt(ba[0]**2 + ba[1]**2)
    mag2 = math.sqrt(bc[0]**2 + bc[1]**2)

    if mag1 * mag2 == 0:
        return 180.0

    return math.degrees(math.acos(dot / (mag1 * mag2)))


# -----------------------
# OFFICIAL BODY-ONLY RULES
# -----------------------
SHOT_RULES = {
    "serve": [
        {
            "condition": lambda k: k["right_wrist"]["y"] < k["right_hip"]["y"],
            "message": "Racket hand must be below waist during serve"
        },
        {
            "condition": lambda k: angle(k["right_shoulder"], k["right_elbow"], k["right_wrist"]) < 150,
            "message": "Serving arm should be straighter"
        },
        {
            "condition": lambda k: abs(k["left_shoulder"]["y"] - k["right_shoulder"]["y"]) > 0.08,
            "message": "Shoulders should be level during serve"
        }
    ],
    "smash": [
        {
            "condition": lambda k: k["right_elbow"]["y"] > k["right_shoulder"]["y"],
            "message": "Raise elbow higher before smashing"
        },
        {
            "condition": lambda k: angle(k["right_hip"], k["right_knee"], k["right_ankle"]) > 165,
            "message": "Bend knees more to generate smash power"
        },
        {
            "condition": lambda k: abs(k["left_shoulder"]["y"] - k["right_shoulder"]["y"]) < 0.04,
            "message": "Rotate shoulders more for a powerful smash"
        }
    ]
}


# -----------------------
# Analyze video function
# -----------------------
def analyze_video(video_path: str, shot_type: str):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    results = []
    frame_index = 0
    shot_type = shot_type.lower()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        # detect pose (returns dict with pixel coordinates)
        keypoints = detect_pose(frame)

        # filter only needed body keypoints
        body_kps = {k: keypoints[k] for k in [
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle"
        ] if k in keypoints}

        # normalize coordinates to 0-1 for Flutter overlay
        for k, v in body_kps.items():
            body_kps[k] = {"x": v["x"] / w, "y": v["y"] / h}

        # calculate feedback for this frame
        feedback = []
        for rule in SHOT_RULES.get(shot_type, []):
            try:
                if rule["condition"](body_kps):
                    feedback.append({"type": "warning", "message": rule["message"]})
            except Exception:
                continue

        # add frame data
        results.append({
            "frame_index": frame_index,
            "keypoints": body_kps,
            "feedback": feedback
        })

        frame_index += 1

    cap.release()

    return {
        "shot_type": shot_type,
        "fps": fps,
        "frames": results
    }
