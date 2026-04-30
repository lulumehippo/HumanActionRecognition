"""
predictor.py
------------
Handles keypoint extraction (MediaPipe) and LSTM inference.
Used by app.py to process uploaded videos and return predictions.
"""

import os
import gc
import cv2
import numpy as np
import mediapipe as mp

# ── Constants ─────────────────────────────────────────────────────────────────
CLASSES = [
    "Fall Down",
    "Lying Down",
    "Sit down",
    "Sitting",
    "Stand up",
    "Standing",
    "Walking",
]

NUM_FRAMES   = 30      # fixed sequence length (1 second @ 30 FPS)
NUM_FEATURES = 99      # 33 landmarks × (x, y, z)


# ── MediaPipe Pose ─────────────────────────────────────────────────────────────
_mp_pose = mp.solutions.pose
_pose    = _mp_pose.Pose(static_image_mode=False,
                         model_complexity=1,
                         smooth_landmarks=True)


def extract_keypoints(video_path: str) -> np.ndarray | None:
    """
    Read a video file and extract MediaPipe Pose keypoints for every frame.

    Returns
    -------
    np.ndarray  shape (30, 99), dtype float32
        Fixed-length sequence of pose keypoints.
    None
        If the video could not be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    keypoints_seq = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = _pose.process(img)

        if result.pose_landmarks:
            kp = []
            for lm in result.pose_landmarks.landmark:
                kp.extend([lm.x, lm.y, lm.z])
        else:
            kp = [0.0] * NUM_FEATURES

        keypoints_seq.append(kp)

    cap.release()
    del cap
    gc.collect()

    # ── Normalise to exactly NUM_FRAMES ────────────────────────────────────────
    if len(keypoints_seq) == 0:
        keypoints_seq = [[0.0] * NUM_FEATURES] * NUM_FRAMES
    elif len(keypoints_seq) < NUM_FRAMES:
        last = keypoints_seq[-1]
        while len(keypoints_seq) < NUM_FRAMES:
            keypoints_seq.append(last)
    else:
        keypoints_seq = keypoints_seq[:NUM_FRAMES]

    return np.array(keypoints_seq, dtype=np.float32)


# ── Model loader (lazy, singleton) ────────────────────────────────────────────
_model = None

def get_model(model_path: str = "model/action_model.keras"):
    """Load and cache the Keras model."""
    global _model
    if _model is None:
        import tensorflow as tf          # deferred import (startup speed)
        _model = tf.keras.models.load_model(model_path)
        print(f"[Predictor] Model loaded from '{model_path}'")
    return _model


# ── Main prediction API ────────────────────────────────────────────────────────
def predict(video_path: str, model_path: str = "model/action_model.keras") -> dict:
    """
    End-to-end prediction for a single video file.

    Returns
    -------
    dict
        {
          "success":    bool,
          "label":      str,            # predicted class name
          "confidence": float,          # 0–100 %
          "scores":     list[dict]      # [{label, score}, ...] for all classes
        }
    """
    kp = extract_keypoints(video_path)

    if kp is None:
        return {"success": False, "error": "Could not open video file."}

    if np.all(kp == 0):
        return {"success": False, "error": "No human pose detected in video."}

    model  = get_model(model_path)
    X      = np.expand_dims(kp, axis=0)          # (1, 30, 99)
    probs  = model.predict(X, verbose=0)[0]      # (7,)

    pred_idx    = int(np.argmax(probs))
    pred_label  = CLASSES[pred_idx]
    confidence  = float(probs[pred_idx]) * 100

    scores = [
        {"label": CLASSES[i], "score": round(float(probs[i]) * 100, 2)}
        for i in range(len(CLASSES))
    ]
    scores.sort(key=lambda x: x["score"], reverse=True)

    return {
        "success":    True,
        "label":      pred_label,
        "confidence": round(confidence, 2),
        "scores":     scores,
    }
