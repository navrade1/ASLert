import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# === MediaPipe Setup ===
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

# Model path configuration
model_path = './src/models/hand_landmarker.task'

# Check if the model file exists
if not os.path.exists(model_path):
    raise RuntimeError(f"Unable to find the model file at {model_path}. Please make sure the model is located at this path.")

HAND_LANDMARKER_CONFIG = {
    'base_options': BaseOptions(model_asset_path=model_path),
    'running_mode': VisionRunningMode.VIDEO,
    'num_hands': 1,
    'min_hand_detection_confidence': 0.5,
    'min_hand_presence_confidence': 0.5,
    'min_tracking_confidence': 0.5
}

DEBUG_MODE = True  # Set to True to enable debugging visualization

# === Video Processing ===
def process_video(video_path: Path):
    options = HandLandmarkerOptions(**HAND_LANDMARKER_CONFIG)
    with HandLandmarker.create_from_options(options) as detector:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Error opening video file: {video_path}")
        results = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            result = detector.detect_for_video(mp_image, frame_timestamp_ms)
            if result.hand_landmarks:
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]])
                results.append(landmarks.flatten())  # flatten to 63D vector (21 landmarks x 3)
                if DEBUG_MODE:
                    visualize_results(frame, result)
            else:
                print(f"No hand landmarks detected at frame {frame_timestamp_ms}")
        cap.release()
        cv2.destroyAllWindows()
    if len(results) == 0:
        print("No hand landmarks detected in the video.")
    return np.array(results)

# === Visualization for Debugging ===
def visualize_results(frame, detection_result):
    for i, landmarks in enumerate(detection_result.hand_landmarks):
        handedness = detection_result.handedness[i][0]
        if handedness.score < HAND_LANDMARKER_CONFIG['min_hand_presence_confidence']:
            continue
        cv2.putText(
            frame,
            f'{handedness.category_name} {handedness.score:.2f}',
            (10, 30*(i+1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )
        cv2.imshow('Hand Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# === Test the Code ===
video_path = Path("sample_hand_video.mp4")  # Replace with your actual video path
raw_data = process_video(video_path)

# Check if there are any frames processed
if len(raw_data) > 0:
    print(f"Number of frames processed: {len(raw_data)}")
else:
    print("No frames with hand landmarks were processed.")
