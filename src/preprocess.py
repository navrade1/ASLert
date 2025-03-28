import json
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

# Tasks API Reference: https://ai.google.dev/edge/api/mediapipe/python/mp/tasks?hl=en

# Configuration packaged here for easy modification & unpacking
# - running_mode: set to `VisionRunningMode.LIVE_STREAM` for webcam input
HAND_LANDMARKER_CONFIG = {
    'base_options': BaseOptions(model_asset_path='./models/hand_landmarker.task'),
    'running_mode': VisionRunningMode.VIDEO,
    'num_hands': 2,
    'min_hand_detection_confidence': 0.5,
    'min_hand_presence_confidence': 0.5,
    'min_tracking_confidence': 0.5
}

DEBUG_MODE = False

def process_video(video_path: Path):
    options = HandLandmarkerOptions(**HAND_LANDMARKER_CONFIG)
    with HandLandmarker.create_from_options(options) as detector:

        cap = cv2.VideoCapture(str(video_path))
        
        results = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Process frame
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            #============================================================================
            # IMPLEMENT LIVE_STREAM MODE LATER (w/ `detect_async()` - https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HandLandmarker?hl=en#detect_async)
            #============================================================================
            result = detector.detect_for_video(mp_image, frame_timestamp_ms)
            if result.hand_landmarks:
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]])
                results.append(landmarks)
                if DEBUG_MODE:
                    visualize_results(frame, result)
        cap.release()

    cv2.destroyAllWindows()
    return np.array(results)

#===========================================================
#                      Visualization
#   - Use (DEBUG_MODE=True) to verify landmark detection
#===========================================================
def visualize_results(frame, detection_result):
    for i, landmarks in enumerate(detection_result.hand_landmarks):
        #  ^^^^^^^^^
        # landmarks used for drawing in legacy mode

        handedness = detection_result.handedness[i][0]
        if handedness.score < HAND_LANDMARKER_CONFIG['min_hand_presence_confidence']:
            continue

        #=================================================================
        # DEPRECATED
        # mp.solutions.drawing_utils.draw_landmarks(
        #     frame,
        #     landmarks,
        #     mp.solutions.hands.HAND_CONNECTIONS,
        #     landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        #     connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(250, 44, 90), thickness=2)
        # )
        #=================================================================

        cv2.putText(
            frame,
            f'{handedness.category_name} {handedness.score:.2f}',
            (10, 30*(i+1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,0,0),
            2
        )
        
        cv2.imshow('Hand Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def preprocess_data(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)

__all__ = ['process_video', 'preprocess_data']