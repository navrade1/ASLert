import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque

# Tasks API Reference: https://ai.google.dev/edge/api/mediapipe/python/mp/tasks?hl=en

# Configuration packaged here for easy selective modification & unpacking
HAND_LANDMARKER_CONFIG = {
    # assumes running any code from home dir
    'base_options': python.BaseOptions(model_asset_path='src/models/hand_landmarker.task'),
    'running_mode': vision.RunningMode.VIDEO,
    'num_hands': 2,
    'min_hand_detection_confidence': 0.5,
    'min_hand_presence_confidence': 0.5,
    'min_tracking_confidence': 0.5
}

#===================================================================

DEBUG_MODE=False

#===================================================================

# Track hand positions across last 5 frames for temporal consistency
POSITION_HISTORY = deque(maxlen=5)

def process_video(video_path: str) -> np.ndarray:
    """
    Returns extracted hand landmarks.
    """
    with vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(**HAND_LANDMARKER_CONFIG)
    ) as detector:

        cap = cv2.VideoCapture(video_path)
        hand_landmarker_results = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Process frame
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            result = detector.detect_for_video(mp_image, frame_timestamp_ms)
            if result.hand_landmarks:
                hands = []
                frame_center = np.array([0.5, 0.5, 0.5]) # normalized coords

                # Noise filtering: only process _most central_ hands (amount specified in config)
                for landmarks in result.hand_landmarks:
                    centroid = np.mean([[lm.x, lm.y, lm.z] for lm in landmarks], axis=0)
                    distance = np.linalg.norm(centroid - frame_center)
                    hands.append((distance, landmarks))
                hands.sort(key=lambda x: x[0])
                central_hands = [
                    landmarks for _, landmarks in hands[:HAND_LANDMARKER_CONFIG['num_hands']]
                ]

                # Apply temporal consistency filtering
                filtered_hands = get_most_consistent_hands(central_hands)

                for hand_landmarks in filtered_hands:
                    hand_landmarker_results.append(
                        np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
                    )

                POSITION_HISTORY.append(filtered_hands)

                if DEBUG_MODE:
                    visualize_results(frame, result)

        cap.release()

    cv2.destroyAllWindows()
    return np.array(hand_landmarker_results)

def get_most_consistent_hands(current_hands):
    """
    Filters hands based on temporal consistency with previous frames (hands with the most stable movement).
    """
    if not POSITION_HISTORY:
        return current_hands[:HAND_LANDMARKER_CONFIG['num_hands']]

    prev_positions = POSITION_HISTORY[-1]
    consistency_scores = []

    for hand in current_hands:
        centroid = np.mean([[lm.x, lm.y, lm.z] for lm in hand], axis=0)
        min_distance = float('inf')

        for prev_hand in prev_positions:
            prev_centroid = np.mean([[lm.x, lm.y, lm.z] for lm in prev_hand], axis=0)
            distance = np.linalg.norm(centroid - prev_centroid)
            min_distance = min(min_distance, distance)

        consistency_scores.append(min_distance)

    sorted_indices = np.argsort(consistency_scores)
    return [
        current_hands[i] for i in sorted_indices[:HAND_LANDMARKER_CONFIG['num_hands']]
    ]

def visualize_results(frame, detection_result):
    """
    Display confidence scores of hand landmarks on processed videos.
    - Set DEBUG_MODE=True to use this solution.
    """
    for i in range(len(detection_result.hand_landmarks)):

        handedness = detection_result.handedness[i][0]
        if handedness.score < HAND_LANDMARKER_CONFIG['min_hand_presence_confidence']:
            continue

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

def preprocess(video_path: str, sequence_length: int = 30) -> np.ndarray:
    """
    Process video into sequences with smart padding.
    
    Args:
        video_path (str): ...
        sequence_length (int): Dictates number of frames in each sequence. "How small do you want your video to be split up into?"
    
    Returns:
        np.ndarray: Array of sequences for a video.
    """

    landmarks = process_video(video_path)
    if len(landmarks) < sequence_length:
        pad = [(0, sequence_length - len(landmarks)), (0,0), (0,0)]
        landmarks = np.pad(landmarks, pad)

    return gen_sequences(landmarks, sequence_length)

def gen_sequences(data: np.ndarray, window_size: int) -> np.ndarray:
    """Generate temporal sequences from a video using specified window size (length of each sequence)."""

    num_sequences = len(data) - window_size + 1
    if num_sequences < 1:
        return data[np.newaxis, ...]
    return np.array([data[i:i+window_size] for i in range(num_sequences)])

__all__ = ['preprocess', 'HAND_LANDMARKER_CONFIG']
