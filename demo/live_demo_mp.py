import time
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks.python import vision
from collections import deque

import sys
sys.path.append('C:/Users/nikoa/source/repos/ASLert/src')
from preprocess import HAND_LANDMARKER_CONFIG

# ==== Configuration ====

SEQUENCE_LENGTH = 30  # Must match training
MODEL_PATH = 'src/models/rnn/model_20250416-114012.keras'
LABELS = ["small emergency", "harsh pain", "help", "help me", 
          "help you", "light pain", "big emergency"]

# ==== Global State ====
current_prediction = "Ready..."
landmark_buffer = deque(maxlen=SEQUENCE_LENGTH)

# ==== MediaPipe Setup ====
def result_callback(result, output_image: mp.Image, timestamp_ms: int):
    global current_prediction, landmark_buffer
    frame_data = np.zeros(21*3)  # 21 landmarks * 3 coordinates
    
    # process landmarks
    if result.hand_landmarks:
        for i, hand in enumerate(result.hand_landmarks[:HAND_LANDMARKER_CONFIG['num_hands']]):
            start = i*21*3
            end = start + 21*3
            frame_data[start:end] = np.array([[lm.x, lm.y, lm.z] for lm in hand]).flatten()
    
    landmark_buffer.append(frame_data)
    
    # make prediction
    if len(landmark_buffer) == SEQUENCE_LENGTH:
        sequence = np.array(landmark_buffer).reshape(1, SEQUENCE_LENGTH, 21*3)
        start = time.time()
        preds = model.predict(sequence, verbose=0)
        print(f'time taken to predict: {time.time() - start:.2f}')
        current_prediction = LABELS[np.argmax(preds)]

HAND_LANDMARKER_CONFIG['result_callback']=result_callback

# ==== RNN Model ===-
model = tf.keras.models.load_model(MODEL_PATH)

# ==== Webcam Processing ====
cap = cv2.VideoCapture(0)
timestamp = 0
HAND_LANDMARKER_CONFIG['running_mode'] = vision.RunningMode.LIVE_STREAM

with vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(**HAND_LANDMARKER_CONFIG)
) as detector:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Webcam feed lost")
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detector.detect_async(mp_image, timestamp)
        timestamp += 33  # 30 FPS = ~33ms per frame
        
        cv2.putText(frame, current_prediction, (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('ASL Interpreter', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Exiting...')
            break

cap.release()
cv2.destroyAllWindows()
del current_prediction, landmark_buffer
