import cv2
import queue
import time
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks.python import vision
from collections import deque
from threading import Thread, Lock
import gc
import psutil  # For process management
from keras import backend as K  # Import K for session clearing

import sys
sys.path.append('C:/Users/nikoa/source/repos/ASLert/src')
from preprocess import HAND_LANDMARKER_CONFIG

# ==== Configuration ====
# Custom exception handler to print exceptions without crashing
def custom_excepthook(type, value, traceback):
    print(f'Uncaught exception: {type.__name__}: {value}')
    import traceback as tb
    tb.print_tb(traceback)

sys.excepthook = custom_excepthook

# Reduce TensorFlow logging verbosity
tf.get_logger().setLevel('ERROR')
tf.debugging.set_log_device_placement(False)  # Disable verbose device placement logs
tf.config.set_soft_device_placement(True)
SEQUENCE_LENGTH = 30
MODEL_PATH = 'src/models/rnn/model_20250416-114012.keras'
LABELS = ["small emergency", "harsh pain", "help", "help me", 
          "help you", "light pain", "big emergency"]

# ==== RNN Model ====
model = tf.keras.models.load_model(MODEL_PATH)
pred_queue = queue.Queue(maxsize=1)  # Single-item buffer

@tf.function
def predict_wrapper(sequence):
    return model(sequence)

class AppState:
    def __init__(self):
        self.landmark_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.current_prediction = 'Ready...'
        self.lock = Lock()
        self.last_frame = None  # Store the last frame here to avoid global variable issues

app_state = AppState()

def prediction_worker(cap):
    while True:
        try:
            sequence = pred_queue.get(timeout=0.5)
            if sequence is None:  # Exit signal
                break
                
            # Run prediction in a controlled environment
            preds = predict_wrapper(sequence)
            pred_index = np.argmax(preds)
            
            if 0 <= pred_index < len(LABELS):
                with app_state.lock:
                    app_state.current_prediction = LABELS[pred_index]
            else:
                print(f"Warning: Invalid prediction index {pred_index}")
                
        except queue.Empty:
            # Just continue waiting for the next item
            continue
        except Exception as e:
            print(f'Prediction error: {e}')
        finally:
            # Clear TensorFlow session to prevent memory leaks
            tf.keras.backend.clear_session()  # Use tf.keras.backend instead of K
            
            # Check if we should exit
            if not cap.isOpened():
                break

def result_callback(result, output_image: mp.Image, timestamp_ms: int):
    try:
        # Process hand landmarks - model expects 21*3 = 63 coordinates
        frame_data = np.zeros(63)  # Initialize with zeros for one hand's worth of data
        
        if result.hand_landmarks:
            detected_hands = result.hand_landmarks[:HAND_LANDMARKER_CONFIG['num_hands']]
            
            if len(detected_hands) == 1:
                # If only one hand is detected, use its data directly
                hand_data = np.array([[lm.x, lm.y, lm.z] for lm in detected_hands[0]]).flatten()
                if len(hand_data) == 63:
                    frame_data = hand_data
            elif len(detected_hands) >= 2:
                # If two hands are detected (for signs like "big emergency")
                # We need to combine the data to fit the 63-feature model input
                
                # Calculate hand centroids to determine which is left/right
                centroids = []
                for hand in detected_hands[:2]:  # Use at most 2 hands
                    centroid = np.mean([[lm.x, lm.y, lm.z] for lm in hand], axis=0)
                    centroids.append(centroid)
                
                # Sort hands by x-coordinate (left to right)
                hand_indices = np.argsort([c[0] for c in centroids])
                left_hand = detected_hands[hand_indices[0]]
                right_hand = detected_hands[hand_indices[1]]
                
                # For two-handed signs, we'll use a combination approach:
                # - Use the full data from the dominant hand (right hand)
                # - For key landmarks from the second hand, we'll encode them in specific ways
                
                # Get the full right hand data
                right_hand_data = np.array([[lm.x, lm.y, lm.z] for lm in right_hand]).flatten()
                
                # Get key points from left hand (e.g., thumb tip, index tip)
                left_hand_key_points = np.array([
                    [left_hand[4].x, left_hand[4].y, left_hand[4].z],  # Thumb tip
                    [left_hand[8].x, left_hand[8].y, left_hand[8].z],  # Index tip
                    [left_hand[12].x, left_hand[12].y, left_hand[12].z],  # Middle tip
                ]).flatten()
                
                # Combine the data - use right hand as base, then modify specific landmarks
                # to encode the presence and position of the left hand
                frame_data = right_hand_data
                
                # Encode left hand information by slightly modifying specific landmarks
                # This is a heuristic approach that preserves the overall structure
                # while encoding the presence of two hands
                if len(left_hand_key_points) == 9:  # 3 points * 3 coordinates
                    # Modify the pinky landmarks to encode left hand key points
                    # (landmarks 17-20 are the pinky finger)
                    for i in range(min(9, len(left_hand_key_points))):
                        idx = 51 + i  # Start at pinky base (landmark 17 * 3 = 51)
                        if idx < 63:
                            # Blend the values to preserve some of the original data
                            frame_data[idx] = (frame_data[idx] * 0.2) + (left_hand_key_points[i] * 0.8)
            
        with app_state.lock:
            # Add the frame data to our buffer
            app_state.landmark_buffer.append(frame_data)
            
            # Only queue for prediction if we have a full sequence and the queue isn't full
            if len(app_state.landmark_buffer) == SEQUENCE_LENGTH:
                # Reshape to match model input shape (1, 30, 63)
                sequence = np.array(app_state.landmark_buffer).reshape(1, SEQUENCE_LENGTH, 63)
                
                # Queue for prediction
                if not pred_queue.full():
                    pred_queue.put(sequence.astype(np.float32))
    except Exception as e:
        print(f"Callback error: {e}")
        import traceback
        traceback.print_exc()

def _get_processed_frame():
    with app_state.lock:
        if app_state.last_frame is None:
            # Return a blank frame if we don't have one yet
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Initializing...", (20, 50),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return blank
            
        # Create a copy to avoid modifying the original
        frame = np.copy(app_state.last_frame)
        cv2.putText(frame, app_state.current_prediction, (20, 50),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame

HAND_LANDMARKER_CONFIG['result_callback'] = result_callback
HAND_LANDMARKER_CONFIG['running_mode'] = vision.RunningMode.LIVE_STREAM

def cleanup_resources():
    """Clean up all resources to prevent memory leaks"""
    print("Cleaning up resources...")
    
    # Force garbage collection
    gc.collect()
    
    # Try to clean up MediaPipe threads if possible
    try:
        current_process = psutil.Process()
        for thread in current_process.threads():
            # We can't actually kill threads from here, but we can log them
            if 'MediaPipe' in thread.name:
                print(f"Found MediaPipe thread: {thread.name}")
    except Exception as e:
        print(f"Error during thread cleanup: {e}")

try:
    print("Starting ASL Interpreter...")
    
    # Create window early to ensure it appears
    cv2.namedWindow('ASL Interpreter', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ASL Interpreter', 640, 480)
    
    # Show initialization message
    init_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(init_frame, "Initializing camera...", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('ASL Interpreter', init_frame)
    cv2.waitKey(1)  # This is crucial to actually display the window
    
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    
    # Add timeout for camera initialization
    camera_init_start = time.time()
    while not cap.isOpened():
        if time.time() - camera_init_start > 5:  # 5 second timeout
            raise RuntimeError("Failed to open camera after 5 seconds")
        print("Waiting for camera...")
        time.sleep(0.5)
        cap = cv2.VideoCapture(0)
    
    print("Camera opened successfully")
    
    timestamp = 0
    
    # Configure MediaPipe for live streaming
    options = vision.HandLandmarkerOptions(**HAND_LANDMARKER_CONFIG)
    options.running_mode = vision.RunningMode.LIVE_STREAM
    options.result_callback = result_callback

    with vision.HandLandmarker.create_from_options(options) as detector:
        prediction_interval = 0.04  # 25fps -> 40ms
        last_pred_time = time.time()
        
        # Start prediction worker thread
        worker = Thread(target=prediction_worker, args=(cap,), daemon=True)
        worker.start()
        print("Worker thread started")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            # Store the frame in app_state for the callback to use
            with app_state.lock:
                app_state.last_frame = frame
            
            # Display the frame directly in the main thread
            processed_frame = _get_processed_frame()
            cv2.imshow('ASL Interpreter', processed_frame)

            # Control frame rate
            elapsed = time.time() - last_pred_time
            sleep_time = max(0, prediction_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Process frame with MediaPipe
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detector.detect_async(mp_image, timestamp)
                # Explicitly delete the image to free memory
                del mp_image
                del rgb_frame
            except Exception as e:
                print(f"Error in detect_async: {e}")
                continue
            
            timestamp = (timestamp + 40) % (2**32)
            last_pred_time = time.time()
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit key pressed")
                break
            
            # Force garbage collection periodically
            if timestamp % 1000 == 0:
                gc.collect()
                
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
finally:
    print("Shutting down...")
    # Signal the worker thread to exit
    try:
        if not pred_queue.full():
            pred_queue.put(None)
    except:
        pass
        
    # Release camera
    if 'cap' in locals() and cap.isOpened():
        cap.release()
        
    # Close all OpenCV windows
    cv2.destroyAllWindows()
    
    # Wait for worker thread to finish
    if 'worker' in locals():
        worker.join(timeout=2)
        
    # Close and delete the detector
    if 'detector' in locals():
        try:
            detector.close()
            del detector
        except:
            pass
    
    # Final cleanup
    cleanup_resources()
    print("Cleanup complete")
