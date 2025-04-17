import time
import queue
from collections import deque
from threading import Thread, Lock
import gc
import psutil
import traceback as tb
import sys

import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

import colorama
from colorama import Fore, Style
colorama.init()

sys.path.append('C:/Users/nikoa/source/repos/ASLert/src')
from preprocess import HAND_LANDMARKER_CONFIG

# Custom exception handler to print exceptions without crashing
def custom_excepthook(type, value, traceback):
    print(f'{Fore.RED}Uncaught exception: {type.__name__}: {value}{Style.RESET_ALL}')
    tb.print_tb(traceback)

sys.excepthook = custom_excepthook

# Reduce TensorFlow logging verbosity
tf.debugging.set_log_device_placement(False)
tf.config.set_soft_device_placement(True)
SEQUENCE_LENGTH = 30
MODEL_PATH = 'src/models/rnn/model_20250416-190035.keras'
LABELS = ["big emergency", "harsh pain", "help", "help me", 
          "help you", "light pain", "small emergency"]

# ==== RNN Model ====
print(f"\n{Fore.CYAN}=== Loading model from {MODEL_PATH} ==={Style.RESET_ALL}")

# feedback mgr warn still present...  single i/o doesnt work
inputs = tf.keras.Input(shape=(SEQUENCE_LENGTH, 63), dtype=tf.float32)
original_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
outputs = original_model(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(f"{Fore.CYAN}=== Model loaded and compiled ==={Style.RESET_ALL}\n")

pred_queue = queue.Queue(maxsize=1)  # single-item buffer

@tf.function
def predict_wrapper(sequence):
    return model(sequence)

# AppState class handles global var access, especially w/ threads
class AppState:
    def __init__(self):
        self.landmark_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.current_prediction = 'Ready...'
        self.lock = Lock()
        self.last_frame = None

app_state = AppState()

def prediction_worker(cap):
    while True:
        try:
            sequence = pred_queue.get(timeout=0.5)
            if sequence is None: break # exit sig
                
            # run prediction in a controlled environment
            preds = predict_wrapper(sequence)
            pred_index = np.argmax(preds)
            
            if 0 <= pred_index < len(LABELS):
                with app_state.lock:
                    app_state.current_prediction = LABELS[pred_index]
            else:
                print(f"{Fore.YELLOW}Warning: Invalid prediction index {pred_index}{Style.RESET_ALL}")
                
        except queue.Empty: continue # wait for next item
        except Exception as e:
            print(f'{Fore.RED}Prediction error: {e}{Style.RESET_ALL}')
        finally:
            # Clear TensorFlow session to prevent memory leaks
            tf.keras.backend.clear_session()
            
            if not cap.isOpened(): break

def result_callback(result, output_image: mp.Image, timestamp_ms: int):
    try:
        # process hand landmarks - model expects 21*3 = 63 coordinates
        frame_data = np.zeros(63)  # init w/ zeros for one hand's worth of data
        
        if result.hand_landmarks:
            # typically detect 2 hands
            detected_hands = result.hand_landmarks[:HAND_LANDMARKER_CONFIG['num_hands']]
            
            if len(detected_hands) == 1:
                # If only one hand is detected, use its data directly
                hand_data = np.array([[lm.x, lm.y, lm.z] for lm in detected_hands[0]]).flatten()
                if len(hand_data) == 63:
                    frame_data = hand_data
            elif len(detected_hands) >= 2:
                # If two hands are detected (for signs like "big emergency"),
                # we need to combine the data to fit 63-feature model input
                
                # Calculate hand centroids to determine which is left/right
                centroids = []
                for hand in detected_hands[:2]:
                    centroid = np.mean([[lm.x, lm.y, lm.z] for lm in hand], axis=0)
                    centroids.append(centroid)
                
                # L -> R
                hand_indices = np.argsort([c[0] for c in centroids])
                left_hand = detected_hands[hand_indices[0]]
                right_hand = detected_hands[hand_indices[1]]
                
                # 2-handed signs will use combo approach...
                # - R hand full data
                # - L hand key landmarks
                #   - these will be slightly modded
                
                right_hand_data = np.array([[lm.x, lm.y, lm.z] for lm in right_hand]).flatten()
                left_hand_key_points = np.array([
                    [left_hand[4].x, left_hand[4].y, left_hand[4].z],  # Thumb tip
                    [left_hand[8].x, left_hand[8].y, left_hand[8].z],  # Index tip
                    [left_hand[12].x, left_hand[12].y, left_hand[12].z],  # Middle tip
                ]).flatten()
                
                frame_data = right_hand_data
                
                # encode L hand info by slightly modding specific landmarks
                # - this preserves overall structure while encoding presence of two hands
                if len(left_hand_key_points) == 9:  # 3 points * 3 coordinates
                    # mod pinky landmarks to encode L hand key points
                    # (landmarks 17-20 are the pinky finger)
                    for i in range(min(9, len(left_hand_key_points))):
                        idx = 51 + i  # Start at pinky base (landmark 17 * 3 = 51)
                        if idx < 63:
                            # blend vals to preserve some original data
                            frame_data[idx] = (frame_data[idx] * 0.2) + (left_hand_key_points[i] * 0.8)
            
        with app_state.lock:
            app_state.landmark_buffer.append(frame_data)
            
            # _only_ queue for prediction if we have a full sequence and queue isn't occupied
            if len(app_state.landmark_buffer) == SEQUENCE_LENGTH:
                # match model input shape (1, 30, 63)
                sequence = np.array(app_state.landmark_buffer).reshape(1, SEQUENCE_LENGTH, 63)
                
                # queue for prediction
                if not pred_queue.full():
                    pred_queue.put(sequence.astype(np.float32))

    except Exception as e:
        print(f"{Fore.RED}Callback error: {e}{Style.RESET_ALL}")
        tb.print_exc()

def _get_processed_frame():
    with app_state.lock:
        if app_state.last_frame is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Initializing...", (20, 50),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return blank
            
        frame = np.copy(app_state.last_frame)
        cv2.putText(frame, app_state.current_prediction, (20, 50),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame

def cleanup_resources():
    """Clean up all resources to prevent memory leaks"""
    print(f"{Fore.CYAN}Cleaning up resources...{Style.RESET_ALL}")
    
    # force garbage collection...
    gc.collect()
    
    # try to clean up MediaPipe threads if possible
    try:
        current_process = psutil.Process()
        for thread in current_process.threads():
            # can't kill threads, but... can log
            if 'MediaPipe' in str(thread):
                print(f"{Fore.CYAN}Found MediaPipe thread: {thread}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.YELLOW}Error during thread cleanup: {e}{Style.RESET_ALL}")

try:
    print(f"\n{Fore.GREEN}=== Starting ASL Interpreter... ==={Style.RESET_ALL}")
    
    cv2.namedWindow('ASL Interpreter', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ASL Interpreter', 640, 480)

    init_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(init_frame, "Initializing camera...", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('ASL Interpreter', init_frame)
    cv2.waitKey(1)
    
    print(f"{Fore.CYAN}=== Opening camera... ==={Style.RESET_ALL}")
    cap = cv2.VideoCapture(0)
    
    camera_init_start = time.time()
    while not cap.isOpened():
        if time.time() - camera_init_start > 5:  # 5 second timeout
            raise RuntimeError("Failed to open camera after 5 seconds")
        print(f"{Fore.YELLOW}Waiting for camera...{Style.RESET_ALL}")
        time.sleep(0.5)
        cap = cv2.VideoCapture(0)
    
    print(f"{Fore.GREEN}=== Camera opened successfully ==={Style.RESET_ALL}")
    
    timestamp = 0

    mp_config = HAND_LANDMARKER_CONFIG.copy()
    mp_config['running_mode'] = vision.RunningMode.LIVE_STREAM
    mp_config['result_callback'] = result_callback
    
    print(f"{Fore.CYAN}=== Setting up MediaPipe ==={Style.RESET_ALL}")
    
    options = vision.HandLandmarkerOptions(**mp_config)

    with vision.HandLandmarker.create_from_options(options) as detector:
        prediction_interval = 0.04  # 25fps -> 40ms
        last_pred_time = time.time()
        
        worker = Thread(target=prediction_worker, args=(cap,), daemon=True)
        worker.start()
        print(f"{Fore.GREEN}=== Worker thread started ==={Style.RESET_ALL}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"{Fore.RED}Failed to capture frame{Style.RESET_ALL}")
                break

            # store frame in app_state for callback later
            with app_state.lock:
                app_state.last_frame = frame
            
            # disp frame directly in the main thread
            processed_frame = _get_processed_frame()
            cv2.imshow('ASL Interpreter', processed_frame)

            # ctrl frame rate
            elapsed = time.time() - last_pred_time
            sleep_time = max(0, prediction_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # mp frame processing
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detector.detect_async(mp_image, timestamp)
                del mp_image
                del rgb_frame
            except Exception as e:
                print(f"{Fore.RED}Error in detect_async: {e}{Style.RESET_ALL}")
                continue
            
            timestamp = (timestamp + 40) % (2**32)
            last_pred_time = time.time()
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(f"{Fore.CYAN}Quit key pressed{Style.RESET_ALL}")
                break
            
            # force periodic garbage collection
            if timestamp % 1000 == 0:
                gc.collect()
                
except Exception as e:
    print(f'{Fore.RED}Error: {e}{Style.RESET_ALL}')
    tb.print_exc()

finally:
    print(f"\n{Fore.CYAN}=== Shutting down... ==={Style.RESET_ALL}")
    try:
        if not pred_queue.full():
            pred_queue.put(None)
    except:
        pass
        
    if 'cap' in locals() and cap.isOpened():
        cap.release()
    
    cv2.destroyAllWindows()
    
    if 'worker' in locals():
        worker.join(timeout=2)
        
    if 'detector' in locals():
        try:
            detector.close()
            del detector
        except:
            pass
    
    cleanup_resources()
    print(f"{Fore.GREEN}=== Cleanup complete ==={Style.RESET_ALL}\n")
