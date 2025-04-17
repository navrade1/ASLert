import datetime
import json
import pickle

import numpy as np
import tensorflow as tf
import torch
from keras import Sequential, layers, callbacks
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from preprocess import preprocess

import colorama
from colorama import Fore, Back, Style
colorama.init()

print(f"\n{Back.BLUE}{Fore.WHITE}{'='*80}{Style.RESET_ALL}")
print(f"{Back.BLUE}{Fore.WHITE}{'ASLert RNN Model Training':^80}{Style.RESET_ALL}")
print(f"{Back.BLUE}{Fore.WHITE}{'='*80}{Style.RESET_ALL}\n")

print(f"\n{Back.MAGENTA}{Fore.WHITE} GPU CONFIGURATION {Style.RESET_ALL}")

# try to force TensorFlow to see the GPU
try:
    # First attempt - standard detection
    physical_devices = tf.config.list_physical_devices('GPU')
    
    if len(physical_devices) == 0:
        print(f"{Fore.YELLOW}   → No GPUs found with standard detection, trying alternatives...{Style.RESET_ALL}")
        
        # Try to force CUDA initialization
        import ctypes
        try:
            ctypes.CDLL("cudart64_110.dll")
            print(f"{Fore.YELLOW}   → CUDA runtime loaded manually{Style.RESET_ALL}")
        except:
            print(f"{Fore.YELLOW}   → Failed to load CUDA runtime manually{Style.RESET_ALL}")
        
        # Try again after manual initialization
        physical_devices = tf.config.list_physical_devices('GPU')
    
    if len(physical_devices) > 0:
        print(f"{Back.GREEN}{Fore.BLACK} ✓ Found {len(physical_devices)} GPU(s) {Style.RESET_ALL}")
        for device in physical_devices:
            print(f"{Fore.GREEN}   → Name: {device.name}, Type: {device.device_type}{Style.RESET_ALL}")
            
        # Configure memory growth to avoid consuming all GPU memory
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"{Fore.GREEN}   → Memory growth enabled for all GPUs{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}   → Warning: Could not set memory growth: {e}{Style.RESET_ALL}")
        
        # Enable mixed precision for faster training
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print(f"{Fore.GREEN}   → Mixed precision enabled{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}   → Warning: Could not enable mixed precision: {e}{Style.RESET_ALL}")
        
        # Set TensorFlow to use GPU
        try:
            tf.config.set_visible_devices(physical_devices, 'GPU')
            print(f"{Fore.GREEN}   → GPU set as visible device{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}   → Warning: Could not set visible devices: {e}{Style.RESET_ALL}")
            
        # Verify GPU is being used
        print(f"{Fore.CYAN}   → Verifying GPU availability...{Style.RESET_ALL}")
        with tf.device('/GPU:0'):
            try:
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
                print(f"{Fore.GREEN}   → GPU test successful: Matrix multiplication verified{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Back.RED}{Fore.WHITE} ✗ GPU test failed: {e} {Style.RESET_ALL}")
                print(f"{Fore.YELLOW}   → Falling back to CPU{Style.RESET_ALL}")
    else:
        print(f"{Back.YELLOW}{Fore.BLACK} ⚠ No GPU found, using CPU {Style.RESET_ALL}")
        print(f"{Fore.YELLOW}   → Check if CUDA and cuDNN are properly installed{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}   → Verify that your GPU is CUDA-compatible{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}   → Training will be significantly slower on CPU{Style.RESET_ALL}")
        
        try:
            print(f"{Fore.YELLOW}   → CUDA available (PyTorch): {torch.cuda.is_available()}{Style.RESET_ALL}")
            if torch.cuda.is_available():
                print(f"{Fore.YELLOW}   → CUDA version: {torch.version.cuda}{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}   → GPU device: {torch.cuda.get_device_name(0)}{Style.RESET_ALL}")
        except:
            pass

except Exception as e:
    print(f"{Back.RED}{Fore.WHITE} ✗ Error during GPU configuration: {e} {Style.RESET_ALL}")
    print(f"{Fore.YELLOW}   → Falling back to CPU{Style.RESET_ALL}")

print('')

def create_serving_signature(model):
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 30, 63], dtype=tf.float32, name='hand_landmarks')
    ])
    def serve(landmarks):
        predictions = model(landmarks)
        return {
            "probabilities": predictions,
            "class_id": tf.argmax(predictions, axis=-1)
        }
    return serve

def load_data(metadata_file: str, sequence_length=30) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess data from a metadata file into uniform sequences.

    Args:
        metadata_file (str): Path to the metadata file containing video information and labels.
        sequence_length (int): The number of frames to include in each sequence.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the processed input data (_array of sequences for each video_) and labels (_labeled classifications for each video_).
    """
    all_sequences = []
    all_labels = []

    with open(metadata_file) as f:
        metadata = json.load(f)

        for metadatum in metadata:
            video_path = metadatum['location'] + metadatum['file_name']
            sequences = preprocess(video_path)

            sequences = sequences.reshape(len(sequences), sequence_length, -1)

            all_sequences.append(sequences)
            all_labels.extend([metadatum['label']] * len(sequences))

    label_encoder = LabelEncoder()
    all_labels = label_encoder.fit_transform(all_labels)
    
    return np.concatenate(all_sequences), np.array(all_labels), label_encoder

def create_rnn(num_classes: int, sequence_length=30, num_features: int = 21*3, dropout_rate: int = 0.3) -> Sequential:
    input_shape = (sequence_length, num_features)
    
    print(f"\n{Back.MAGENTA}{Fore.WHITE} MODEL ARCHITECTURE {Style.RESET_ALL}")
    
    # Input shape spec
    inputs = layers.Input(shape=input_shape)
    
    # Create the model using functional API to avoid warnings
    x = layers.Normalization()(inputs)
    print(f"{Fore.CYAN}   → Input normalization applied{Style.RESET_ALL}")
    
    # First RNN layer
    x = layers.SimpleRNN(256, return_sequences=True, activation='tanh', 
                       kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(rate=dropout_rate)(x)
    print(f"{Fore.CYAN}   → First RNN layer: 256 units with batch normalization{Style.RESET_ALL}")

    # Second RNN layer
    x = layers.SimpleRNN(256, return_sequences=False, activation='tanh',
                       kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(rate=dropout_rate)(x)
    print(f"{Fore.CYAN}   → Second RNN layer: 256 units with batch normalization{Style.RESET_ALL}")

    # Dense hidden layer
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(rate=dropout_rate/2)(x)
    print(f"{Fore.CYAN}   → Dense layer: 128 units{Style.RESET_ALL}")

    # Classification layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    print(f"{Fore.CYAN}   → Output layer: {num_classes} classes with softmax activation{Style.RESET_ALL}\n")
    
    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def train(X, y, model, epochs=100, batch_size=32, validation_split=0.2):
    print(f"\n{Back.MAGENTA}{Fore.WHITE} TRAINING PROCESS {Style.RESET_ALL}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"{Fore.CYAN}   → Data split: {len(X_train)} training samples, {len(X_test)} test samples{Style.RESET_ALL}")
    
    # Apply data augmentation to training data
    print(f"{Back.CYAN}{Fore.BLACK} Data Augmentation {Style.RESET_ALL}")
    X_train_augmented = []
    y_train_augmented = []
    X_train_augmented.append(X_train)
    y_train_augmented.append(y_train)
    print(f"{Fore.CYAN}   → Original data added: {len(X_train)} samples{Style.RESET_ALL}")
    
    # Add noise - for the sake of robustness
    for noise_scale in [0.05, 0.1]:
        noisy_X = []
        for seq in X_train:
            noise = np.random.normal(0, noise_scale, seq.shape)
            noisy_seq = seq + noise
            noisy_X.append(noisy_seq)
        X_train_augmented.append(np.array(noisy_X))
        y_train_augmented.append(y_train)
        print(f"{Fore.CYAN}   → Added noise augmentation (scale={noise_scale}): {len(noisy_X)} samples{Style.RESET_ALL}")
    
    X_train = np.concatenate(X_train_augmented)
    y_train = np.concatenate(y_train_augmented)
    
    print(f"{Fore.GREEN}   → Final dataset: {X_train.shape[0]} training samples ({X_train.shape[0]/len(X):.1f}x original){Style.RESET_ALL}")
    
    # Set up callbacks for better training
    print(f"{Back.CYAN}{Fore.BLACK} Training Configuration {Style.RESET_ALL}")
    log_dir = RNN_PATH + 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    callbacks_list = [
        callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        
        # Early stopping to prevent overfitting
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        ),
        
        # Reduce learning rate when training plateaus
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
    ]
    
    print(f"{Fore.CYAN}   → Epochs: {epochs}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}   → Batch size: {batch_size}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}   → Validation split: {validation_split}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}   → Early stopping patience: 15 epochs{Style.RESET_ALL}")
    print(f"{Fore.CYAN}   → TensorBoard logs: {log_dir}{Style.RESET_ALL}")
    
    print(f"\n{Back.GREEN}{Fore.BLACK} Starting Training {Style.RESET_ALL}")

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks_list,
        shuffle=True,
        verbose=1
    )
    
    # Evaluate on test set
    print(f"\n{Back.MAGENTA}{Fore.WHITE} MODEL EVALUATION {Style.RESET_ALL}")
    loss, accuracy = model.evaluate(X_test, y_test)
    
    # Display results with colored formatting
    print(f"\n{Back.CYAN}{Fore.BLACK} Test Results {Style.RESET_ALL}")
    
    # Format accuracy with color based on performance
    if accuracy >= 0.9:
        acc_color = f"{Back.GREEN}{Fore.BLACK}"
    elif accuracy >= 0.7:
        acc_color = f"{Back.BLUE}{Fore.WHITE}"
    else:
        acc_color = f"{Back.RED}{Fore.WHITE}"
        
    # Format loss with color based on performance
    if loss <= 0.3:
        loss_color = f"{Back.GREEN}{Fore.BLACK}"
    elif loss <= 0.7:
        loss_color = f"{Back.BLUE}{Fore.WHITE}"
    else:
        loss_color = f"{Back.RED}{Fore.WHITE}"
    
    print(f"{acc_color} Accuracy: {accuracy:.2%} {Style.RESET_ALL}")
    print(f"{loss_color} Loss: {loss:.4f} {Style.RESET_ALL}\n")

    return model, history

def save(model) -> str:
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    export_path = f'{RNN_PATH}model_{timestamp}.keras'
    model.save(export_path)
    return export_path

print(f"\n{Back.BLUE}{Fore.WHITE} INITIALIZING TRAINING {Style.RESET_ALL}")

# assumes running from home dir
metadata_file = 'data/.labels.json'
RNN_PATH = 'src/models/rnn/'

print(f"{Fore.CYAN}   → Loading data from: {metadata_file}{Style.RESET_ALL}")
print(f"{Fore.CYAN}   → Model output path: {RNN_PATH}{Style.RESET_ALL}")

# Load and preprocess data
print(f"\n{Back.MAGENTA}{Fore.WHITE} DATA LOADING {Style.RESET_ALL}")
X, y, label_encoder = load_data(metadata_file)
print(f"{Fore.GREEN}   → Loaded {len(X)} sequences with {len(label_encoder.classes_)} classes{Style.RESET_ALL}")

# Create the model
model = create_rnn(
    num_classes=len(label_encoder.classes_)
)

# Display model summary
print(f"\n{Back.MAGENTA}{Fore.WHITE} MODEL SUMMARY {Style.RESET_ALL}")
model.summary()

# Display class labels
print(f"\n{Back.MAGENTA}{Fore.WHITE} CLASS LABELS {Style.RESET_ALL}")
for i, label in enumerate(label_encoder.classes_):
    print(f"{Fore.CYAN}   → Class {i}: {label}{Style.RESET_ALL}")

# Train the model
trained_model, history = train(X, y, model)

# Save the label encoder
print(f"\n{Back.MAGENTA}{Fore.WHITE} SAVING MODEL {Style.RESET_ALL}")
with open(RNN_PATH + 'label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"{Fore.GREEN}   → Label encoder saved to {RNN_PATH}label_encoder.pkl{Style.RESET_ALL}")

# Save the model
MODEL_PATH = save(trained_model)
print(f"{Back.GREEN}{Fore.BLACK} ✓ Model successfully saved to {MODEL_PATH} {Style.RESET_ALL}")

print(f"\n{Back.BLUE}{Fore.WHITE}{'='*80}{Style.RESET_ALL}")
print(f"{Back.BLUE}{Fore.WHITE}{'Training Complete':^80}{Style.RESET_ALL}")
print(f"{Back.BLUE}{Fore.WHITE}{'='*80}{Style.RESET_ALL}\n")
