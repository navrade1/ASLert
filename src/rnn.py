import json
import numpy as np
import pickle
import tensorflow as tf
import datetime
from keras import Sequential, layers, callbacks
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from preprocess import preprocess

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

def create_rnn(num_classes: int, sequence_length=30, num_features: int = 21*3, dropout_rate: int = 0.2) -> Sequential:
    input_shape = (sequence_length, num_features)
    
    model = Sequential([
        # input specification/normalization
        layers.Input(shape=input_shape),
        layers.Normalization(input_shape=input_shape),

        # RNN - temporal seq processing -> vector
        layers.SimpleRNN(64, return_sequences=False),

        # regularization
        layers.Dropout(rate=dropout_rate),

        # classification
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def train(X, y, model, epochs=50, batch_size=32, validation_split=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    log_dir = RNN_PATH + 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[tensorboard_callback]
    )
    
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Train accuracy: {accuracy:.2f}')
    print(f'Loss: {loss:.2f}')

    return model

def save(model) -> str:
    export_path = f'{RNN_PATH}model_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.keras'
    model.save(export_path)
    return export_path

def add_noise(X: np.ndarray, num_features: int = 21*3, noise_prob: float = 0.3) -> np.ndarray:
    """
    Add random secondary hands to 30% of samples.

    Args:
        X (np.ndarray): Data to add hand landmarks to.
        num_features (int): Number of features in this case is number of hand landmarks. MediaPipe default is 21 landmarks per hand, 3 dimensions per landmark
        noise_prob (float): The probability noise is added to the sequence. Dictates amount of noisiness.
    
    Returns:
        np.ndarray: Noisy data.
    """
    noisy_X = []
    for seq in X:
        if np.random.rand() < noise_prob:
            noise = np.random.normal(0, 0.1, (seq.shape[0], 21*3))
            noisy_seq = np.concatenate([seq, noise], axis=-1)
        else:
            noisy_seq = np.concatenate([seq, np.zeros((seq.shape[0], num_features))], axis=-1)
        noisy_X.append(noisy_seq)
    return np.array(noisy_X)

# assumes running from home dir
metadata_file = 'data/.labels.json'
RNN_PATH = 'src/models/rnn/'

X, y, label_encoder = load_data(metadata_file)

model = create_rnn(
    num_classes=len(label_encoder.classes_)
)
model.summary()
print('Labels:')
for i, label in enumerate(label_encoder.classes_):
    print(f'{i}: {label}')

trained_model, history = train(X, y, model)

with open(RNN_PATH + 'label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

MODEL_PATH = save(trained_model)
print(f'Model saved to {MODEL_PATH}')
