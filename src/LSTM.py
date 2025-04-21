import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, BatchNormalization, Dropout, Dense, Normalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from colorama import Fore, Style
from preprocess import load_data
import json  # Added in case your preprocess script or data uses it

# === Load Processed Landmark Data ===
metadata_file = 'data/.labels.json'
X, y, label_encoder = load_data(metadata_file)

# === Train/Test Split ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === Model Parameters ===
input_shape = X_train.shape[1:]  # (timesteps, features)
num_classes = len(label_encoder.classes_)
dropout_rate = 0.3

# === Build LSTM Model ===
inputs = Input(shape=input_shape)

# Input normalization
x = Normalization()(inputs)
print(f"{Fore.CYAN} → Input normalization applied{Style.RESET_ALL}")

# First LSTM layer
x = LSTM(256, return_sequences=True, activation='tanh',
         kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = Dropout(dropout_rate)(x)

# Second LSTM layer
x = LSTM(128, return_sequences=False, activation='tanh',
         kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = Dropout(dropout_rate)(x)

# Output layer
outputs = Dense(num_classes, activation='softmax')(x)

# Compile the model
model = Model(inputs, outputs)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# === Train the Model ===
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=32
)

# Evaluate on validation set instead of test set (since no separate test set is defined)
print(f"\n{Fore.MAGENTA}{Style.BRIGHT}MODEL EVALUATION (on Validation Set){Style.RESET_ALL}")
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

# === Save the Model (optional) ===
model.save("src/models/lstm/asl_lstm_model.keras")
model.summary()