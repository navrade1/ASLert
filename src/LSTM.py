import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Generate dummy sequential data
X = np.random.rand(1000, 10, 1)  # 1000 samples, 10 time steps, 1 feature
y = np.random.rand(1000, 1)      # 1000 target values

# Define LSTM model
model = Sequential([
    LSTM(50, activation='tanh', input_shape=(10, 1)),
    Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X, y, epochs=10, batch_size=32, verbose=1)

# Make a prediction with a test sample
test_sample = np.random.rand(1, 10, 1)
prediction = model.predict(test_sample)

print("Prediction:", prediction)
