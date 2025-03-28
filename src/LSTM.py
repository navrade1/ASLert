import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Generate dummy sequential data
X = np.random.rand(1000, 10, 1)  # 1000 samples, 10 time steps, 1 feature
y = np.random.rand(1000, 1)  # 1000 target values

# Define LSTM model
model = Sequential([
    LSTM(50, activation='tanh', return_sequences=False, input_shape=(10, 1)),
    Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X, y, epochs=10, batch_size=32)

# Make predictions
test_sample = np.random.rand(1, 10, 1)  # Single test sequence
prediction = model.predict(test_sample)
print("Prediction:", prediction)