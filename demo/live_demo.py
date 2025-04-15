import os
import json
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import cv2
from collections import deque

class ASL3DCNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(ASL3DCNN, self).__init__()

        self.conv1 = nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(16)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(32)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(64)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(64, num_classes)

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ==== CONFIG ====
MODEL_PATH = 'asl3dcnn_model.pth'
LABELS = ["small emergency", "harsh pain", "help", "help me", "help you", "light pain", "big emergency"]
  # replace with your actual labels
SEQUENCE_LENGTH = 90
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Load model ====
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model = ASL3DCNN(num_classes=7)
model.load_state_dict(state_dict)
model.eval()

# ==== Preprocessing ====
def preprocess_frame(frame, target_size=(224, 224)):
    # Resize to target input size
    frame = cv2.resize(frame, target_size)
    
    # Convert BGR (OpenCV default) to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    frame = frame.astype(np.float32) / 255.0

    # Normalize using ImageNet mean/std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    frame = (frame - mean) / std

    # Convert to tensor shape [C, H, W]
    frame = np.transpose(frame, (2, 0, 1))
    frame = torch.tensor(frame, dtype=torch.float32)

    return frame

# ==== Capture video clip ====
cap = cv2.VideoCapture(0)
clip = deque(maxlen=SEQUENCE_LENGTH)
print("Recording... Perform your ASL sign!")

while len(clip) < SEQUENCE_LENGTH:
    ret, frame = cap.read()
    if not ret:
        break

    resized = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE))
    tensor = preprocess_frame(resized, target_size=(224, 224))

    clip.append(tensor)

    # Show live frame
    display_frame = frame.copy()
    cv2.putText(display_frame, f"Recording: {len(clip)}/{SEQUENCE_LENGTH}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Recording ASL Clip', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close webcam window
cap.release()
cv2.destroyAllWindows()

# ==== Prediction ====
def sliding_window_predict(clip, model, labels, device, window_size=16, stride=8):
    model.eval()
    predictions = []
    
    for start in range(0, len(clip) - window_size + 1, stride):
        window = list(clip)[start:start + window_size]
        tensor = torch.stack(window, dim=1).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
            prob = torch.nn.functional.softmax(output[0], dim=0)
            predictions.append(prob.cpu().numpy())

    avg_probs = np.mean(predictions, axis=0)
    pred_idx = np.argmax(avg_probs)
    confidence = avg_probs[pred_idx]
    return labels[pred_idx], confidence

# Use sliding window for prediction
pred_class, confidence = sliding_window_predict(clip, model, LABELS, DEVICE)
"""try:
    clip_tensor = torch.stack(list(clip), dim=1).unsqueeze(0).to(DEVICE)  # [1, C, T, H, W]
    with torch.no_grad():
        outputs = model(clip_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        print("Probabilities:", probs.cpu().numpy())
        pred_class = LABELS[probs.argmax().item()]
        confidence = probs.max().item()
except Exception as e:
    pred_class = "Error"
    confidence = 0
    print("Prediction failed:", e)"""

# ==== Show result ====
last_frame = frame.copy()
cv2.putText(last_frame, f"Prediction: {pred_class} ({confidence:.2f})", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
cv2.imshow("Result", last_frame)
print(f"Prediction: {pred_class} ({confidence:.2f})")

cv2.waitKey(0)
cv2.destroyAllWindows()
