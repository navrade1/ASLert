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

class ASLVideoDataset(Dataset):
    def __init__(self, json_path, video_dir,
                 target_frames=90, target_size=(112, 112)):
        with open(json_path, 'r') as f:
            self.video_data = json.load(f)['videos']

        self.video_dir = video_dir
        self.target_frames = target_frames
        self.target_size = target_size
        self.label_map = self._create_label_map()

    def _create_label_map(self):
        unique_labels = sorted(set(video['label'] for video in self.video_data))
        return {label: idx for idx, label in enumerate(unique_labels)}

    def _load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        while len(frames) < self.target_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, self.target_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame = frame.astype(np.float32) / 255.0
            frame = (frame - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            frame = transforms.ToTensor()(frame).type(torch.float32)

            frames.append(frame)

        cap.release()

        if len(frames) < self.target_frames:
            padding = [torch.zeros_like(frames[-1])] * (self.target_frames - len(frames))
            frames.extend(padding)

        else:
            frames = frames[:self.target_frames]

        video_tensor = torch.stack(frames)
        video_tensor = video_tensor.permute(1, 0, 2, 3)

        return video_tensor

    def __len__(self):
        return len(self.video_data)

    def __getitem__(self, idx):
        video_info = self.video_data[idx]
        video_path = os.path.join(self.video_dir, video_info['file_name'])

        video_tensor = self._load_video(video_path)
        label = self.label_map[video_info['label']]

        return video_tensor, label

class ASL3DCNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(ASL3DCNN, self).__init__()

        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.fc5 = nn.Linear(256 * 12 * 7 * 7, 1024)
        self.fc6 = nn.Linear(1024, num_classes)

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3(x))
        x = self.pool3(x)

        x = self.relu(self.conv4(x))
        x = self.pool4(x)

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc5(x))
        x = self.dropout(x)

        x = self.relu(self.fc6(x))
        x = self.dropout(x)

        return x

def train_model(train_loader, val_loader, model, criterion,
                optimizer, device, num_epochs=50):

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for videos, labels in train_loader:
            videos, labels = videos.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for videos, labels in val_loader:
                videos, labels = videos.to(device), labels.to(device)
                outputs = model(videos)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Train Accuracy: {train_accuracy:.2f}%')

def test_model(test_loader, model, criterion, device):
    """
    Separate testing method
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Detailed predictions list for further analysis
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for videos, labels in test_loader:
            videos, labels = videos.to(device), labels.to(device)

            outputs = model(videos)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

            # Store predictions and true labels
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    test_loss = total_loss / len(test_loader)
    test_accuracy = 100 * total_correct / total_samples

    print("\n--- Test Results ---")
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.2f}%')

    # Optional: Detailed classification report
    from sklearn.metrics import classification_report
    print("\nDetailed Classification Report:")
    print(classification_report(all_true_labels, all_predictions))

    return {
        'loss': test_loss,
        'accuracy': test_accuracy,
        'predictions': all_predictions,
        'true_labels': all_true_labels
    }

batch_size = 5
learning_rate = 0.0001
num_epochs = 3
target_frames = 60
target_size = (224, 224)

train_json_path = '../data/.labels.json'
test_json_path = '../test/labels.json'
train_video_dir = '../'
test_video_dir = '../'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = ASLVideoDataset(
    json_path = train_json_path,
    video_dir = train_video_dir,
    target_frames = target_frames,
    target_size = target_size
)

test_dataset = ASLVideoDataset(
    json_path = test_json_path,
    video_dir = test_video_dir,
    target_frames = target_frames,
    target_size = target_size
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_classes = len(train_dataset.label_map)
model = ASL3DCNN(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_model(train_loader, test_loader, model, criterion, optimizer, device, num_epochs)

test_results = test_model(test_loader, model, criterion, device)
print(test_results)