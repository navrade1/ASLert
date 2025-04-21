import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import transforms
import cv2

class ASLVideoDataset(Dataset):
    def __init__(self, json_path, video_dir,
                 target_frames, target_size, transform=None, is_train=True, preload=False):
        with open(json_path, 'r') as f:
            self.video_data = json.load(f)['videos']

        self.video_dir = video_dir
        self.target_frames = target_frames
        self.target_size = target_size
        self.label_map = self._create_label_map()
        self.transform = transform
        self.is_train = is_train
        self.preload = preload
        
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        self.cached_videos = {}
        if self.preload:
            for i, video_info in enumerate(self.video_data):
                video_path = os.path.join(self.video_dir, video_info['file_name'])
                self.cached_videos[i] = self._extract_frames(video_path)

    def _create_label_map(self):
        unique_labels = sorted(set(video['label'] for video in self.video_data))
        return {label: idx for idx, label in enumerate(unique_labels)}

    def _extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames - 1, self.target_frames, dtype=int)
        
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                if frames:
                    frames.append(frames[-1].clone())
                else:
                    frames.append(torch.zeros((3, *self.target_size)))
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
            
            if self.transform and self.is_train:
                frame = self.transform(frame)
            else:
                frame = NumpyToTensor(self.mean, self.std)(frame)
                
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            return torch.zeros((self.target_frames, 3, *self.target_size))
        
        video_tensor = torch.stack(frames)
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        
        return video_tensor

    def __len__(self):
        return len(self.video_data)

    def __getitem__(self, idx):
        video_info = self.video_data[idx]
        
        if self.preload and idx in self.cached_videos:
            video_tensor = self.cached_videos[idx]
        else:
            video_path = os.path.join(self.video_dir, video_info['file_name'])
            video_tensor = self._extract_frames(video_path)
            
        label = self.label_map[video_info['label']]

        return video_tensor, label

class NumpyToTensor:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
    
    def __call__(self, x):
        x = x.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        return torch.from_numpy(x).permute(2, 0, 1).float()

class ASL3DCNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(ASL3DCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)
        )

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

def train_model(model, train_loader, val_loader, criterion,
                optimizer, scheduler, device, num_epochs, early_stopping_patience=10):
    
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': []
    }

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
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
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
                
        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), 'best_asl_model.pth')
            
        else:
            early_stopping_counter += 1
            
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break
            
    return history

def optimize_transformed_dataset(dataset, batch_size=4, num_workers=0):
    pin_memory = torch.cuda.is_available()
    
    prefetch_factor = 2 if num_workers > 0 else None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=num_workers > 0
    )
    
    return loader

batch_size = 4
learning_rate = 1e-3
weight_decay = 1e-4
num_epochs = 50
target_frames = 60
target_size = (224, 224)
val_split = 0.2
preload_data = True  # Make this false if data is already preloaded

train_json_path = '../data/.labels.json'
test_json_path = '../test/labels.json'
train_video_dir = '../data'
test_video_dir = '../'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

special_transforms = transforms.Compose([
    NumpyToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
])

train_dataset = ASLVideoDataset(
    json_path = train_json_path,
    video_dir = train_video_dir,
    target_frames = target_frames,
    target_size = target_size,
    transform=special_transforms,
    is_train=True,
    preload=preload_data
)

train_size = int((1 - val_split) * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

import multiprocessing
num_workers = min(4, multiprocessing.cpu_count())

train_loader = optimize_transformed_dataset(
    train_subset,
    batch_size=batch_size,
    num_workers=num_workers
)
val_loader = optimize_transformed_dataset(
    val_subset,
    batch_size=batch_size,
    num_workers=num_workers
)

num_classes = len(train_dataset.label_map)
model = ASL3DCNN(num_classes=num_classes, dropout_rate=0.5).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    verbose=True
)

history = train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs=num_epochs
    early_stopping_patience=10
)

# Uncomment the below line to save the cached videos
# torch.save(train_dataset.cached_videos, 'cached_videos.pt')

test_dataset = ASLVideoDataset(
    json_path=test_json_path,
    video_dir=test_video_dir,
    target_frames=target_frames,
    target_size=target_size,
    transform=None,
    is_train=False,
    preload=preload_data
)

test_loader = optimize_transformed_dataset(
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers
)
model.load_state_dict(torch.load('best_asl_model.pth'))

model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0

all_predictions = []
all_labels = []

with torch.no_grad():
    for videos, labels in test_loader:
        videos, labels = videos.to(device), labels.to(device)

        outputs = model(videos)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
test_loss = test_loss / len(test_loader)
test_accuracy = 100 * test_correct / test_total
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
