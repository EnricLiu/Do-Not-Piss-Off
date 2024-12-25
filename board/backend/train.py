import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pickle
import torch.nn.functional as F

AUDIO_DIR = "E:\\大学资料\\大三上资料\\公选课\\以Vela挑战嵌入式AloT\\emotion_detect_system\\audio_detect_final\\dataset\\CREMA-D\\AudioWAV"
LABEL_CSV = "E:\\大学资料\\大三上资料\\公选课\\以Vela挑战嵌入式AloT\\emotion_detect_system\\audio_detect_final\\dataset\\CREMA-D\\processedResults\\tabulatedVotes.csv"
VALID_EMOTIONS = ["A", "D", "F", "H", "N", "S"]
MAX_LEN = 128  # 设定一个固定的长度

def load_data():
    if os.path.exists("data.pkl"):
        print("Loading data from data.pkl")
        with open("data.pkl", "rb") as f:
            audio_paths, labels = pickle.load(f)
    else:
        print("Loading data from CSV")
        df = pd.read_csv(LABEL_CSV)
        df.columns = df.columns.str.strip()
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        audio_paths, labels = [], []
        for _, row in df.iterrows():
            file_name = row["fileName"]  
            emotion = row["emoVote"]
            if ":" in emotion:
                emotion = random.choice(emotion.split(":"))
            if emotion in VALID_EMOTIONS:
                audio_paths.append(os.path.join(AUDIO_DIR, file_name+".wav"))
                labels.append(emotion)
        with open("data.pkl", "wb") as f:
            pickle.dump((audio_paths, labels), f)
    return audio_paths, labels

def extract_features(audio_paths):
    features = []
    for path in tqdm(audio_paths):
        y, sr = librosa.load(path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        if mfcc.shape[1] < MAX_LEN:
            mfcc = np.pad(mfcc, ((0, 0), (0, MAX_LEN - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :MAX_LEN]
        features.append(np.mean(mfcc, axis=1))
    return np.array(features)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
            if stride != 1 or in_channels != out_channels else None
        )

    def forward(self, x):
        shortcut = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            shortcut = self.downsample(shortcut)
        out += shortcut
        return F.relu(out)

class EmotionResNet(nn.Module):
    def __init__(self, num_classes=6):
        super(EmotionResNet, self).__init__()
        self.layer1 = ResidualBlock(1, 64, stride=2)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        self.layer4 = ResidualBlock(256, 512, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)

def main():
    audio_paths, labels = load_data()
    X = extract_features(audio_paths)
    print(f"X has shape of {X.shape}")
    label2idx = {lbl: i for i, lbl in enumerate(VALID_EMOTIONS)}
    y = np.array([label2idx[lbl] for lbl in labels])

    # Split data
    split_idx = int(len(X)*0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).unsqueeze(2)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).unsqueeze(2)
    y_val = torch.tensor(y_val, dtype=torch.long)

    model = EmotionResNet(len(VALID_EMOTIONS))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_acc = 0.7371
    best_model_path = "best_model.pth"

    # Training loop
    patience = 10
    epochs_no_improve = 0
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_loss = criterion(val_out, y_val)
            val_acc = calculate_accuracy(val_out, y_val)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve += 1
            print(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc:.4f}")
            if epochs_no_improve == patience:
                print("Early stopping")
                break
        else:
            epochs_no_improve = 0
        print(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc:.4f}")

if __name__ == "__main__":
    main()