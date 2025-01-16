import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import pickle
import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
from transformers import Wav2Vec2Tokenizer, Wav2Vec2Model
from torch.utils.data import DataLoader, TensorDataset

AUDIO_DIR = "E:\\大学资料\\大三上资料\\公选课\\以Vela挑战嵌入式AloT\\emotion_detect_system\\audio_detect_final\\dataset\\CREMA-D\\AudioWAV"
LABEL_CSV = "E:\\大学资料\\大三上资料\\公选课\\以Vela挑战嵌入式AloT\\emotion_detect_system\\audio_detect_final\\dataset\\CREMA-D\\processedResults\\tabulatedVotes.csv"
VALID_EMOTIONS = ["A", "D", "F", "H", "N", "S"]

def load_data():
    if os.path.exists("data_openl3.pkl"):
        with open("data_openl3.pkl", "rb") as f:
            audio_paths, labels = pickle.load(f)
    else:
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
                audio_paths.append(os.path.join(AUDIO_DIR, file_name + ".wav"))
                labels.append(emotion)
        with open("data_openl3.pkl", "wb") as f:
            pickle.dump((audio_paths, labels), f)
    return audio_paths, labels

def extract_features(audio_paths):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    model.eval()

    extracted_features = []
    for path in tqdm(audio_paths):
        y, _ = librosa.load(path, sr=16000)
        inputs = tokenizer(y, return_tensors="pt", sampling_rate=16000, padding="longest").to(device)
        with torch.no_grad():
            outputs = model(inputs.input_values).last_hidden_state
        feature_vector = outputs.mean(dim=1).cpu().numpy().squeeze()
        extracted_features.append(feature_vector)
    return np.array(extracted_features)

class TransformerClassifier(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_layers=5, num_classes=6):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = x + self.pos_embedding
        x = self.transformer(x)
        return self.fc(x.mean(dim=0))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio_paths, labels = load_data()
    X = extract_features(audio_paths)
    label2idx = {lbl: i for i, lbl in enumerate(VALID_EMOTIONS)}
    y = np.array([label2idx[lbl] for lbl in labels])

    split_idx = int(len(X)*0.8)
    X_train_np, X_val_np = X[:split_idx], X[split_idx:]
    y_train_np, y_val_np = y[:split_idx], y[split_idx:]

    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_np, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_np, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val_np, dtype=torch.long)

    # Add batch_size and create DataLoader for training
    BATCH_SIZE = 16
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    X_val_tensor = X_val_tensor.unsqueeze(0).to(device)
    y_val_tensor = y_val_tensor.to(device)

    model = TransformerClassifier(embed_dim=768, num_classes=len(VALID_EMOTIONS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    best_val_acc = 0.7481
    for epoch in tqdm(range(500)):
        model.train()
        total_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.unsqueeze(0).to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_out = model(X_val_tensor)
            val_loss = criterion(val_out, y_val_tensor)
            val_preds = torch.argmax(val_out, dim=1)
            val_acc = (val_preds == y_val_tensor).sum().item() / len(y_val_tensor)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model_openl3.pth")

        print(f"Epoch {epoch+1}, Train Loss: {total_train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc:.4f}")

if __name__ == "__main__":
    main()