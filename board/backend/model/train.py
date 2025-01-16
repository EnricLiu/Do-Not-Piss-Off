import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import random
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

import librosa
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split, DataLoader


class EmoVocalDataset(Dataset):
    MAX_LEN = 80
    VALID_EMOTIONS = ["A", "D", "F", "H", "N", "S"]
    
    def __init__(self, attr_csv: pd.DataFrame, vocal_path: Path, npz_path: Path):
        attr_csv.columns = attr_csv.columns.str.strip()
        attr_csv = attr_csv.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        
        if npz_path is not None and npz_path.is_file():
            with np.load(npz_path) as data:
                self.audios = data["audios"]
                self.labels = data["labels"]
                self.len = len(self.audios)
                print(self.audios.shape)
                print(self.labels.shape)
                return
        
        vocal_paths, labels = [], []
        label2idx = {lbl: i for i, lbl in enumerate(EmoVocalDataset.VALID_EMOTIONS)}
        for _, row in attr_csv.iterrows():
            file_name = row["fileName"]
            emotion = row["emoVote"]
            if ":" in emotion:
                emotion = random.choice(emotion.split(":"))
            if emotion in EmoVocalDataset.VALID_EMOTIONS:
                vocal_paths.append(vocal_path / f"{file_name}.wav")
                label = np.zeros(len(EmoVocalDataset.VALID_EMOTIONS))
                label[label2idx[emotion]] = 1
                labels.append(label)
        
        with tqdm(total=len(vocal_paths)) as pbar:
            with ThreadPoolExecutor(os.cpu_count()//6) as executor:
                def worker(tp):
                    id, _path = tp
                    res = EmoVocalDataset.extract_features(_path)
                    pbar.update(1)
                    return id, res
                res = executor.map(worker, enumerate(vocal_paths))
        
        res = sorted(res, key=lambda x: x[0])
        res = [x[1] for x in res]
        
        self.audios = np.asarray(res)
        self.labels = np.asarray(labels)
        self.len = len(self.audios)
        
        if npz_path is not None:
            np.savez(npz_path, audios=self.audios, labels=self.labels)

    def __len__(self):
        return self.len
        
    def __getitem__(self, index):
        if index >= self.len: return None
        audio, label = self.audios[index], self.labels[index]
        return {"vocal": audio, "label": label}
        
    def extract_features(_path):
        y, sr = librosa.load(_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        if mfcc.shape[1] < EmoVocalDataset.MAX_LEN:
            mfcc = np.pad(mfcc, ((0, 0), (0, EmoVocalDataset.MAX_LEN - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :EmoVocalDataset.MAX_LEN]
        return mfcc[np.newaxis, :]
        

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
        self.in_channels = 1
        self.layer1 = self._make_layer( 64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        # self.softmax = nn.Softmax(dim=1)

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = self.softmax(x)
        return x

class LegacyEmotionResNet(nn.Module):
    def __init__(self, num_classes=6):
        super(LegacyEmotionResNet, self).__init__()
        self.layer1 = ResidualBlock(1, 64, stride=2)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        self.layer4 = ResidualBlock(256, 512, stride=2)
        self.fc = nn.Linear(2560, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.softmax(x)

def train(train_id, device, ckpt_path: Path, use_amp=False,
          save_interval=None, epochs=1000, set_split_seed=0, set_ratio=0.8,
          bs=32, lr=1e-4, wd=1e-5):
    
    attr_csv = pd.read_csv("./dataset/tabulatedVotes.csv")
    vocal_path = Path("./dataset/AudioWAV")
    npz_path = Path("./dataset/features.npz")
    
    dataset = EmoVocalDataset(attr_csv, vocal_path, npz_path)
    
    train_set, val_set = \
        random_split(dataset, [int(len(dataset)*set_ratio), len(dataset)-int(len(dataset)*set_ratio)],
                     generator=torch.Generator().manual_seed(set_split_seed))
    
    loader_args  = dict(batch_size=bs, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set,  shuffle=True, **loader_args)
    val_loader   = DataLoader(  val_set, shuffle=False, **loader_args)
    
    model = LegacyEmotionResNet(len(EmoVocalDataset.VALID_EMOTIONS)).to(device)
    scaler = amp.GradScaler(enabled=use_amp)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    cplt_epoches = 0
    
    def save_ckpt(val_acc, epoch):
        if ckpt_path is None: return
        torch.save(model, os.path.join(ckpt_path, f'acc={val_acc:.3f}-e{epoch}-{train_id}.pth'))
        print(f'Checkpoint {epoch} saved!')
    

    # Training loop
    try:
        best_acc = 0
        if ckpt_path is not None: ckpt_path.mkdir(parents=True, exist_ok=True)
        for epoch in range(1, epochs+1):
            
            model.train()
            epoch_loss = 0.0
            
            with tqdm(total=len(train_set), desc=f'Epoch {epoch}/{epochs}') as pbar:
                for batch in train_loader:
                    inputs = batch["vocal"].to(device)
                    labels = batch["label"].to(device)

                    optimizer.zero_grad()
                    with amp.autocast(enabled=use_amp):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels.float())

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    pbar.update(bs)
                    epoch_loss += loss.item()
                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                epoch_loss = epoch_loss / len(train_loader)

            # train acc
            model.eval()
            train_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in train_loader:
                    inputs = batch["vocal"].to(device)
                    labels = batch["label"].to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels.float())
                    train_loss += loss.item()

                    total += labels.size(0)
                    predicted = torch.argmax(outputs, dim=1)
                    labels = torch.argmax(labels, dim=1)
                    correct += (predicted == labels).sum().item()

            train_epoch_loss = train_loss / len(train_loader)
            train_accuracy = correct / total
            print(f"[Train] Loss: {train_epoch_loss:.4f}, Acc: {train_accuracy:.4f}", end="\t")

            # 验证
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch["vocal"].to(device)
                    labels = batch["label"].to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels.float())
                    val_loss += loss.item()

                    total += labels.size(0)
                    predicted = torch.argmax(outputs, dim=1)
                    labels = torch.argmax(labels, dim=1)
                    correct += (predicted == labels).sum().item()

            val_epoch_loss = val_loss / len(val_loader)
            val_accuracy = correct / total
            print(f"[Validation] Loss: {val_epoch_loss:.4f}, Acc: {val_accuracy:.4f}", end="\t")
            # print(temp)

            if val_accuracy > best_acc:
                best_acc = val_accuracy
                save_ckpt(val_accuracy, epoch)
            elif save_interval is not None and epoch % save_interval == 0:
                best_acc = max(best_acc, val_accuracy)
                save_ckpt(val_accuracy, epoch)
            else:
                print()

            train_accs.append(train_accuracy)
            train_losses.append(train_epoch_loss)
            val_accs.append(val_accuracy)
            val_losses.append(val_epoch_loss)
            cplt_epoches = epoch

    except KeyboardInterrupt:
        print("Training interrupted by user.")
        if ckpt_path:
            print("Saving last checkpoint...")
            save_ckpt(val_accuracy, epoch)
    except Exception as e:
        print(e)
        raise e
    finally:
        return train_losses, train_accs, val_losses, val_accs, cplt_epoches


if __name__ == "__main__":
    train_id = round(time.time())
    
    train_loss, train_acc, val_loss, val_acc, epoches \
       = train(train_id, device="cuda", ckpt_path=Path("./ckpt"))
    
    import matplotlib.pyplot as plot
    plot_save_path = Path(f"./results/{train_id}.png")
    plot_save_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot.figure(figsize=(24, 10))
    fig.suptitle(f"id={train_id}")
    ax = plot.subplot(1, 2, 1)
    ax.plot(train_loss, label="train loss")
    ax.plot(val_loss, label="val loss")
    ax.set_title(f"Loss over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax = plot.subplot(1, 2, 2)
    ax.plot(train_acc, label="train acc")
    ax.plot(val_acc, label="val acc")
    ax.set_title(f"Accuracy over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    plot.savefig(plot_save_path,
                 bbox_inches='tight')
    plot.show()