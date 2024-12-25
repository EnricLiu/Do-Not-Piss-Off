import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Tokenizer, Wav2Vec2Model
from train_transformer import TransformerClassifier
import torch.nn as nn

VALID_EMOTIONS = ["A", "D", "F", "H", "N", "S"]


def extract_features(audio_path, model, tokenizer, device):
    y, _ = librosa.load(audio_path, sr=16000)
    inputs = tokenizer(y, return_tensors="pt", sampling_rate=16000, padding="longest").to(device)
    with torch.no_grad():
        outputs = model(inputs.input_values).last_hidden_state
    feature_vector = outputs.mean(dim=1).cpu().numpy().squeeze()
    return feature_vector

def predict(audio_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    wav2vec_model.eval()

    feature_vector = extract_features(audio_path, wav2vec_model, tokenizer, device)
    feature_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    model = TransformerClassifier(embed_dim=768, num_classes=len(VALID_EMOTIONS)).to(device)
    model.load_state_dict(torch.load("best_model_transformer.pth"))
    model.eval()

    with torch.no_grad():
        outputs = model(feature_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy().squeeze()

    for emotion, prob in zip(VALID_EMOTIONS, probabilities):
        print(f"{emotion}: {prob:.4f}")

if __name__ == "__main__":
    audio_path = "E:\\大学资料\\大三上资料\\公选课\\以Vela挑战嵌入式AloT\\emotion_detect_system\\audio_detect_final\\src\\Train\\呀哈哈_耳聆网_[声音ID：18968].wav"  # Replace with your audio file path
    predict(audio_path)