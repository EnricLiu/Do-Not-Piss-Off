import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time

import librosa
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from transformers import Wav2Vec2Tokenizer, Wav2Vec2Model
from train_transformer import TransformerClassifier


class EmotionClassifier:
    VALID_EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful"]

    def __init__(self, model_path: str, wav2vec_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.wav2vec_path = wav2vec_path

        self._tokenizer: Wav2Vec2Tokenizer = None
        self._wav2vec_model: Wav2Vec2Model = None
        self._model: TransformerClassifier = None

        self.loaded = False

    def predict(self, audio: np.ndarray|Path):
        if not self.loaded: self.load_model()
        features = self._extract_features(audio)
        with torch.no_grad():
            outputs = self._model(features)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy().squeeze()

        return dict(zip(EmotionClassifier.VALID_EMOTIONS, probabilities))


    def _extract_features(self, audio: np.ndarray|Path) -> torch.Tensor:
        if isinstance(audio, Path): audio, _ = librosa.load(audio, sr=16000)
        inputs = self._tokenizer(audio, return_tensors="pt", sampling_rate=16000, padding="longest").to(self.device)
        with torch.no_grad():
            outputs = self._wav2vec_model(inputs.input_values).last_hidden_state
        feature_vector = outputs.mean(dim=1).squeeze()
        return feature_vector

    def load_model(self):
        try:
            self._tokenizer = Wav2Vec2Tokenizer.from_pretrained(self.wav2vec_path)
            self._wav2vec_model = Wav2Vec2Model.from_pretrained(self.wav2vec_path).to(self.device).eval()
            self._model = TransformerClassifier(embed_dim=768, num_classes=len(EmotionClassifier.VALID_EMOTIONS)).to(self.device).eval()
            self._model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        except Exception as e:
            raise Exception(f"EmotionClassifier: Failed loading model. {e}")
        
        self.loaded = True


if __name__ == "__main__":
    # from record_audio import record_audio
    # audio = record_audio()
    predicter = EmotionClassifier("./ckpt/best_transformer.pth", "./pretrained/wav2vec2-base-960h")
    audio = Path("./res/test.wav")

    start = time.perf_counter()
    predicter.load_model()
    load_fin = time.perf_counter()
    print(f"Load time: {(load_fin-start):.6f} seconds")
    print(predicter.predict(audio))
    pred_fin = time.perf_counter()
    print(f"Prediction time: {(pred_fin-load_fin):.6f} seconds")
    
