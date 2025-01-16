import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from modelscope.pipelines import pipeline, Pipeline
from modelscope.utils.constant import Tasks

VALID_EMOTIONS = ["A", "D", "F", "H", "N", "S"]

def record_audio(tmp_path: Path, device=0, duration=2, fs=48000):
    from scipy.io.wavfile import write
    tmp_path.mkdir(parents=True, exist_ok=True)
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16', device=device)
    sd.wait()
    p = tmp_path / f"{int(time.time())}.wav"
    write(p, fs, audio.astype('int16'))
    return p

class Emo2VecPredictor:
    def __init__(self, model_dir:str, device='cpu'):
        self._model = pipeline(
            task=Tasks.emotion_recognition,
            model=model_dir,
            device="cpu",
        )
        
    def infer(self, wav_path: Path):
        rec_result = self._model(str(wav_path), granularity="utterance", extract_embedding=False)

        rec_result = rec_result[0]
        greatest_score = 0
        for idx, score in enumerate(rec_result["scores"]):
            if score > greatest_score: 
                greatest_score = score
                greatest_idx = idx
                
        return rec_result['labels'][greatest_idx], greatest_score

if __name__ == "__main__":
    predictor = Emo2VecPredictor("./pretrained/emo2vec/seed")
    
    devices = sd.query_devices()
    print(devices)
    device_id = int(input("device id: "))
    
    while True:
        audio_path = record_audio(Path("./tmp"), device = device_id)
        label, acc = predictor.infer(audio_path)
        print(f"{label} @ {round(acc*100, 2)}%")
        # time.sleep(10)