import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

class Emo2VecPredictor:
    EMO_MAP = {
        '生气/angry':       "angry",
        '厌恶/disgusted':   "disgust",
        '恐惧/fearful':     "fear",
        '开心/happy':       "happy",
        '中立/neutral':     "neutral",
        '其他/other':       "neutral",
        '难过/sad':         "sad",
        '吃惊/surprised':   "surprised",
        '<unk>':            "neutral",
    }
    
    def __init__(self, model_dir:str, device='cpu'):
        self._model = pipeline(
            task=Tasks.emotion_recognition,
            model=model_dir,
            device=device,
            disable_update=True
        )
        
    def infer(self, wav_path: Path):
        rec_result = self._model(str(wav_path), granularity="utterance", extract_embedding=False)[0]

        result = {"happy": 0, "sad": 0, "angry": 0, "disgust": 0, "fear": 0, "neutral": 0, "surprised": 0}
        for label, score in zip(rec_result["labels"], rec_result["scores"]):
            result[Emo2VecPredictor.EMO_MAP[label]] += score
        result = sorted(result.items(), key=lambda x: x[1], reverse=True)
                
        return result

if __name__ == "__main__":
    predictor = Emo2VecPredictor("./pretrained/emo2vec/seed")
    
    audio_path = Path("./res/test.wav")
    result = predictor.infer(audio_path)
    print(f"{result[0][0]} @ {result[0][1]*100}%")