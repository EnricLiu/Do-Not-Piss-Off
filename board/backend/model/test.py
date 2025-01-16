# from funasr import AutoModel

# # model="iic/emotion2vec_base"
# # model="iic/emotion2vec_base_finetuned"
# # model="iic/emotion2vec_plus_seed"
# # model="iic/emotion2vec_plus_base"
# # model_id = "iic/emotion2vec_base_finetuned"

# model_id = "emotion2vec/emotion2vec_plus_base"
# model = AutoModel(
#     model=model_id,
#     device="cpu",
#     hub="hf",  # "ms" or "modelscope" for China mainland users; "hf" or "huggingface" for other overseas users
# )

# # wav_file = f"{model.model_path}/example/test.wav"
# wav_file = "./res/allall.wav"
# rec_result = model.generate(wav_file, output_dir="./outputs", granularity="utterance", extract_embedding=False)
# print(rec_result)

# rec_result = rec_result[0]
# greatest_score = 0
# for idx, score in enumerate(rec_result["scores"]):
#     if score > greatest_score: 
#         greatest_score = score
#         greatest_idx = idx

# print(f"{rec_result['labels'][greatest_idx]} @ {round(greatest_score*100, 2)}%")

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
["happy", "sad", "angry", "disgust", "fear", "neutral", "surprised"]
emo_map = {
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

inference_pipeline = pipeline(
    task=Tasks.emotion_recognition,
    model="./pretrained/emo2vec/seed",
    device="cpu",
)

wav_file = "./res/allall.wav"
rec_result = inference_pipeline(wav_file, granularity="utterance", extract_embedding=False)
rec_result = rec_result[0]

result = {"happy": 0, "sad": 0, "angry": 0, "disgust": 0, "fear": 0, "neutral": 0, "surprised": 0}
for label, score in zip(rec_result["labels"], rec_result["scores"]):
    result[emo_map[label]] += score
result = sorted(result.items(), key=lambda x: x[1], reverse=True)

print(result)
