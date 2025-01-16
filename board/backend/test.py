import json
import asyncio
from pathlib import Path

from emo2vec import Emo2VecPredictor
from bsp import Board


if __name__ == "__main__":
    async def voice_infer_task():
        config = json.load(open("config.json"))
        board = Board(config["serialPort"], Path("./tmp"))
        
        predictor = Emo2VecPredictor("./model/pretrained/emo2vec/seed")
        while True:
            try:
                audio_path = await board.record(2)
                res = predictor.infer(audio_path)
                print(res)
            except Exception as err:
                print(err)
        
    asyncio.run(voice_infer_task())