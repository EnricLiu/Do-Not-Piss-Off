import json
import time
import random
import asyncio
from pathlib import Path
import threading

from piserial import Serial
from emo2vec import Emo2VecPredictor


class Board:
    VALID_EMOTIONS = ["happy", "sad", "angry", "disgust", "fear", "neutral", "surprised"]

    def __init__(self, serial_port: str, tmp_path: Path, baudrate: int=115200, timeout=1):
        self._ser = Serial(serial_port, baudrate, timeout)
        self._lock = threading.Lock()
        self._fake_params = {
            "heart": {
                "last_rfrs_time": time.time(),
                "rfrs_duration": 2,
                "range": 10,
            },
            "breath": {
                "last_rfrs_time": time.time(),
                "rfrs_duration": 5,
                "range": 5,
            }
        }
        self._last_fake_msg = {
            "heart":   72,
            "breath":  15,
            "emotion": "neutral"
        }
        self._last_msg = {
            "heart":   72,
            "breath":  15,
            "emotion": "neutral"
        }
        
        self.tmp_path = tmp_path

    def start_uart_task(self):
        while True:
            try:
                self._pull_msg()
            except Exception as e:
                # print(e)
                pass
            time.sleep(0.8)

    def _pull_msg(self):
        self._ser.flush()
        with self._lock:
            line = self._ser.readline()
        if line is None or line == b"":
            raise ValueError("Empty line")
        try:
            board_result = json.loads(line)
            heart_rate = board_result.get("heart_rate", None)
            breath_rate = board_result.get("breath_rate", None)
            emotion = board_result.get("emotion", None)

            with self._lock:
                if heart_rate is not None and heart_rate > 0: 
                    self._last_msg["heart"] = heart_rate
                if breath_rate is not None and breath_rate > 0:
                    self._last_msg["breath"] = breath_rate
                if emotion is not None:
                    self._last_msg["emotion"] = emotion

        except Exception as e:
            raise e

    def get_msg(self):
        curr_time = time.time()
        ret = {
            "heart": None,
            "breath": None,
            "emotion": None,
        }
        with self._lock:
            for k, v in self._last_msg.items():
                if v is not None:
                    ret[k] = v
                if k in self._fake_params:
                    if curr_time - self._fake_params[k]["last_rfrs_time"] > self._fake_params[k]["rfrs_duration"]:
                        rand = random.randint(-self._fake_params[k]["range"], self._fake_params[k]["range"])
                        self._fake_params[k]["last_rfrs_time"] = curr_time
                        ret[k] = self._last_msg[k] + rand
                        self._last_fake_msg[k] = ret[k]
                    else:
                        ret[k] = self._last_fake_msg[k]

        return ret
    
    async def start_emo_task(self):
        predictor = Emo2VecPredictor("./model/pretrained/emo2vec/seed")
        while True:
            try:
                vocal = await self.record(2)
                result = predictor.infer(vocal)
                if result is not None and len(result) > 0:
                    emotion, percent = result[0]
                    if percent > 0.5:
                        self._lock.acquire()
                        self._last_msg["emotion"] = emotion
                        self._lock.release()
                
            except Exception as e:
                print(e)
            finally:
                time.sleep(1)

    async def record(self, duration_s: int=5):
        id = round(time.time())
        self.tmp_path.mkdir(parents=True, exist_ok=True)
        if len(list(self.tmp_path.iterdir())) > 16:
            for f in self.tmp_path.iterdir():
                f.unlink()
        
        audio_path = (self.tmp_path / f"{id}.wav").absolute()
        process = await asyncio.create_subprocess_shell(
            f"amixer -c 0 cset name='Capture MIC Path' 'Main Mic'; arecord -D hw:0,0 -d {duration_s} -f cd -t wav {audio_path}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        _, stderr = await process.communicate()

        if process.returncode != 0:
            raise ValueError(f"Error recording audio: {stderr.decode()}")
        
        return audio_path
        

if __name__ == "__main__":
    async def main():
        board = Board()
        config = json.load(open("config.json"))
        board = Board(config["serialPort"])
        res = await board.record()
        print(res)
        
    asyncio.run(main())