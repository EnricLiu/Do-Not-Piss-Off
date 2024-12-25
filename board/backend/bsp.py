import json
import time
import random
import threading
from piserial import Serial


class Board:
    VALID_EMOTIONS = ["happy", "sad", "angry", "disgust", "fear", "neutral"]

    def __init__(self, serial_port: str, baudrate: int=115200, timeout=1):
        self._ser = Serial(serial_port, baudrate, timeout)
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
        self._latest_msg = {
            "heart":   None,
            "breath":  None,
            "emotion": None
        }

    def start_task(self):
        while True:
            self.pull_msg()
            time.sleep(0.6)

    def pull_msg(self):
        self._ser.flush()
        with self._lock:
            line = self._ser.readline()
        if line is None or line == b"":
            raise ValueError("Empty line")
        try:
            line = line.decode("utf-8")
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
                    self._latest_msg["emotion"] = emotion

        except Exception as e:
            print(e)

    def get_msg(self):
        curr_time = time.time()
        with self._lock:
            ret = {
                "heart":   self._last_msg["heart"],
                "breath":  self._last_msg["breath"],
                "emotion": self._last_msg["emotion"]
            }
            for k, v in self._latest_msg.items():
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