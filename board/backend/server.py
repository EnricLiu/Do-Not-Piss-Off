from flask import Flask, request, jsonify

import os
import json
import time
import random
import requests
import threading
from bsp import Board

config = json.load(open("config.json"))
baseURL = config["baseURL"]

app = Flask(__name__,
            static_folder=os.path.join(os.path.dirname(__file__), '../frontend'),
            static_url_path='/')
print(app.static_folder)

g_override = {
    "heart": {
        "is_override": False,
        "mean": None,
        "range": None
    },
    "breath": {
        "is_override": False,
        "mean": None,
        "range": None
    },
    "emotion": {
        "is_override": False,
        "target": None
    },
}

@app.route('/')
def serve_index_html():
    return app.send_static_file('index.html')

@app.route('/data')
def get_data():
    try:
        resp = requests.get(f"{baseURL}/override", timeout=1)
        resp.raise_for_status()
        g_override = resp.json()
    except Exception as e:
        print(e)

    try:
        ret = {}
        real_state = board.get_msg()
        for k, v in real_state.items():
            if g_override[k]["is_override"]:
                if k == "emotion":
                    ret[k] = g_override[k]["target"]
                    continue
                mean = round(float(g_override[k]["mean"]))
                range = round(float(g_override[k]["range"]))
                ret[k] = mean + random.randint(-range, range)
                continue

            ret[k] = v
    except Exception as e:
        print(e)
        ret = board.get_msg()
    print(ret)
    return jsonify(ret)


if __name__ == '__main__':
    from pathlib import Path
    config = json.load(open("config.json"))
    board = Board(config["serialPort"], Path("./tmp"))
    board_task_uart = threading.Thread(target=board.start_uart_task)
    board_task_emo = threading.Thread(target=board.start_emo_task)
    board_task_uart.start()
    board_task_emo.start()
    # app.run(port=45678, debug=True)
    app.run(port=45678)
    
    board_task_uart.join()
    board_task_emo.join()
