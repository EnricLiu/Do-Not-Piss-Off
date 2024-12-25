from flask import Flask, request, jsonify
from pathlib import Path
import json
import random
import requests
import threading
from bsp import Board

config = json.load(open("config.json"))
baseURL = config["baseURL"]

app = Flask(__name__,
            static_folder=Path(__file__) / '../../frontend',
            static_url_path='/') 

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

                ret[k] = g_override[k]["mean"] + random.randint(-g_override[k]["range"], g_override[k]["range"])
                continue

            ret[k] = v
    except Exception as e:
        print(e)
        ret = board.get_msg()
    
    return jsonify(ret)


if __name__ == '__main__':
    board = Board(config["serialPort"])
    board_task = threading.Thread(target=board.start_task)
    board_task.start()
    app.run(port=45678, debug=True)
    board_task.join()