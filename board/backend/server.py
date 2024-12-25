from flask import Flask, request, jsonify
from pathlib import Path
import json
import requests
import threading

config = json.load(open("config.json"))
baseURL = config["baseURL"]

app = Flask(__name__,
            static_folder=Path(__file__) / '../../frontend',
            static_url_path='/') 

g_heart_override = {
    "😈": False,
}
g_breath_override = {
    "😈": False,
}
g_emotion_override = {
    "😈": False,
}
print(app.static_folder)
@app.route('/')
def serve_index_html():
    return app.send_static_file('index.html')

@app.route('/data')
def get_data():
    resp = requests.get(f"{baseURL}/override")
    print(resp)


if __name__ == '__main__':
    
    app.run(port=45678, debug=True)