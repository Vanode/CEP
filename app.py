# app.py
import base64
import cv2
import numpy as np
import pandas as pd
import copy
import itertools
import string
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from tensorflow import keras
import mediapipe as mp

# Flask app
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Load model
model = keras.models.load_model("model.h5")

# Mediapipe hands
mp_hands = mp.solutions.hands
alphabet = ['1','2','3','4','5','6','7','8','9'] + list(string.ascii_uppercase)

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, lm in enumerate(landmarks.landmark):
        x = min(int(lm.x * image_width), image_width - 1)
        y = min(int(lm.y * image_height), image_height - 1)
        landmark_point.append([x, y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp = copy.deepcopy(landmark_list)
    base_x, base_y = temp[0][0], temp[0][1]
    for i, point in enumerate(temp):
        temp[i][0] -= base_x
        temp[i][1] -= base_y
    flat = list(itertools.chain.from_iterable(temp))
    max_val = max(list(map(abs, flat))) if flat else 1
    flat = [v / max_val for v in flat]
    return flat

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("frame")
def handle_frame(data):
    try:
        img_data = base64.b64decode(data["image"])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:
            results = hands.process(rgb)
            if not results.multi_hand_landmarks:
                emit("prediction", {"letter": ""})
                return
            hand_landmarks = results.multi_hand_landmarks[0]
            lm_list = calc_landmark_list(frame, hand_landmarks)
            processed = pre_process_landmark(lm_list)
            arr = np.array(processed).reshape(1, -1)
            preds = model.predict(arr, verbose=0)
            pred_class = np.argmax(preds, axis=1)[0]
            letter = alphabet[pred_class]
            emit("prediction", {"letter": letter})
    except Exception as e:
        print("Error:", e)
        emit("prediction", {"letter": ""})

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)
