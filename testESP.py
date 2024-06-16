from flask import Flask, request
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import logging
from threading import Thread
import threading
import time
import requests
import sys

# Load the model and labels
model = load_model("./Other Files/finetuned_vgg_face_model.h5")
labels = np.load('./Other Files/class_labels.npy', allow_pickle=True).item()
labels = {v: k for k, v in labels.items()}  # Reverse the dictionary
face_cascade = cv2.CascadeClassifier('./Other Files/haarcascade_frontalface_default.xml')
CONFIDENCE_THRESHOLD = 0.75

app = Flask(__name__)
latest_image_lock = threading.Lock()
latest_image = None
should_exit = threading.Event()  # Event to signal exit

# Suppress normal request logs
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)



last_label = "NoFace"
from datetime import datetime
import os

@app.route('/image', methods=['POST'])
def handle_image():
    global latest_image
    global last_label
    label = "No Face"
    with latest_image_lock:
        latest_image = request.data
        nparr = np.frombuffer(latest_image, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = frame[y:y+h, x:x+w]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)
            face /= 255.0

            # Predict using the VGG-Face model
            prediction = model.predict(face)
            class_idx = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction)
            
            if confidence >= CONFIDENCE_THRESHOLD:
                label = labels[class_idx]
            elif CONFIDENCE_THRESHOLD > 0.1:
                label = "Unknown"
    print(label) 
    value = 200
    if label == "Unknown":
        value += 1
    if label == "No Face":
        value += 2

    if label != "No Face" and last_label != label:
        urlApp = "http://192.168.133.124:1713/customers/haveFace"
        urlEsp = "http://192.168.133.211/" + str(value)
        files = {'image': ('filename.jpg', latest_image, 'image/jpeg')}
        data = {'name': label}
        # response =
        requests.post(urlApp, files=files, data=data)
        requests.get(urlEsp)
        # print(response)
    last_label = label
    
    

    return str(label), value






@app.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_func = request.environ.get('werkzeug.server.shutdown')
    if shutdown_func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    shutdown_func()
    return 'Server shutting down...'

def start_flask_app():
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)



if __name__ == '__main__':
    # Start Flask in a separate thread
    flask_thread = Thread(target=start_flask_app)
    flask_thread.start()

    try:
        while not should_exit.is_set():
            if cv2.waitKey(500) & 0xFF == ord('q'):
                print("Exiting the program...")
                requests.post('http://localhost:5000/shutdown')  # Gửi yêu cầu để dừng Flask
                should_exit.set()  # Signal to exit the program
            time.sleep(0.5)

    finally:
        flask_thread.join()
        cv2.destroyAllWindows()  # Make sure to destroy all the cv2 windows
