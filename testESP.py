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
CONFIDENCE_THRESHOLD = 0.8

app = Flask(__name__)
latest_image_lock = threading.Lock()
latest_image = None
should_exit = threading.Event()  # Event to signal exit

# Suppress normal request logs
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


@app.route('/command', methods=['POST'])
def handle_command():
    data = request.get_json()  # Lấy dữ liệu JSON từ request
    command = data.get('command')
    
    if command == 'unlock':
        # Thực hiện lệnh mở khóa
        print("Unlock command received.")
        # Thêm code để mở khóa ở đây
        return "Unlocking", 200
    elif command == 'lock': 
        # Thực hiện lệnh khóa cửa
        print("Lock command received.")
        # Thêm code để khóa cửa ở đây
        return "Locking", 200
    else:
        return "Command not recognized", 400

last_label = "NoFace"
@app.route('/image', methods=['POST'])
def handle_image():
    global latest_image
    global last_label
    tmp_image = latest_image
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
    
    # Gửi ảnh và nhãn đến một server khác trước khi t rả về response
    # url = "http://192.139.66.104:1713/customers"
    # files = {'image': ('filename.jpg', latest_image, 'image/jpeg')}
    # data = {'name': label}
    # response = requests.post(url, files=files, data=data)
    
    print(label) 
    if label != "No Face" and last_label != label:
        url = "http://192.139.66.104:1713/customers/haveFace"
        files = {'image': ('filename.jpg', latest_image, 'image/jpeg')}
        data = {'name': label}
        response = requests.post(url, files=files, data=data)
        print(response)
    last_label = label
    
    
    # print(label) 
    latest_image = tmp_image
    return str(label), 200 + int(label != "Unknown" and label != "No Face")



def display_video():
    cv2.namedWindow("Live Video Stream")
    
    while not should_exit.is_set():
        with latest_image_lock:
            if latest_image is not None:
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

                    if confidence < CONFIDENCE_THRESHOLD:
                        label = "Unknown"
                        similar_label = labels[class_idx]
                        cv2.putText(frame, f'{label}, but looks like {similar_label}: {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    else:
                        label = labels[class_idx]
                        cv2.putText(frame, f'{label}: {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                cv2.imshow('Live Video Stream', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    should_exit.set()
                    break
        time.sleep(0.01)
    cv2.destroyAllWindows()
    sys.exit()

@app.route('/shutdown', methods=['GET'])
def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    return 'Server shutting down...'

def start_flask_app():
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)

def shutdown_server():
    requests.get('http://127.0.0.1:5000/shutdown')  # Trigger Flask server shutdown

if __name__ == '__main__':
    # Start Flask in a separate thread
    flask_thread = Thread(target=start_flask_app)
    flask_thread.start()

    # Start the display in a separate thread
    # video_thread = Thread(target=display_video)
    # video_thread.start()


    try:
        while not should_exit.is_set():
            time.sleep(0.5)  # Reduce CPU usage

        shutdown_server()  # Shutdown the Flask server
    finally:
        flask_thread.join()
        # video_thread.join()
         