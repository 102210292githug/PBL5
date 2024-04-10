from keras.preprocessing import image
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

# Load the model
model = load_model("./Other Files/finetuned_vgg_face_model.h5")
labels = np.load('./Other Files/class_labels.npy', allow_pickle=True).item()
labels = {v: k for k, v in labels.items()}  # Reverse the dictionary
print(labels)

# Định nghĩa ngưỡng xác suất
CONFIDENCE_THRESHOLD = 0.8

def cam_test():
    # Khởi tạo Haar Cascade để phát hiện khuôn mặt
    face_cascade = cv2.CascadeClassifier('./Other Files/haarcascade_frontalface_default.xml')

    # Khởi tạo camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            # Khoanh khung mặt
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Cắt và xử lý khuôn mặt
            face = frame[y:y+h, x:x+w]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)
            face /= 255.0  # Đảm bảo sử dụng cùng một chuẩn hóa với khi huấn luyện

            # Dự đoán
            prediction = model.predict(face)
            class_idx = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction)
            
            # Kiểm tra ngưỡng xác suất
            if confidence < CONFIDENCE_THRESHOLD:
                label = "Unknown"
                # Thêm thông tin về người giống nhất
                similar_label = labels[class_idx]
                cv2.putText(frame, f'{label}, but looks like {similar_label}: {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else:
                label = labels[class_idx]
                # Hiển thị label và xác suất
                cv2.putText(frame, f'{label}: {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

cam_test()
