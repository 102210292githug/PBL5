import cv2
import numpy as np
import requests
import time

def fetch_and_display_image(url):
    while True:
        response = requests.get(url)
        image_array = np.array(bytearray(response.content), dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if frame is not None:
            cv2.imshow('ESP32-CAM', frame)
            if cv2.waitKey(1) == ord('q'):
                break
        time.sleep(0.1)  # Delay ngắn để giảm tải cho ESP32-CAM

    cv2.destroyAllWindows()

if __name__ == "__main__":
    esp_ip = '192.139.66.101'  # Thay đổi IP phù hợp với môi trường của bạn
    fetch_and_display_image(f'http://{esp_ip}/jpeg')



