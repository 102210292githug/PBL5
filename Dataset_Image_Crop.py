import os
import cv2
from PIL import Image

def crop_and_resize_faces(input_dir, output_dir, cascade_path, subject_name, img_size=(224, 224)):
    """
    Xác định khuôn mặt trong hình ảnh từ 'input_dir', cắt và thay đổi kích cỡ chúng 
    sang 'img_size', sau đó lưu vào 'output_dir' với tên bắt đầu bằng 'subject_name'.
    """
    # Tạo thư mục đầu ra nếu nó chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Khởi tạo chỉ số để tránh ghi đè dữ liệu
    i = len(os.listdir(output_dir)) + 1
    
    # Tạo cascade để nhận diện khuôn mặt
    faceCascade = cv2.CascadeClassifier(cascade_path)

    # Duyệt qua mỗi hình ảnh trong thư mục đầu vào
    for filename in os.listdir(input_dir):
        # Chỉ xử lý các file hình ảnh
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            imagePath = os.path.join(input_dir, filename)
        else:
            continue

        # Đọc hình ảnh và chuyển đổi sang màu xám
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Nhận diện khuôn mặt
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        # Kiểm tra nếu không có khuôn mặt nào được tìm thấy
        if len(faces) == 0:
            print(f"No faces found in image {filename}.")
            continue

        # Xử lý mỗi khuôn mặt tìm thấy
        for (x, y, w, h) in faces:
            crop_image = Image.open(imagePath).crop((x, y, x+w, y+h)).resize(img_size)
            save_name = os.path.join(output_dir, f"{subject_name}{i}.jpg")
            crop_image.save(save_name)
            i += 1
            print(f"Saved cropped image to {save_name}.")

# Thiết lập đường dẫn
input_directory = "./Webcam Captures"
# output_directory = "./Dataset/validation/102210312"
output_directory = "./Dataset/train/102210312"
haarcascade_path = "./Other Files/haarcascade_frontalface_default.xml"
subject = "102210312"

# Chạy hàm
crop_and_resize_faces(input_directory, output_directory, haarcascade_path, subject)
