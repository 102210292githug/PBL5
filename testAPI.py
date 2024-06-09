import requests

# Địa chỉ của endpoint
url = 'http://10.10.59.36:1713/upload'

# Tên bạn muốn gửi đi
name = '102210312'

# Mở file ảnh để gửi. Cần dùng 'rb' mode để đọc ở dạng binary
with open('./Dataset/train/102210312/1022103121.jpg', 'rb') as img:
    # Định nghĩa các file và data sẽ được gửi trong form
    files = {
        'testImage': img,
        'name': (None, name)
    }
    
    # Thực hiện POST request
    response = requests.post(url, files=files)

# Kiểm tra kết quả trả về
if response.status_code == 200:
    print("File and data were uploaded successfully.")
    print(response.text)  # In response trả về từ server
else:
    print(f"Failed to upload: {response.status_code}")
