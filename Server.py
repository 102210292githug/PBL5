import requests

label = "some_label"
last_label = None

print(label)
if label != "No Face" and last_label != label:
    url = "http://192.139.66.104:1713/customers/haveFace"
    
    # Đọc dữ liệu từ tệp hình ảnh
    with open('D:\Downloads\mew.jpg', 'rb') as img_file:
        files = {'image': ('filename.jpg', img_file, 'image/jpeg')}
        data = {'name': label}
        try:
            response = requests.post(url, files=files, data=data)
            response.raise_for_status()  # Sẽ raise exception nếu status code không phải 2xx
            print(response.status_code, response.text)
        except requests.exceptions.HTTPError as err:
            print(f"HTTP error occurred: {err}")  # In ra lỗi HTTP cụ thể
        except Exception as err:
            print(f"Other error occurred: {err}")  # In ra lỗi khác nếu có
    last_label = label
