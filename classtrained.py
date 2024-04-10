import os
import numpy as np

# Đường dẫn đến thư mục dữ liệu huấn luyện và file nhãn
train_dir = './Dataset/train'
labels_path = 'class_labels.npy'

# Tải danh sách nhãn từ file
labels = np.load(labels_path, allow_pickle=True).item()

# Lấy tên các thư mục trong thư mục huấn luyện (mỗi thư mục tương ứng với một lớp)
train_classes = os.listdir(train_dir)

# Kiểm tra xem tất cả các thư mục có trong nhãn không
missing_classes = [cls for cls in train_classes if cls not in labels]

# In kết quả
if missing_classes:
    print("MISS", missing_classes)
else:
    print("OK")
print(labels)