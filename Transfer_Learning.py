import os
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint  # CheckPoint

# Đường dẫn tới mô hình VGG-Face không có lớp đầu và thư mục dữ liệu
model_path = "./Other Files/VGG_Face_pretrained_model_no_top.h5"
train_dir = "./Dataset/train"
validation_dir = "./Dataset/validation"

# Load mô hình VGG-Face không có lớp đầu
base_model = load_model(model_path)

# Đảm bảo rằng tất cả các lớp của mô hình cơ sở đều không được huấn luyện
for layer in base_model.layers:
    layer.trainable = False

# Xây dựng mô hình mới trên cơ sở mô hình đã tải
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(os.listdir(train_dir)), activation='softmax')  # Số lượng class tương ứng với số thư mục trong 'train_dir'
])

# Biên dịch mô hình
model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Chuẩn bị dữ liệu
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

# # Thiết lập callback để lưu mô hình với độ chính xác validation tốt nhất
# checkpoint_path = "./Other Files/best_model.h5"
# checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)

# Huấn luyện mô hình với callback
# model.fit(train_generator, epochs=30, validation_data=validation_generator, callbacks=[checkpoint_callback], verbose=1)
model.fit(train_generator, epochs=100, validation_data=validation_generator, verbose=1)

class_labels = train_generator.class_indices

# Lưu class_labels dưới dạng file .npy
np.save('./Other Files/class_labels.npy', class_labels)
# Lưu mô hình đã được huấn luyện
model.save("./Other Files/finetuned_vgg_face_model.h5")


# import os
# import numpy as np
# from tensorflow.keras.models import load_model, Sequential
# from tensorflow.keras.layers import Dense, Flatten, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import ModelCheckpoint  # CheckPoint

# # Đường dẫn tới mô hình VGG-Face không có lớp đầu và thư mục dữ liệu
# model_path = "./Other Files/VGG_Face_pretrained_model_no_top.h5"
# train_dir = "./Dataset/train"
# validation_dir = "./Dataset/validation"

# # Load mô hình VGG-Face không có lớp đầu
# base_model = load_model(model_path)

# # Đảm bảo rằng tất cả các lớp của mô hình cơ sở đều không được huấn luyện
# for layer in base_model.layers:
#     layer.trainable = False

# # Xây dựng mô hình mới trên cơ sở mô hình đã tải
# model = Sequential([
#     base_model,
#     Flatten(),
#     Dense(256, activation='relu'),
#     Dropout(0.5),
#     Dense(len(os.listdir(train_dir)), activation='softmax')  # Số lượng class tương ứng với số thư mục trong 'train_dir'
# ])

# # Biên dịch mô hình
# model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# # Chuẩn bị dữ liệu với data augmentation cho tập huấn luyện
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# # Chuẩn bị dữ liệu cho tập validation chỉ với việc rescale
# validation_datagen = ImageDataGenerator(rescale=1./255)

# # Tạo generators
# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='categorical'
# )

# validation_generator = validation_datagen.flow_from_directory(
#     validation_dir,
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='categorical'
# )

# # Thiết lập callback để lưu mô hình với độ chính xác validation tốt nhất
# checkpoint_path = "./Other Files/best_model.h5"
# checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)

# # Huấn luyện mô hình với callback
# model.fit(
#     train_generator,
#     epochs=10,  # Số epoch có thể điều chỉnh tùy ý
#     validation_data=validation_generator,
#     callbacks=[checkpoint_callback],
#     verbose=1
# )

# # Lưu mô hình cuối cùng sau khi huấn luyện
# model.save("./Other Files/finetuned_vgg_face_model.h5")
