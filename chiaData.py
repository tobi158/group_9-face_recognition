# face_recognition_pipeline.py

import os
import cv2
import numpy as np
import random
import shutil
import pickle
import json
import time
from pathlib import Path
from tqdm import tqdm
from collections import deque
import face_recognition
import pyttsx3

# ============================================
# 1. CHIA DỮ LIỆU THÀNH TRAIN / VAL / TEST
# ============================================
def split_dataset(input_dir="dataset", output_dir="dataset_split", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Tổng tỷ lệ phải bằng 1"
    random.seed(seed)

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"⚠️ Thư mục '{input_dir}' không tồn tại.")
        return

    for split in ["train", "val", "test"]:
        (output_path / split).mkdir(parents=True, exist_ok=True)

    for person_dir in input_path.iterdir():
        if not person_dir.is_dir():
            continue

        images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png")) + list(person_dir.glob("*.jpeg"))
        if len(images) < 3:
            print(f"⚠️ Bỏ qua '{person_dir.name}' vì có ít hơn 3 ảnh.")
            continue

        random.shuffle(images)
        total = len(images)
        n_train = int(total * train_ratio)
        n_val = int(total * val_ratio)

        split_map = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

        for split, files in split_map.items():
            person_output = output_path / split / person_dir.name
            person_output.mkdir(parents=True, exist_ok=True)
            for img in files:
                shutil.copy(img, person_output / img.name)

    print("✅ Đã chia dữ liệu thành công vào thư mục:", output_dir)


# ============================================
# 2. Xử LÝ 1 ẢNH THÀNH EMBEDDING
# ============================================
def process_image(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    if not face_locations:
        print(f"⚠️ Không phát hiện khuôn mặt: {image_path}")
        return None
    encodings = face_recognition.face_encodings(image, face_locations)
    if not encodings:
        print(f"⚠️ Không encode được khuôn mặt: {image_path}")
        return None
    return encodings[0]

# ============================================
# 3. XÂY DỰNG Dữ LIỆU EMBEDDING Từa FOLDER
# ============================================
def build_face_dataset(dataset_folder="dataset_split/train", output_file="face_data_train.pkl"):
    dataset_path = Path(dataset_folder)
    known_encodings, known_names = [], []
    total_images, success_count, fail_count = 0, 0, 0

    for person_dir in dataset_path.iterdir():
        if person_dir.is_dir():
            for filename in os.listdir(person_dir):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    path = os.path.join(person_dir, filename)
                    total_images += 1
                    encoding = process_image(path)
                    if isinstance(encoding, np.ndarray) and encoding.shape == (128,):
                        known_encodings.append(encoding)
                        known_names.append(person_dir.name)
                        success_count += 1
                    else:
                        fail_count += 1

    if known_encodings:
        with open(output_file, "wb") as f:
            pickle.dump({"encodings": known_encodings, "names": known_names}, f)
        print(f"\n✅ Đã lưu dữ liệu vào {output_file}")
        print(f"Tổng ảnh: {total_images} | Thành công: {success_count} | Thất bại: {fail_count}")
    else:
        print("⚠️ Không tìm thấy khuôn mặt nào để lưu.")

# ============================================
# 4. API NHẬN DIỆN TOÀN BỘ 1 FOLDER ẢNH
# ============================================
def recognize_folder(folder_path="images_to_recognize", model_path="face_data_train.pkl", resize_scale=0.5, tolerance=0.6):
    try:
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        known_encodings = data["encodings"]
        known_names = data["names"]
    except Exception as e:
        print(f"❌ Lỗi khi tải model: {e}")
        return

    engine = pyttsx3.init()
    spoken = set()

    image_paths = list(Path(folder_path).glob("*.jpg")) + list(Path(folder_path).glob("*.png")) + list(Path(folder_path).glob("*.jpeg"))
    if not image_paths:
        print("⚠️ Không tìm thấy ảnh trong thư mục.")
        return

    start_time = time.time()

    for image_path in tqdm(image_paths, desc="Đang nhận diện"):
        img = cv2.imread(str(image_path))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (0, 0), fx=resize_scale, fy=resize_scale)
        locs = face_recognition.face_locations(small)
        locs = [(int(t/resize_scale), int(r/resize_scale), int(b/resize_scale), int(l/resize_scale)) for (t, r, b, l) in locs]
        encs = face_recognition.face_encodings(rgb, locs)

        for enc, (top, right, bottom, left) in zip(encs, locs):
            name = "Unknown"
            if known_encodings:
                distances = face_recognition.face_distance(known_encodings, enc)
                if distances.min() < tolerance:
                    idx = distances.argmin()
                    name = known_names[idx]
            if name != "Unknown" and name not in spoken:
                engine.say(f"Xin chào {name}")
                engine.runAndWait()
                spoken.add(name)
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.imshow("Result", img)
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break

    end_time = time.time()
    duration = end_time - start_time
    print(f"⏱️ Thời gian nhận diện: {duration:.2f} giây cho {len(image_paths)} ảnh.")
    cv2.destroyAllWindows()


# ============================================
# Ví dụ sử dụng
# ============================================
if __name__ == "__main__":
    # B1. Chia dữ liệu
    split_dataset(r"C:\Users\BHXH\Desktop\venv_demo\lfw-funneled\lfw_funneled", "dataset_split")

    # B2. Xây dữ liệu huấn luyện
    build_face_dataset("dataset_split/train", "face_data_train.pkl")

    # B3. Nhận diện 1 folder ảnh
    recognize_folder("images_to_recognize", "face_data_train.pkl")