import cv2
import face_recognition
import os
import pickle
import json
from tqdm import tqdm
import pyttsx3
import matplotlib.pyplot as plt
from pathlib import Path
import random
import numpy as np


# Hàm lấy encoding từ 1 ảnh gốc #

def edit_end_encoding(img, resize_scale_temp):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (0, 0), fx=resize_scale_temp, fy=resize_scale_temp)
    locs = face_recognition.face_locations(small, model="cnn")
    locs = [(int(t/resize_scale_temp), int(r/resize_scale_temp), int(b/resize_scale_temp), int(l/resize_scale_temp)) for (t, r, b, l) in locs]
    encs_temp = face_recognition.face_encodings(img, locs)
    return encs_temp, locs, rgb


#hàm nhận diện và ghi file
def process_writeFile(img, rgb, encs, locs, known_encodings_temp, known_names_temp, tolerance_temp, is_image_temp, is_video_temp, metadata_temp, engine_temp, spoken_temp):
    total_faces = 0
    frame_id = 0
    #số lượng ảnh đã nhận diện trong foder output/faces
    existing_faces = [f for f in os.listdir("output/faces") if f.startswith("face_") and f.endswith(".jpg")]
    start_index = len(existing_faces)
    
    for i, (enc, (top, right, bottom, left)) in enumerate(zip(encs, locs)):
            name = "Unknown"
            if known_encodings_temp:
                distances = face_recognition.face_distance(known_encodings_temp, enc)
                if distances.min() < tolerance_temp:
                    idx = distances.argmin()
                    min_distance = distances.min()
                    name = known_names_temp[idx]
                    # Độ tin cậy có thể quy đổi bằng 1 - normalized_distance
                    confidence = (1.0 - min_distance) * 100  # chuyển sang phần trăm
                    label = f"{name} ({confidence:.2f}%)"
                else:
                    label = name  # vẫn là "Unknown"
            if name != "Unknown" and name not in spoken_temp:
                engine_temp.say(f"Xin chào {name}")
                engine_temp.runAndWait()
                spoken_temp.add(name)
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(img, label, (left- 40, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            face_crop = rgb[top:bottom, left:right]
            start_index += 1
            cv2.imwrite(f"output/faces/{name}_{start_index}.jpg", cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
            total_faces += 1
            if is_image_temp:
                metadata_temp.append({"name": name, "top": top, "right": right, "bottom": bottom, "left": left})
            elif is_video_temp:
                metadata_temp.append({"frame": frame_id, "name": name, "top": top, "right": right, "bottom": bottom, "left": left})
    if is_image_temp:
        return total_faces
    elif is_video_temp:
        return total_faces, frame_id



# Hàm hiển thị ảnh (thay thế imshow) #

def show_image(img, title="Result"):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 6))
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()



# Hàm encode khuôn mặt từ 1 file ảnh #

def process_image(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    if not face_locations:
        print(f"⚠️ Không phát hiện khuôn mặt: {image_path}")
        return None, None
    encodings = face_recognition.face_encodings(image, face_locations)
    if not encodings:
        print(f"⚠️ Không encode được khuôn mặt: {image_path}")
        return None, None
    return encodings[0], image


# Huấn luyện và lưu dữ liệu đã cân bằng lớp #

def build_face_dataset(dataset_folder="dataset", output_file="face_data.pkl", balance_data=True):
    def augment_image(image):
        aug_images = []

        # Lật ngang
        flip = cv2.flip(image, 1)
        aug_images.append(flip)

        # Xoay ±15 độ
        for angle in [-15, 15]:
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0) #tạo ra ma trận xoay
            rotated = cv2.warpAffine(image, M, (w, h)) # áp dụng ma trận xoay vào ảnh
            aug_images.append(rotated)

        # Tăng/giảm độ sáng
        for alpha in [0.6, 1.4]:
            bright = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
            aug_images.append(bright)

        # Thêm nhiễu Gaussian
        noise = image.copy().astype(np.float32)
        noise += np.random.normal(0, 10, image.shape).astype(np.float32)
        noise = np.clip(noise, 0, 255).astype(np.uint8)
        aug_images.append(noise)

        return aug_images

    dataset_path = Path(dataset_folder)
    known_encodings, known_names = [], []
    class_image_dict = {}

    total_images, success_count, fail_count = 0, 0, 0

    for person_dir in dataset_path.iterdir():
        if person_dir.is_dir():
            class_name = person_dir.name
            class_encodings = []
            for filename in os.listdir(person_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(person_dir, filename)
                    total_images += 1
                    encoding, original_image = process_image(path)
                    if encoding is not None:
                        class_encodings.append(encoding)
                        success_count += 1

                        # Tăng cường dữ liệu và trích xuất thêm encoding
                        aug_images = augment_image(original_image)
                        for aug_img in aug_images:
                            face_locations = face_recognition.face_locations(aug_img)
                            aug_encodings = face_recognition.face_encodings(aug_img, face_locations)
                            if aug_encodings:
                                class_encodings.append(aug_encodings[0])
                    else:
                        fail_count += 1
            if class_encodings:
                class_image_dict[class_name] = class_encodings

    # Cân bằng dữ liệu giữa các lớp nếu cần
    if balance_data:
        max_count = max(len(encs) for encs in class_image_dict.values())
        for class_name, encodings in class_image_dict.items():
            count = len(encodings)
            if count < max_count:
                multiplier = max_count // count
                remainder = max_count % count
                oversampled = encodings * multiplier + random.sample(encodings, remainder)
            else:
                oversampled = encodings
            known_encodings.extend(oversampled)
            known_names.extend([class_name] * len(oversampled))
    else:
        for class_name, encodings in class_image_dict.items():
            known_encodings.extend(encodings)
            known_names.extend([class_name] * len(encodings))

    # Lưu dữ liệu
    if known_encodings:
        with open(output_file, "wb") as f:
            pickle.dump({"encodings": known_encodings, "names": known_names}, f)
        print(f"✅ Đã lưu {len(known_encodings)} mẫu khuôn mặt vào '{output_file}'")
        print(f"📊 Tổng ảnh: {total_images} | Thành công: {success_count} | Lỗi: {fail_count}")
        print("➕ Đã áp dụng tăng cường dữ liệu.")
        if balance_data:
            print("⚖️ Đã cân bằng dữ liệu giữa các lớp.")
    else:
        print("❌ Không có khuôn mặt nào được lưu.")







# Nhận diện khuôn mặt từ ảnh hoặc video
def detect_faces(path, model_path="face_data.pkl", resize_scale=0.5, tolerance=0.6, display=True):
    try:
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        known_encodings, known_names = data["encodings"], data["names"]
    except FileNotFoundError:
        print(f"⚠️ File {model_path} không tồn tại! Vui lòng tạo dữ liệu trước.")
        return
    engine = pyttsx3.init()
    spoken = set()
    is_image = path.lower().endswith(('.jpg', '.jpeg', '.png'))
    is_video = path.lower().endswith(('.mp4', '.avi', '.mov'))

    os.makedirs("output/faces", exist_ok=True)

    metadata  = []

    if is_image:
        img = cv2.imread(path)
        if img is None:
            print(f"Không đọc được ảnh từ {path}. Kiểm tra lại đường dẫn.")
            return
        #xử lý và trích xuất đặc trưng ảnh
        encs, locs, rgb = edit_end_encoding(img, resize_scale_temp = resize_scale)

        #nhận diện ảnh và ghi vào file
        total_faces = process_writeFile(img, rgb = rgb, encs = encs, locs = locs, known_encodings_temp = known_encodings, known_names_temp = known_names, tolerance_temp = tolerance, is_image_temp = True, is_video_temp = False, metadata_temp = metadata, engine_temp = engine, spoken_temp = spoken)

            
        cv2.imwrite("output/result_image.jpg", img)
        print(f"Kết quả lưu tại: output/result_image.jpg. Số khuôn mặt: {total_faces}")

        if display:
            cv2.imshow("Result", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    elif is_video:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Không mở được video: {path}")
            return
        out = None
        pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Đang phân tích video")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            #xử lý và trích xuất đặc trưng ảnh
            encs, locs, rgb = edit_end_encoding(frame, resize_scale_temp = resize_scale)

            #nhận frame và ghi vào file
            total_faces, frame_id = process_writeFile(frame, rgb = rgb, encs = encs, locs = locs, known_encodings_temp = known_encodings, known_names_temp = known_names, tolerance_temp = tolerance, is_image_temp = False, is_video_temp = True, metadata_temp = metadata, engine_temp = engine, spoken_temp = spoken)

            if out is None:
                h, w = frame.shape[:2]
                out = cv2.VideoWriter("output/result_video.avi", fourcc, 20.0, (w, h))
            out.write(frame)

            if display and frame_id % 10 == 0:
                cv2.imshow("Video", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            frame_id += 1
            pbar.update(1)
        pbar.close()
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        print("✅ Video đã được lưu vào: output/result_video.avi")

    with open("output/metadata.json", "w", encoding="utf-8") as f:
        json.dump({"file": path, "faces": metadata, "total": total_faces}, f, indent=2)
    print(f"✅ Phát hiện {total_faces} khuôn mặt. Lưu metadata vào output/metadata.json")