import cv2
import face_recognition
import os
import pickle
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pyttsx3
from collections import deque

def augment_image(image):
    if np.random.rand() > 0.5:
        image = np.fliplr(image)
    factor = 1.0 + (np.random.rand() - 0.5) * 0.4
    image = np.clip(image * factor, 0, 255).astype(np.uint8)
    return image

def process_image(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    if not face_locations:
        return None
    encodings = face_recognition.face_encodings(image, face_locations)
    if not encodings:
        return None
    return encodings[0]

def build_face_dataset(dataset_folder=r"C:\Users\BHXH\Desktop\venv_demo\dataset_split\train", output_file="face_data_2.pkl"):
    dataset_path = Path(dataset_folder)
    known_encodings, known_names = [], []
    total_images, success_count, fail_count = 0, 0, 0

    for person_dir in dataset_path.iterdir():
        if person_dir.is_dir():
            for filename in os.listdir(person_dir):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    path = os.path.join(person_dir, filename)
                    total_images += 1
                    try:
                        image = face_recognition.load_image_file(path)
                        for _ in range(2):  # mỗi ảnh tạo 2 biến thể
                            aug_image = augment_image(image.copy())
                            face_locations = face_recognition.face_locations(aug_image)
                            encodings = face_recognition.face_encodings(aug_image, face_locations)
                            for encoding in encodings:
                                if isinstance(encoding, np.ndarray) and encoding.shape == (128,):
                                    known_encodings.append(encoding)
                                    known_names.append(person_dir.name)
                                    success_count += 1
                    except:
                        fail_count += 1

    if known_encodings:
        with open(output_file, "wb") as f:
            pickle.dump({"encodings": known_encodings, "names": known_names}, f)
        print(f"\nĐã lưu dữ liệu vào {output_file}")
        print(f"Tổng ảnh: {total_images} | Thành công: {success_count} | Thất bại: {fail_count}")
    else:
        print("⚠️ Không tìm thấy khuôn mặt nào để lưu.")

def detect_faces(path, model_path="face_data.pkl", resize_scale=0.5, tolerance=0.7, display=True):
    try:
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        raw_encodings, raw_names = data["encodings"], data["names"]
        known_encodings, known_names = [], []
        for enc, name in zip(raw_encodings, raw_names):
            if isinstance(enc, np.ndarray) and enc.shape == (128,):
                known_encodings.append(enc)
                known_names.append(name)
    except FileNotFoundError:
        print(f"⚠️ File {model_path} không tồn tại!")
        return

    engine = pyttsx3.init()
    spoken = set()
    os.makedirs("output/faces", exist_ok=True)
    metadata = []
    is_image = path.lower().endswith((".jpg", ".jpeg", ".png"))
    is_video = path.lower().endswith((".mp4", ".avi", ".mov"))

    def edit_end_encoding(img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (0, 0), fx=resize_scale, fy=resize_scale)
        locs = face_recognition.face_locations(small, model="cnn")
        locs = [(int(t/resize_scale), int(r/resize_scale), int(b/resize_scale), int(l/resize_scale)) for (t, r, b, l) in locs]
        encs = face_recognition.face_encodings(img, locs)
        return encs, locs, rgb

    def process_writeFile(img, rgb, encs, locs):
        existing_faces = [f for f in os.listdir("output/faces") if f.endswith(".jpg")]
        start_index = len(existing_faces)
        for enc, (top, right, bottom, left) in zip(encs, locs):
            name = "Unknown"
            label = name
            if known_encodings:
                distances = face_recognition.face_distance(known_encodings, enc)
                if distances.min() < tolerance:
                    idx = distances.argmin()
                    min_distance = distances.min()
                    name = known_names[idx]
                    confidence = (1.0 - min_distance) * 100
                    label = f"{name} ({confidence:.2f}%)"
            if name != "Unknown" and name not in spoken:
                engine.say(f"Xin chào {name}")
                engine.runAndWait()
                spoken.add(name)
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(img, label, (left - 40, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            face_crop = rgb[top:bottom, left:right]
            start_index += 1
            cv2.imwrite(f"output/faces/{name}_{start_index}.jpg", cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
            metadata.append({"name": name, "top": top, "right": right, "bottom": bottom, "left": left})

    if is_image:
        img = cv2.imread(path)
        encs, locs, rgb = edit_end_encoding(img)
        process_writeFile(img, rgb, encs, locs)
        cv2.imwrite("output/result_image.jpg", img)
        print("Đã lưu kết quả vào output/result_image.jpg")
        if display:
            cv2.imshow("Result", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    elif is_video:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print("Không mở được video")
            return
        out = None
        frame_id = 0
        pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Phân tích video")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            encs, locs, rgb = edit_end_encoding(frame)
            process_writeFile(frame, rgb, encs, locs)
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
        json.dump({"file": path, "faces": metadata, "total": len(metadata)}, f, indent=2)
    print(f"✅ Phát hiện {len(metadata)} khuôn mặt. Lưu metadata vào output/metadata.json")