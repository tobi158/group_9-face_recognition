import cv2
import face_recognition
import os
import pickle
import json
from tqdm import tqdm
import pyttsx3
import matplotlib.pyplot as plt

# Hàm hiển thị ảnh (tùy chọn thay imshow nếu cần)
def show_image(img, title="Result"):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 6))
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()

# Tạo dữ liệu khuôn mặt
def build_face_dataset(dataset_folder="dataset", output_file="face_data.pkl"):
    known_encodings, known_names = [], []
    for filename in os.listdir(dataset_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(dataset_folder, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image, model="cnn")
            if encodings:
                name = filename.split("_")[0]
                known_encodings.append(encodings[0])
                known_names.append(name)
    if known_encodings:
        with open(output_file, "wb") as f:
            pickle.dump({"encodings": known_encodings, "names": known_names}, f)
        print(f"✅ Lưu dữ liệu nhận diện vào {output_file}. Tổng: {len(known_names)} khuôn mặt.")
    else:
        print("⚠️ Không tìm thấy khuôn mặt nào để huấn luyện!")

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

    metadata, total_faces = [], 0

    if is_image:
        img = cv2.imread(path)
        if img is None:
            print(f"❌ Không đọc được ảnh từ {path}. Kiểm tra lại đường dẫn.")
            return
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (0, 0), fx=resize_scale, fy=resize_scale)
        locs = face_recognition.face_locations(small, model="cnn")
        locs = [(int(t/resize_scale), int(r/resize_scale), int(b/resize_scale), int(l/resize_scale)) for (t, r, b, l) in locs]
        encs = face_recognition.face_encodings(img, locs)

        for i, (enc, (top, right, bottom, left)) in enumerate(zip(encs, locs)):
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
            cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            face_crop = rgb[top:bottom, left:right]
            cv2.imwrite(f"output/faces/face_{i}.jpg", cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
            metadata.append({"name": name, "top": top, "right": right, "bottom": bottom, "left": left})
            total_faces += 1

        cv2.imwrite("output/result_image.jpg", img)
        print(f"✅ Kết quả lưu tại: output/result_image.jpg. Số khuôn mặt: {total_faces}")

        if display:
            cv2.imshow("Result", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    elif is_video:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"❌ Không mở được video: {path}")
            return
        out = None
        pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Đang phân tích video")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        frame_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small = cv2.resize(rgb, (0, 0), fx=resize_scale, fy=resize_scale)
            locs = face_recognition.face_locations(small, model="cnn")
            locs = [(int(t/resize_scale), int(r/resize_scale), int(b/resize_scale), int(l/resize_scale)) for (t, r, b, l) in locs]
            encs = face_recognition.face_encodings(frame, locs)

            for i, (enc, (top, right, bottom, left)) in enumerate(zip(encs, locs)):
                name = "Unknown"
                if known_encodings:
                    distances = face_recognition.face_distance(known_encodings, enc)
                    if distances.min() < tolerance:
                        idx = distances.argmin()
                        name = known_names[idx]
                if name != "Unknown" and name not in spoken:
                    engine.say(f"{name}")
                    engine.runAndWait()
                    spoken.add(name)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                metadata.append({"frame": frame_id, "name": name, "top": top, "right": right, "bottom": bottom, "left": left})
                total_faces += 1

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

# Chạy chương trình
def main():
    print("🚀 Bắt đầu chương trình nhận diện khuôn mặt...")
    dataset_folder = r"C:\Users\BHXH\Desktop\venv_demo\huanLuyen"
    test_path = r"C:\Users\BHXH\Desktop\venv_demo\video.mp4"

    build_face_dataset(dataset_folder=dataset_folder, output_file="face_data.pkl")
    detect_faces(path=test_path, model_path="face_data.pkl", display=True)

if __name__ == "__main__":
    main()