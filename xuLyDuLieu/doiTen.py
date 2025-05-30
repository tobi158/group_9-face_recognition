import os

def rename_images_in_folder(base_folder=r"C:\Users\BHXH\Desktop\venv_demo\fileTestAll", start_index=100):
    for label in os.listdir(base_folder):
        label_path = os.path.join(base_folder, label)
        if not os.path.isdir(label_path):
            continue

        images = [f for f in os.listdir(label_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        images.sort()  # Đảm bảo thứ tự nhất định

        for i, image_name in enumerate(images):
            if i >= 11:  # Giới hạn từ 100 đến 110 => 11 ảnh
                break
            new_name = f"{label.replace(' ', '_')}_{start_index + i}.jpg"
            old_path = os.path.join(label_path, image_name)
            new_path = os.path.join(label_path, new_name)
            os.rename(old_path, new_path)
            print(f"✅ Đổi tên: {old_path} -> {new_path}")

if __name__ == "__main__":
    rename_images_in_folder()
