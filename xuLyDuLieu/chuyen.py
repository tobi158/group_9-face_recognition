import os
import shutil

def move_images_to_test_folder(source_folder="test", target_folder=r"C:\Users\BHXH\Desktop\venv_demo\dataset_split\test"):
    for label in os.listdir(source_folder):
        src_label_path = os.path.join(source_folder, label)
        if not os.path.isdir(src_label_path):
            continue

        # Tên thư mục nhãn trong 'test'
        dst_label_name = label.replace(" ", "_")
        dst_label_path = os.path.join(target_folder, dst_label_name)
        os.makedirs(dst_label_path, exist_ok=True)

        images = [f for f in os.listdir(src_label_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for image in images:
            src_image_path = os.path.join(src_label_path, image)
            dst_image_path = os.path.join(dst_label_path, image)
            shutil.move(src_image_path, dst_image_path)
            print(f"📦 Di chuyển: {src_image_path} → {dst_image_path}")

if __name__ == "__main__":
    move_images_to_test_folder()
