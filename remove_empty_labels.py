import os

label_dir = r'C:\Users\debni\PycharmProjects\isro\yolo_training\project\labels'
image_dir = r'C:\Users\debni\PycharmProjects\isro\yolo_training\project\images\train'

for file in os.listdir(label_dir):
    print(f"Checking: {file}")  # 👈 Add this line

    if file.endswith('.txt'):
        label_path = os.path.join(label_dir, file)
        if os.path.getsize(label_path) == 0:
            print(f"Removing empty label: {label_path}")
            os.remove(label_path)

            base = os.path.splitext(file)[0]
            for ext in ['.jpg', '.png', '.jpeg', '.tif']:
                image_path = os.path.join(image_dir, base + ext)
                if os.path.exists(image_path):
                    print(f"Removing image: {image_path}")
                    os.remove(image_path)
                    break
