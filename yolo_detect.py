import os
from ultralytics import YOLO
import cv2


def run_yolo_detection(image_path):
    # Path to weights
    model_path = os.path.join(os.path.dirname(__file__), "weights", "boulder_yolo.pt")

    # Check if file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ YOLO model not found at {model_path}")

    # Load model
    model = YOLO(model_path)

    # Run prediction
    results = model(image_path)[0]

    # Extract bounding boxes
    boulders = []
    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
        boulders.append((x1, y1, x2, y2))

    # Optional visualization
    image = cv2.imread(image_path)
    for (x1, y1, x2, y2) in boulders:
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return image, boulders
