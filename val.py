from ultralytics import YOLO

# Model evaluation
model = YOLO("./runs/detect/train9/weights/best.pt")  # Load the trained model weights. This pt file for an example
model.val(save_json=True, iou=0.6)  # Evaluate the model and save results in COCO format with IoU threshold set to 0.6
