from ultralytics import YOLO

# Inference (prediction)
model = YOLO("ultralytics/runs/detect/train/weights/best.pt")  # Load the best trained weights for inference
model.predict(source='data/GPest14/test', conf=0.6, save_txt=True, save=True)  
