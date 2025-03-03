from ultralytics import YOLO

# Inference (prediction)
model = YOLO("./runs/detect/train9/weights/best.pt")  # Load the best trained weights. This pt file for an example
model.predict(source='data/GPest14/test', conf=0.6, save_txt=True, save=True)  
