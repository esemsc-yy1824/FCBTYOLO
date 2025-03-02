from ultralytics import YOLO

# Model training
model = YOLO("ultralytics/models/v8/yolov8n_FCBT.yaml")
model.train(**{'cfg': 'ultralytics/yolo/cfg/default.yaml'})  # Override default parameters using the specified config file