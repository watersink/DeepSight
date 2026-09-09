from ultralytics import YOLO

# Load a model
model = YOLO("yolo11s.yaml").load("yolo11s.pt")  # build from YAML and transfer weights
# Train the model
results = model.train(data="lifting_load.yaml", epochs=300, imgsz=640)


# Load a model
#model = YOLO("runs/detect/train12/weights/best.pt")
# Export the model
#model.export(format="onnx")
