from ultralytics import YOLOWorld
from ultralytics import YOLO

# Load a pretrained YOLOv8s-worldv2 model
#model = YOLOWorld("yolov8s-worldv2.pt")

# Train the model on the COCO8 dataset for 100 epochs
#results = model.train(data="static_electricity_discharge_post.yaml", epochs=100, imgsz=640)



# Initialize a YOLO-World model
#model = YOLO("runs/detect/train26/weights/best.pt")  # or select yolov8m/l-world.pt

# Define custom classes
#model.set_classes(["static_electricity_discharge_post"])

# Save the model with the defined offline vocabulary
#model.save("runs/detect/train26/weights/static_electricity_discharge_post.pt")
# Export the model


model = YOLO("runs/detect/train26/weights/static_electricity_discharge_post.pt")  # or select yolov8m/l-world.pt
model.export(format="onnx")
