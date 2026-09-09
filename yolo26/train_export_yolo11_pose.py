from ultralytics import YOLO

# Load a model
model = YOLO("yolo11m-pose.pt")  # load an official model

"""
# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    data="ultralytics/cfg/datasets/coco-pose.yaml",  # Path to dataset configuration file
    epochs=100,  # Number of training epochs
    imgsz=640,  # Image size for training
    device=7,  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
)



# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered



# Export the model
model.export(format='onnx',dynamic=False, simplify=False, opset=12)

# Export the model to TensorRT format
#model.export(format="engine")  # creates 'yolo11n.engine'

# Load the exported TensorRT model
#tensorrt_model = YOLO("yolo11s-pose.engine")

# Run inference
#results = tensorrt_model("https://ultralytics.com/images/bus.jpg")



"""
# Predict with the model
results = model("9-74-0004-000888.jpg")  # predict on an image
results[0].save()
# Access the results
for result in results:
    xy = result.keypoints.xy  # x and y coordinates
    xyn = result.keypoints.xyn  # normalized
    kpts = result.keypoints.data  # x, y, visibility (if available)
    print(kpts)

