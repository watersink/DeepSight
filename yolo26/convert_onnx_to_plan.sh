docker run --gpus all --rm -v $(pwd):/workspace nvcr.io/nvidia/tensorrt:24.06-py3 \trtexec   --onnx=./runs/detect/train12/weights/best.onnx   --saveEngine=./runs/detect/train12/weights/model.plan    --fp16

docker run --gpus all --rm `
   -v "C:\Users\Administrator\Desktop\yolo26:/workspace" `
   -w /workspace `
   nvcr.io/nvidia/tensorrt:24.06-py3 `
   trtexec --onnx=./yolo26s.onnx --saveEngine=./model.plan --fp16