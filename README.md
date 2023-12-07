# yolov8-onnx
Repository to infer Yolov8-seg models from Ultralytics.

## 1. Information
Put your exported ONNX model in ./weights/ directory

If you do not have a trained and converted model yet, you can follow Ultralytics Documentation
- Train a pytorch model [Training Docs](https://docs.ultralytics.com/modes/train/#usage-examples)
- Convert to ONNX format [Export Docs](https://docs.ultralytics.com/modes/export/#usage-examples)
- Put your ONNX model in **weights/** directory


## 2. Requirements
- numpy==1.24.2
- onnxruntime_gpu==1.14.1
- opencv_python==4.8.1.78


## 3. Usage
Run the script **infer_onnx.py** with the following cli:
```
python infer_onnx.py -c [checkpoint_path] -i [image_path]
```

where:
- **checkpoint_path**: path to your ONNX model
- **image_path**: path to an image

example:

```
python pipeline.py -c weights/yolov8n-seg-v1.onnx -i data/5.png
```