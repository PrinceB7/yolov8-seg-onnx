import cv2
import argparse
from yoloseg_onnx.yoloseg import YOLOSeg

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Yolov8 onnx inference')
    parser.add_argument('-c', '--checkpoint_path', default="../yolov8/weights/yolov8n-seg-v1.onnx",
                        help='Dataset version with choices')
    parser.add_argument('-i','--image_path', default="../data-raw/4/5.png",
                        help='List of model IDs with default values')
    args = parser.parse_args()
    
    
    yoloseg = YOLOSeg(args.checkpoint_path, conf_thres=0.5, iou_thres=0.3)

    # Read image
    img = cv2.imread(args.image_path)

    # Detect Objects
    boxes, scores, class_ids, masks = yoloseg(img)

    # Draw detections
    combined_img = yoloseg.draw_masks(img)
    print(combined_img.shape)
    cv2.imshow("Detected Objects", combined_img)
    cv2.waitKey(0)