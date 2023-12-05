import cv2
import argparse
from yoloseg_onnx.yoloseg import YOLOSeg

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Yolov8-seg onnx inference')
    parser.add_argument('-c', '--checkpoint_path', default="weights/yolov8n-seg-v1.onnx",
                        help='Dataset version with choices')
    parser.add_argument('-i','--image_path', default="data/5.png",
                        help='List of model IDs with default values')
    args = parser.parse_args()
    
    yoloseg = YOLOSeg(args.checkpoint_path, conf_thres=0.5, iou_thres=0.3)

    # Read image
    img = cv2.imread(args.image_path)

    # Segment Objects
    boxes, scores, class_ids, masks = yoloseg(img)

    # Draw masks
    combined_img = yoloseg.draw_masks(img)
    cv2.imshow("Detected Objects", combined_img)
    cv2.waitKey(0)