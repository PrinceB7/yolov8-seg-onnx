import cv2
import argparse
import numpy as np
from nets.yolo_segment import Segment

class_names = ['ID']
colors = [(0,0,255)]


def draw_detections(image, boxes, scores, class_ids, mask_maps=None, mask_alpha=0.5):
    img_height, img_width = image.shape[:2]
    size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    mask_img = draw_masks(image, boxes, class_ids, mask_alpha, mask_maps)

    # Draw bounding boxes and labels of detections
    for box, score, class_id in zip(boxes, scores, class_ids):
        color = colors[class_id]

        x1, y1, x2, y2 = box.astype(int)

        # Draw rectangle
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, 2)

        label = class_names[class_id]
        caption = f'{label} {int(score * 100)}%'
        (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=size, thickness=text_thickness)
        th = int(th * 1.2)

        cv2.rectangle(mask_img, (x1, y1), (x1 + tw, y1 - th), color, -1)

        cv2.putText(mask_img, caption, (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

    return mask_img


def draw_masks(image, boxes, class_ids, mask_alpha=0.3, mask_maps=None):
    mask_img = image.copy()

    # Draw bounding boxes and labels of detections
    for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
        color = colors[class_id]

        x1, y1, x2, y2 = box.astype(int)

        # Draw fill mask image
        if mask_maps is None:
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
        else:
            crop_mask = mask_maps[i][y1:y2, x1:x2, np.newaxis]
            crop_mask_img = mask_img[y1:y2, x1:x2]
            crop_mask_img = crop_mask_img * (1 - crop_mask) + crop_mask * color
            mask_img[y1:y2, x1:x2] = crop_mask_img

    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Yolov8-seg onnx inference')
    parser.add_argument('-c', '--checkpoint_path', default="weights/yolov8n-seg-v1.onnx",
                        help='Dataset version with choices')
    parser.add_argument('-i','--image_path', default="data/5.png",
                        help='List of model IDs with default values')
    args = parser.parse_args()
    
    yoloseg = Segment(args.checkpoint_path, conf_thres=0.5, iou_thres=0.3)
    
    img = cv2.imread(args.image_path)
    boxes, scores, class_ids, masks = yoloseg(img)
    
    combined_img = draw_detections(img, boxes, scores, class_ids, masks)
    
    cv2.imshow("Detected Objects", combined_img)
    cv2.waitKey(0)