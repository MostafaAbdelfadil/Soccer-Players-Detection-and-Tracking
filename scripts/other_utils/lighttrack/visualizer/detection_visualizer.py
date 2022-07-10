
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from random import random as rand
import os
import cv2
import numpy as np
from utils_io_file import is_image
from utils_io_folder import create_folder
from utils_json import read_json_from_file

bbox_thresh = 0.4

# set up class names for COCO
num_classes = 81  # 80 classes + background class
classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
           'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
           'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
           'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
           'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
           'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
           'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
           'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
           'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def show_boxes_from_python_data(img, dets, classes, output_img_path, scale = 1.0):
    plt.cla()
    plt.axis("off")
    plt.imshow(img)
    for cls_idx, cls_name in enumerate(classes):
        cls_dets = dets[cls_idx]
        for det in cls_dets:
            bbox = det[:4] * scale
            color = (rand(), rand(), rand())
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                  bbox[2] - bbox[0],
                                  bbox[3] - bbox[1], fill=False,
                                  edgecolor=color, linewidth=2.5)
            plt.gca().add_patch(rect)

            if cls_dets.shape[1] == 5:
                score = det[-1]
                plt.gca().text(bbox[0], bbox[1],
                               '{:s} {:.3f}'.format(cls_name, score),
                               bbox=dict(facecolor=color, alpha=0.5), fontsize=9, color='white')
    plt.show()
    plt.savefig(output_img_path)
    return img





def draw_bbox(img, bbox, classes, method, track_id = -1, img_id = -1, flag_method = False):
    color=(0,0,255)
    #print(color)

    cv2.rectangle(img,
                  (bbox[0], bbox[1]),
                  (bbox[2], bbox[3]),
                  color = color,
                  thickness = 2)

    cls_name = classes[0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    color=(255,0,0)
    if track_id == -1:
        cv2.putText(img,
                    #'{:s} {:.2f}'.format(cls_name, score),
                    '{:s}'.format(cls_name),
                    (bbox[0], bbox[1]-5),
                    font,
                    fontScale=0.8,
                    color=color,
                    thickness = 2,
                    lineType = cv2.LINE_AA)
    else:
        cv2.putText(img,
                    #'{:s} {:.2f}'.format("ID:"+str(track_id), score),
                    '{:s}'.format("ID:"+str(track_id)),
                    (bbox[0], bbox[1]-5),
                    font,
                    fontScale=0.8,
                    color=color,
                    thickness = 2,
                    lineType = cv2.LINE_AA)

        if flag_method and method is not None and method is not 'spacial' :
            cv2.putText(img,
                        #'{:s} {:.2f}'.format("ID:"+str(track_id), score),
                        '{:s}'.format(method),
                        (bbox[0], bbox[3] + 10),
                        font,
                        fontScale=0.5,
                        color=color,
                        thickness = 2,
                        lineType = cv2.LINE_AA)

    return img


