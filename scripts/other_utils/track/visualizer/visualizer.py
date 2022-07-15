
import sys, os
#sys.path.insert(0, os.path.abspath("../detect_to_standard/"))
#from detection_visualizer import *
import numpy as np

#sys.path.append(os.path.abspath("../keypoint_to_standard/"))
#from keypoint_visualizer import *
import torchvision
from PIL import Image
import cv2
draw_threshold = 0.4
classes = ['person']
def rescale_img(img,rescale_img_factor):
    shape = img.size
    w = shape[0]
    h = shape[1]
    desired_h = h*rescale_img_factor
    desired_w = w*rescale_img_factor
    img = torchvision.transforms.Resize([int(desired_h), int(desired_w)])(img)
    w_pad = (w - desired_w)/2.
    h_pad = (h - desired_h)/2.
    img = torchvision.transforms.Pad((int(w_pad),int(h_pad)))(img)
    return(img)



def show_all_from_dict(keypoints_list_list, bbox_dets_list_list, classes, rescale_img_factor = 1., path = None, output_folder_path = None, flag_track= False, flag_method = False):
    cap=cv2.VideoCapture(path)
    #ret,im=cap.read()

    img_name=0
    flag =0
    for i,bbox_dets_list in enumerate(bbox_dets_list_list) :
        _,img=cap.read()
        bbox_dets_init = bbox_dets_list[0]

        if flag == 0:
           h,w=img.shape[0],img.shape[1]
           fourcc=cv2.VideoWriter_fourcc(*'MPEG')
           vid=cv2.VideoWriter(output_folder_path,fourcc,25.0,(w,h))
           flag+=1


        for j,candidate in enumerate(bbox_dets_list) :

            bbox = np.array(candidate["bbox"]).astype(int)
            method = candidate["method"]

            if flag_track is True:
                track_id = candidate["track_id"]
                img = draw_bbox(img, bbox, classes, method, track_id = track_id, flag_method = flag_method)
  
        img=np.uint8(img)
        vid.write(img)
     
    cap.release() 
    vid.release()       
    return

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

