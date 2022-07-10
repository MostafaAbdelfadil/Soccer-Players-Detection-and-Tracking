
import sys, os
sys.path.insert(0, os.path.abspath("../detect_to_standard/"))
from detection_visualizer import *
import numpy as np

sys.path.append(os.path.abspath("../keypoint_to_standard/"))
from keypoint_visualizer import *
import torchvision
from PIL import Image
import cv2
draw_threshold = 0.4

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



def show_all_from_dict(keypoints_list_list, bbox_dets_list_list, classes, joint_pairs, joint_names, rescale_img_factor = 1., path = None, output_folder_path = None, display_pose = False, flag_track= False, flag_method = False):
    cap=cv2.VideoCapture(path)

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
            else:
                img = draw_bbox(img, bbox, classes, method, -1, candidate["img_id"])  #for lighttrack

            if display_pose :
                pose_dict = keypoints_list_list[i][j]
                pose_keypoints_2d = pose_dict["keypoints"]
                joints = reshape_keypoints_into_joints(pose_keypoints_2d)

                if flag_track is True:
                    track_id = candidate["track_id"]
                    img = show_poses_from_python_data(img, joints, joint_pairs, joint_names, track_id = track_id)
                else:
                    img = show_poses_from_python_data(img, joints, joint_pairs, joint_names)
        img=np.uint8(img)
        vid.write(img)

    cap.release() 
    vid.release()       
    return



