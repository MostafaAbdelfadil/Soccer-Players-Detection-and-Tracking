
import sys, os, time
import cv2
import numpy as np
import collections

import torchvision

import torchvision.transforms as T
from collections import Counter

# detector utils
import sys
sys.path.append(os.path.abspath("../other_utils/track/utils"))
sys.path.append(os.path.abspath("../other_utils/track/visualizer"))

from utils_json import *
from visualizer import *
from utils_io_file import *
from utils_io_folder import *
from math import *
#from natsort import natsorted, ns
import scipy.optimize as scipy_opt
import motmetrics as mm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

import csv



def enlarge_bbox(bbox, scale, image_shape):
    min_x, min_y, max_x, max_y = bbox
    margin_x = int(0.5 * scale[0] * (max_x - min_x))
    margin_y = int(0.5 * scale[1] * (max_y - min_y))

    min_x -= margin_x
    max_x += margin_x
    min_y -= margin_y
    max_y += margin_y

    min_x = max(0,min_x)
    min_y = max(0,min_y)
    max_x = min(image_shape[1],max_x)
    max_y = min(image_shape[0],max_y)

    bbox_enlarged = [min_x, min_y, max_x, max_y]
    return bbox_enlarged



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



def player_detection(im, rescale_img_factor, model_detection, thres_detection):
    bbox_list = []
    score_list = []
    max_w = 150
    max_h = 150
    with torch.no_grad():
        
        x = [T.ToTensor()(im).to(torch.device('cuda'))]
        output, features = model_detection(x)
        output = output[0]
        scores = output['scores']
        labels =  output['labels']
        boxes = output['boxes']
        for i in range(len(scores)):
            if scores[i]>thres_detection :
                xmin,ymin,xmax,ymax = int(boxes[i][0]),int(boxes[i][1]),int(boxes[i][2]),int(boxes[i][3])
                if 0 < xmax-xmin < max_w and 0 < ymax-ymin < max_h :
                    bbox_list.append([xmin,ymin,xmax,ymax])
                    score_list.append(scores[i])
    return(bbox_list,score_list,features)


def track( model_detection, visual_feat_model, layer,
            rescale_img_factor,input_path,output_video_path, use_features,
            w_spacial, w_visual, use_IOU, spacial_iou_thresh, thres_detection,
             use_visual_feat, imagenet_model,
             use_GT_position, flag_method,
             keyframe_interval, visualize,
            use_filter_tracks, thres_count_ids,visual_metric,
            N_frame_lost_keep, N_past_to_keep, use_ReID_module,
            N_past_to_keep_reID, max_vis_feat, max_dist_factor_feat, max_vis_reID, max_dist_factor_reID,
            use_track_branch):

    total_time_DET = 0
    total_num_PERSONS = 0
    total_time_ALL = 0
    total_time_POSE = 0
    total_time_FEAT = 0
    st_time_total = time.time()

    bbox_dets_list = []
    frame_prev = -1
    frame_cur = 0
    img_id = -1
    next_id = 0
    bbox_dets_list_list = []
    track_ids_dict_list = []
    GT_bbox_list_list = []
    GT_idx_list_list = []
    bbox_lost_player_list = []
    track_feat_dict_list = []

    flag_mandatory_keyframe = False

    use_ReID_module=False
    acc = mm.MOTAccumulator(auto_id=True)
    path=input_path
    cap=cv2.cv2.VideoCapture(path)
    ret,imm=cap.read()

    N_IOU = 0
    N_feat = 0
    N_reID = 0
    counter=0
    while True:
       
        img_id += 1
        counter+=1
        ret,im=cap.read()
        if ret == False:
          break
        image_shape = im.shape


        frame_cur = img_id
        if (frame_cur == frame_prev):
            frame_prev -= 1

        if is_keyframe(img_id, keyframe_interval) or flag_mandatory_keyframe :

            flag_mandatory_keyframe = False
            bbox_dets_list = []

            # perform detection at keyframes
            st_time_detection = time.time()
           
            player_candidates, player_scores, img_feat = player_detection(im, rescale_img_factor, model_detection, thres_detection)
              
            end_time_detection = time.time()
            total_time_DET += (end_time_detection - st_time_detection)

            num_dets = len(player_candidates)
          

            # if nothing detected at keyframe, regard next frame as keyframe because there is nothing to track
            if num_dets <= 0 :

                flag_mandatory_keyframe = True

                # add empty result
                bbox_det_dict = {"img_id": img_id,
                                 "det_id":  0,
                                 "track_id": None,
                                 "bbox": [0, 0, 2, 2],
                                 "visual_feat": []
                                 }
                bbox_dets_list.append(bbox_det_dict)

                bbox_dets_list_list.append(bbox_dets_list)
                track_ids_dict_list.append({})

                flag_mandatory_keyframe = True
                continue

            total_num_PERSONS += num_dets

            if img_id > 0 :   # First frame does not have previous frame
                bbox_list_prev_frame = bbox_dets_list_list[img_id - 1].copy()
                track_ids_dict_prev = track_ids_dict_list[img_id - 1].copy()

            # Perform data association

            for det_id in range(num_dets):

                # obtain bbox position
                bbox_det = player_candidates[det_id]
                score_det = float(player_scores[det_id].cpu())

                # enlarge bbox by 20% with same center position
                bbox_det = enlarge_bbox(bbox_det, [0.,0.], image_shape)

                # update current frame bbox
                bbox_det_dict = {"img_id": img_id,
                                 "det_id": det_id,
                                 "bbox": bbox_det,
                                 "score_det": score_det}

                if img_id == 0 or len(bbox_list_prev_frame) == 0 :   # First frame, all ids are assigned automatically
                    track_id = next_id
                    next_id += 1
                    method = None

                else : # Perform data association

                    if use_IOU :  # use IOU as first criteria
                        spacial_intersect = get_spacial_intersect(bbox_det, bbox_list_prev_frame)
                        track_id, match_index = get_track_id_SpatialConsistency(spacial_intersect, bbox_list_prev_frame, spacial_iou_thresh,-1)
                    else :
                        track_id = -1

                    if track_id != -1:
                        method = 'spacial'
                    else :
                        method = None

                # update current frame bbox
                bbox_det_dict = {"img_id": img_id,
                                 "det_id": det_id,
                                 "track_id": track_id,
                                 "bbox": bbox_det,
                                 "score_det": score_det,
                                 "method": method,
                                 "visual_feat": []
                                 }

                bbox_dets_list.append(bbox_det_dict)

            # Check for repetitions in track ids and remove them.
            track_ids = [bbox_det_dict["track_id"] for bbox_det_dict in bbox_dets_list]
            track_ids_dict = collections.defaultdict(list)
            for idx, track in enumerate(track_ids) :
                track_ids_dict[track].append(idx)
            keys = list(track_ids_dict.keys())
            for track in keys :
                if len(track_ids_dict[track]) > 1 :
                    for el in track_ids_dict[track] :
                        bbox_dets_list[el]["track_id"] = -1
                
                        bbox_dets_list[el]["method"] = None
                    del track_ids_dict[track]    
            if img_id > 0 and len(bbox_list_prev_frame) > 0 :

                # Remove already assigned elements in the previous frame.
                remaining_det_id_list = []
                prev_to_remove = []
                for det_id in range(num_dets):
                    track_id = bbox_dets_list[det_id]["track_id"]
                    if track_id == -1 :
                        remaining_det_id_list.append(det_id)
                    else :
                        prev_idx = track_ids_dict_prev[track_id]
                        if prev_idx[0] not in prev_to_remove:
                          prev_to_remove.append(prev_idx[0])
                          N_IOU+=1
                for index in sorted(prev_to_remove, reverse=True):
                    del bbox_list_prev_frame[index]

                # For candidates that are not associated yet

                if len(bbox_list_prev_frame) == 0 or (not use_features and not use_ReID_module) :

                    # If no more candidates in previous frame : assign new ids to remaining detections
                    for det_id in remaining_det_id_list :


                        next_iddd=-1
                        bbox_dets_list[det_id]["track_id"] = next_iddd
                        bbox_dets_list[det_id]["method"] = None
                        track_ids_dict[next_iddd].append(det_id)

                elif len(remaining_det_id_list) > 0 :

                    #For each remaining detections, perform association with a combinaison of features.
                    if (use_ReID_module or use_visual_feat) and not imagenet_model :
                        if use_GT_position :
                            img_feat, image_sizes = get_img_feat_FasterRCNN(visual_feat_model, im, rescale_img_factor)
                        img_feat_prev, image_sizes = get_img_feat_FasterRCNN(visual_feat_model, im, rescale_img_factor)

                    past_track_bbox_list_list = []

                    for bbox_prev_dict in bbox_list_prev_frame :

                        prev_track_id = bbox_prev_dict['track_id']
                        past_track_idx_list = []
                        past_track_bbox_list = []
                        for i in range(1,min(N_past_to_keep,img_id)+1):
                            past_track_ids_dict = track_ids_dict_list[img_id-i]
                            if prev_track_id in past_track_ids_dict.keys() :
                                idx = past_track_ids_dict[prev_track_id][0]
                                past_track_idx_list.append(idx)
                                past_track_bbox_list.append(bbox_dets_list_list[img_id-i][idx])

                        for past_track_bbox in past_track_bbox_list :
            
                            if use_visual_feat :
                                if not list(past_track_bbox["visual_feat"]) :
                                    st_time_feat = time.time()
                                    if imagenet_model :
                                        visual_feat = get_visual_feat_imagenet(visual_feat_model,layer,past_track_bbox, rescale_img_factor)
                                    else :
                                        visual_feat = get_visual_feat_fasterRCNN(visual_feat_model,past_track_bbox,img_feat_prev,image_sizes,use_track_branch)
                                    end_time_feat = time.time()
                                    total_time_FEAT += (end_time_feat - st_time_feat)
                                else :
                                    visual_feat = past_track_bbox["visual_feat"]
                            else :
                                visual_feat = []

                            past_track_bbox["visual_feat"] = visual_feat

                        past_track_bbox_list_list.append(past_track_bbox_list)

                    for det_id in remaining_det_id_list :

                        bbox_det_dict = bbox_dets_list[det_id]

                        if use_visual_feat :
                            st_time_feat = time.time()
                            if imagenet_model :
                                visual_feat = get_visual_feat_imagenet(visual_feat_model,layer,bbox_det_dict, rescale_img_factor)
                            else :
                                visual_feat = get_visual_feat_fasterRCNN(visual_feat_model,bbox_det_dict, img_feat,image_sizes,use_track_branch)
                            end_time_feat = time.time()
                            total_time_FEAT += (end_time_feat - st_time_feat)
                        else :
                            visual_feat = []

                        bbox_det_dict["visual_feat"] = visual_feat

                    if use_features :
                        log = ''
                        bbox_dets_list, bbox_list_prev_frame, past_track_bbox_list_list, track_ids_dict, N_feat = feature_matching(bbox_dets_list,remaining_det_id_list, bbox_list_prev_frame,
                                                        past_track_bbox_list_list, track_ids_dict, visual_metric, max_dist_factor_feat, max_vis_feat, w_visual, w_spacial,
                                                        use_visual_feat, image_shape, log, N_past_to_keep, N_feat)

                    if use_ReID_module :

                        # Adjust lost player list
                        bbox_lost_player_list = [bbox_lost_player for bbox_lost_player in bbox_lost_player_list if img_id - bbox_lost_player['img_id'] < N_frame_lost_keep]
                        bbox_lost_player_list += bbox_list_prev_frame

                        past_track_bbox_list_list_reID = []

                        for bbox_prev_dict in bbox_lost_player_list :
                            prev_track_id = bbox_prev_dict['track_id']
                            prev_im_id = bbox_prev_dict['img_id']
                            past_track_idx_list = []
                            past_track_bbox_list = []
                            for i in range(min(N_past_to_keep_reID,prev_im_id+1)):
                                past_track_ids_dict = track_ids_dict_list[prev_im_id-i]
                                if prev_track_id in past_track_ids_dict.keys() :
                                    idx = past_track_ids_dict[prev_track_id][0]
                                    past_track_idx_list.append(idx)
                                    past_track_bbox_list.append(bbox_dets_list_list[prev_im_id-i][idx])

                            for past_track_bbox in past_track_bbox_list :

                  

                                if use_visual_feat :
                                    if not list(past_track_bbox["visual_feat"]) :
                                        st_time_feat = time.time()
                                        if imagenet_model :
                                            visual_feat = get_visual_feat_imagenet(visual_feat_model,layer,past_track_bbox, rescale_img_factor)
                                        else :
                                            visual_feat = get_visual_feat_fasterRCNN(visual_feat_model,past_track_bbox,img_feat_prev,image_sizes,use_track_branch)
                                        end_time_feat = time.time()
                                        total_time_FEAT += (end_time_feat - st_time_feat)
                                    else :
                                        visual_feat = past_track_bbox["visual_feat"]
                                else :
                                    visual_feat = []

                                past_track_bbox["visual_feat"] = visual_feat

                            past_track_bbox_list_list_reID.append(past_track_bbox_list)


                        # Get non_associated dets
                        remaining_det_id_list = []
                        for det_id in range(num_dets):
                            track_id = bbox_dets_list[det_id]["track_id"]
                            if track_id == -1 :
                                remaining_det_id_list.append(det_id)
                        # Re-ID module
                        if len(remaining_det_id_list) > 0 and len(bbox_lost_player_list) > 0 :
                            log = ''
                            bbox_dets_list, bbox_lost_player_list, past_track_bbox_list_list_reID, track_ids_dict, N_reID = feature_matching(bbox_dets_list,remaining_det_id_list, bbox_lost_player_list,
                                                            past_track_bbox_list_list_reID, track_ids_dict, visual_metric, max_dist_factor_reID, max_vis_reID, w_visual, w_spacial, use_visual_feat,
                                                             image_shape, log, N_past_to_keep_reID, N_reID)

                    
                    # if still can not find a match from previous frame, then -1
                    if img_id ==2:
                        num_im=img_id-2
                        cou=1
                    else:
                        num_im=img_id-3
                        cou=2 
                    for det_id in range(num_dets):
                        track_id = bbox_dets_list[det_id]["track_id"]
                        if track_id==-1:
                            if img_id>1:
                              for i in reversed(range(num_im,num_im+cou)):
                                  boxes_list=bbox_dets_list_list[ i].copy()
                                  bbox_det = player_candidates[det_id]
                                  bbox_det = enlarge_bbox(bbox_det, [0.,0.], image_shape)
                                  spacial_intersect = get_spacial_intersect(bbox_det, boxes_list)
                                  track_id, match_index = get_track_id_SpatialConsistency(spacial_intersect, boxes_list, spacial_iou_thresh,-1)

                                  if  track_id != -1:
                                      for de in range(num_dets):
                                        x,y=bbox_dets_list[de]["det_id"],bbox_dets_list[de]["track_id"]
                                        if y==track_id:
                                            next_iddd=-1    
                                            bbox_dets_list[x]["track_id"] = next_iddd
                                            bbox_dets_list[x]["method"] = None
                                            track_ids_dict[next_iddd].append(x)
                                            #next_id+=1
                                      break

                            next_iddd=track_id
                            bbox_dets_list[det_id]["track_id"] = next_iddd
                            bbox_dets_list[det_id]["method"] = None
                            track_ids_dict[next_iddd].append(det_id)             

       
                        
                else :
                    pass

            if img_id ==2:
                num_im=img_id-2
                cou=1
            else:
                num_im=img_id-3
                cou=2 
            for det_id in range(num_dets):
                track_id = bbox_dets_list[det_id]["track_id"]
                if track_id==-1:
                    if img_id>1:
                        for i in reversed(range(num_im,num_im+cou)):
                          boxes_list=bbox_dets_list_list[i].copy()
                          bbox_det = player_candidates[det_id]
                          bbox_det = enlarge_bbox(bbox_det, [0.,0.], image_shape)
                          spacial_intersect = get_spacial_intersect(bbox_det, boxes_list)
                          track_id, match_index = get_track_id_SpatialConsistency(spacial_intersect, boxes_list, spacial_iou_thresh,-1)
                          if  track_id != -1:
                              for de in range(num_dets):
                                  x,y=bbox_dets_list[de]["det_id"],bbox_dets_list[de]["track_id"]
                                  if y==track_id:
                                      for j in range(num_dets):
                                          if j not in track_ids_dict:
                                            next_iddd=j
                                            break
                                      bbox_dets_list[x]["track_id"] = next_iddd
                                      bbox_dets_list[x]["method"] = None
                                      track_ids_dict[next_iddd].append(x)
                              break

                    next_iddd=track_id
                    if  track_id==-1 or img_id ==1 :
                         for j in range(num_dets):
                             if j not in track_ids_dict:
                                 next_iddd=j
                                 break                     
                    bbox_dets_list[det_id]["track_id"] = next_iddd
                    bbox_dets_list[det_id]["method"] = None
                    track_ids_dict[next_iddd].append(det_id)        

            # update frame

            bbox_dets_list_list.append(bbox_dets_list)
            track_ids_dict_list.append(track_ids_dict)
            frame_prev = frame_cur

        else:
            ''' NOT KEYFRAME: multi-target pose tracking '''
            print('we only work with keyframes for now')

    cap.release()       



    if use_filter_tracks :
        bbox_dets_list_list = filter_tracks(bbox_dets_list_list, thres_count_ids)
    ''' 1. statistics: get total time for lighttrack processing'''
    end_time_total = time.time()
    total_time_ALL += (end_time_total - st_time_total)

    print("N IOU : ", N_IOU)
    print("N FEAT : ", N_feat)
    print("N REID : ", N_reID)

    # visualization
    if visualize :
        print("Visualizing Tracking Results...")
       
        show_all_from_dict([], bbox_dets_list_list, classes, 
        rescale_img_factor = rescale_img_factor
        ,path = path, output_folder_path = output_video_path,
        flag_track = True, flag_method = flag_method)
        total_num_FRAMES=counter
        print("Visualization Finished!")
        print("Finished video {}".format(output_video_path))

        ''' Display statistics '''
        print("total_time_ALL: {:.2f}s".format(total_time_ALL))
        print("total_time_DET: {:.2f}s".format(total_time_DET))
        print("total_time_POSE: {:.2f}s".format(total_time_POSE))
        print("total_time_FEAT: {:.2f}s".format(total_time_FEAT))
        print("total_time_TRACK: {:.2f}s".format(total_time_ALL - total_time_DET - total_time_POSE - total_time_FEAT))
        print("total_num_FRAMES: {:d}".format(total_num_FRAMES))
        print("total_num_PERSONS: {:d}".format(total_num_PERSONS))
        print("Average FPS: {:.2f}fps".format(total_num_FRAMES / total_time_ALL))
        print("Average FPS for Detection only : {:.2f}fps".format(total_num_FRAMES / (total_time_DET)))
        print("Average FPS excluding Detection: {:.2f}fps".format(total_num_FRAMES / (total_time_ALL - total_time_DET)))
        print("Average FPS for framework only: {:.2f}fps".format(total_num_FRAMES / (total_time_ALL - total_time_DET - total_time_POSE - total_time_FEAT) ))



   

def feature_matching(bbox_dets_list, remaining_det_id_list, bbox_list_prev_frame, past_track_bbox_list_list, track_ids_dict,
        visual_metric, max_dist_factor, max_vis, w_visual, w_spacial, use_visual_feat
        , image_shape, log, N_past_to_keep, N_meth, show_track = False, show_NN = False):

    dist_tab = []
    weight_tab = []
    spacial_dist = np.array([list(get_spacial_distance(bbox_dets_list[det_id]["bbox"], past_track_bbox_list_list, image_shape)) for det_id in remaining_det_id_list])
    dist_tab.append(spacial_dist)
    weight_tab.append(w_spacial)
    if use_visual_feat :
        visual_dist = np.array([list(get_visual_similarity(bbox_dets_list[det_id]['visual_feat'], past_track_bbox_list_list, N_past_to_keep, metric = visual_metric)) for det_id in remaining_det_id_list])
        dist_tab.append(visual_dist)
        weight_tab.append(w_visual)




    # for 5 players : display first visual similarity players for control
    if show_track :
        for i in range(5):
            bbox_det = bbox_dets_list[-i]
            img_path = bbox_det['imgpath']
            img = Image.open(img_path).convert('RGB')
            img = rescale_img(img,0.6)
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            bbox = bbox_det['bbox']
            patch = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            cv2.imshow('ref',patch)
            for j in range(len(past_track_bbox_list_list[i])) :
                bbox_past_frame = past_track_bbox_list_list[i][j]
                img_path = bbox_past_frame['imgpath']
                img = Image.open(img_path).convert('RGB')
                img = rescale_img(img,0.6)
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                bbox = bbox_past_frame['bbox']
                patch = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                cv2.imsave(str(id)+'.png',patch)

    if show_NN :
        for i in range(3):
            det_id = remaining_det_id_list[i]
            bbox_curr_frame = bbox_dets_list[det_id]
            img_path = bbox_curr_frame['imgpath']
            img = Image.open(img_path).convert('RGB')
            img = rescale_img(img,0.6)
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            bbox = bbox_curr_frame['bbox']
            patch = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            best_visual_similarities = np.argsort(visual_dist[i])
            cv2.imshow('ref',patch)
            for id,j in enumerate(best_visual_similarities[:2]) :
                past_track = past_track_bbox_list_list[j]
                bbox_prev_frame = past_track[0]
                img_path = bbox_prev_frame['imgpath']
                img = Image.open(img_path).convert('RGB')
                img = rescale_img(img,0.6)
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                bbox = bbox_prev_frame['bbox']
                patch = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                cv2.imsave(str(i)+'_'+str(id)+'.png',patch)

 

    distx = image_shape[0]/max_dist_factor
    disty = image_shape[1]/max_dist_factor
    max_dist = np.sqrt(distx**2+disty**2)/np.sqrt(image_shape[0]**2 + image_shape[1]**2)
    matches = compute_matches(dist_tab, weight_tab, max_dist = max_dist, max_vis = max_vis, bipart_match_algo = 'hungarian')

    idx_to_remove_prev = []
    for i,match in enumerate(matches):
        track_id = bbox_list_prev_frame[match]["track_id"]
        if match != -1:
            det_id = remaining_det_id_list[i]
            bbox_dets_list[det_id]["track_id"] = track_id
            bbox_dets_list[det_id]["method"] = log
            track_ids_dict[track_id].append(det_id)
            idx_to_remove_prev.append(match)
            N_meth += 1

        # if still can not find a match from previous frame, then -1

        if match == -1 :
            det_id = remaining_det_id_list[i]
            bbox_dets_list[det_id]["track_id"] = -1
            bbox_dets_list[det_id]["method"] = None
          

    for index in sorted(idx_to_remove_prev, reverse=True):
        del past_track_bbox_list_list[index]
        del bbox_list_prev_frame[index]

    return(bbox_dets_list, bbox_list_prev_frame, past_track_bbox_list_list, track_ids_dict, N_meth)




def filter_tracks(bbox_dets_list_list, thres_count_ids = 1):
    all_track_ids = [bbox_det['track_id'] for bbox_dets_list in bbox_dets_list_list for bbox_det in bbox_dets_list]
    ids_counter = Counter(all_track_ids)
    track_ids_to_remove = []
    n = 0
    for k,v in ids_counter.items() :
        if v <= thres_count_ids :
            track_ids_to_remove.append(k)
            n+=1
    print(n, 'tracks removed out of ', len(ids_counter.keys()))
    for b,bbox_dets_list in enumerate(bbox_dets_list_list) :
        remlist = []
        for bb,bbox_det in enumerate(bbox_dets_list) :
            track_id = bbox_det['track_id']
            if track_id in track_ids_to_remove :
                remlist.append(bb)
        for index in sorted(remlist, reverse=True):
            del bbox_dets_list_list[b][index]
    return(bbox_dets_list_list)



def get_track_id_SpatialConsistency(spacial_similarities, bbox_list_prev_frame, spacial_thresh,x):

    if len(spacial_similarities) == 1 :
        if spacial_similarities[0] > spacial_thresh :
            max_index = 0
            track_id = bbox_list_prev_frame[max_index]["track_id"]
            return track_id, max_index
        else :
            return x, None

    sim_argsort = np.argsort(spacial_similarities)
    sim_sort = spacial_similarities[sim_argsort]
    if sim_sort[-1] <= 0 :
        return x, None
    elif sim_sort[-1] > 0 and sim_sort[-2] <= 0 :
        max_index = sim_argsort[-1]
        track_id = bbox_list_prev_frame[max_index]["track_id"]
        return track_id, max_index
    else :
        if sim_sort[-1]>0.5*sim_sort[-2] and sim_sort[-1] > spacial_thresh :
            max_index = sim_argsort[-1]
            track_id = bbox_list_prev_frame[max_index]["track_id"]
            return track_id, max_index
        else :
            return x, None

def get_spacial_intersect(bbox_cur_frame, bbox_list_prev_frame):

    spacial_sim = np.zeros(len(bbox_list_prev_frame))

    for bbox_index, bbox_det_dict in enumerate(bbox_list_prev_frame):
        bbox_prev_frame = bbox_det_dict["bbox"]
        boxA = bbox_cur_frame
        boxB = bbox_prev_frame
        spacial_sim[bbox_index] = iou(boxA, boxB)

    return(spacial_sim)

def get_spacial_distance(bbox_cur_frame, past_track_bbox_list_list, image_shape):

    bbox_list_prev_frame = [past_track_bbox_list[0] for past_track_bbox_list in past_track_bbox_list_list]

    spacial_sim = np.zeros(len(bbox_list_prev_frame))

    for bbox_index, bbox_det_dict in enumerate(bbox_list_prev_frame):
        bbox_prev_frame = bbox_det_dict["bbox"]
        centAx = (bbox_cur_frame[0]+bbox_cur_frame[2])/2.
        centAy = (bbox_cur_frame[1]+bbox_cur_frame[3])/2.
        centBx = (bbox_prev_frame[0]+bbox_prev_frame[2])/2.
        centBy = (bbox_cur_frame[1]+bbox_cur_frame[3])/2.
        distx = np.abs(centAx-centBx)
        disty = np.abs(centAy-centBy)
        dist = np.sqrt(distx**2+disty**2)/np.sqrt(image_shape[0]**2 + image_shape[1]**2)
        spacial_sim[bbox_index] = dist

    return(spacial_sim)

def get_visual_similarity(feat, past_track_bbox_list_list, N_past_to_keep, metric = 'cos_similarity') :
    weights = np.array([(1/2)**n for n in range(N_past_to_keep)])
    weights = np.array([(1)**n for n in range(N_past_to_keep)])
    res = []
    feat = np.array(feat)
    for past_track_bbox_list in past_track_bbox_list_list :
        feat_vector = np.array([past_track_bbox_list[i]["visual_feat"].numpy()*weights[i] for i in range(len(past_track_bbox_list))])
        feat_vector = np.mean(feat_vector,axis=0)
        if metric == 'cos_similarity' :
            res.append(np.dot(feat/np.linalg.norm(feat),feat_vector/np.linalg.norm(feat_vector)))
        if metric == 'correlation' :
            res.append(np.dot(feat,feat_vector))
        if metric == 'l1' :
            res.append(np.linalg.norm(feat-feat_vector,1))
        if metric == 'l2' :
            res.append(np.linalg.norm(feat-feat_vector,2))
    return(np.array(res))






def iou(boxA, boxB):
    # box: (x1, y1, x2, y2)
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou






def get_visual_feat_imagenet(model,layer,data, rescale_img_factor):
    with torch.no_grad():
        scaler = transforms.Scale((224, 224))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        to_tensor = transforms.ToTensor()
        img = Image.open(data['imgpath']).convert('RGB')
        img = rescale_img(img,rescale_img_factor)
        bbox = data['bbox']
        box = (bbox[0],bbox[1],bbox[2],bbox[3])
        patch = img.crop(box)
        t_img = Variable(normalize(to_tensor(scaler(patch))).unsqueeze(0)).to(torch.device('cuda'))
        my_embedding = torch.zeros(2048)
        def copy_data(m, i, o):
            my_embedding.copy_(o.data.squeeze())
        h = layer.register_forward_hook(copy_data)
        model(t_img)
        h.remove()
        feat = my_embedding
    return feat

def get_img_feat_FasterRCNN(model,im,rescale_img_factor):
    with torch.no_grad():
        image = [T.ToTensor()(im).to(torch.device('cuda'))]
        image,_ = model.transform(image, None)
        features = model.backbone(image.tensors)
        return(features,image.image_sizes)

def get_visual_feat_fasterRCNN(model,data,features,image_sizes,use_track_branch):
    with torch.no_grad():
        bbox = data['bbox']
        box = (float(bbox[0]),float(bbox[1]),float(bbox[2]),float(bbox[3]))
        proposals = [torch.tensor([box]).to(torch.device('cuda'))]
        if not use_track_branch :
            feat = model.roi_heads(features, proposals, image_sizes, get_feature_only=True)[0].cpu()
        else :
            feat = model.track_heads(features, proposals, image_sizes)[0].cpu()
            feat2 = model.roi_heads(features, proposals, image_sizes, get_feature_only=True)[0].cpu()
        return feat



def is_keyframe(img_id, interval=10):
    if img_id % interval == 0:
        return True
    else:
        return False



def x1y1x2y2_to_xywh(det):
    x1, y1, x2, y2 = det
    w, h = int(x2) - int(x1), int(y2) - int(y1)
    return [x1, y1, w, h]


def xywh_to_x1y1x2y2(det):
    x1, y1, w, h = det
    x2, y2 = x1 + w, y1 + h
    return [x1, y1, x2, y2]



def bipartite_matching_greedy(C):
    """
    Computes the bipartite matching between the rows and columns, given the
    cost matrix, C.
    """
    C = C.copy()  # to avoid affecting the original matrix
    prev_ids = []
    cur_ids = []
    row_ids = np.arange(C.shape[0])
    col_ids = np.arange(C.shape[1])
    while C.size > 0:
        # Find the lowest cost element
        i, j = np.unravel_index(C.argmin(), C.shape)
        # Add to results and remove from the cost matrix
        row_id = row_ids[i]
        col_id = col_ids[j]
        prev_ids.append(row_id)
        cur_ids.append(col_id)
        C = np.delete(C, i, 0)
        C = np.delete(C, j, 1)
        row_ids = np.delete(row_ids, i, 0)
        col_ids = np.delete(col_ids, j, 0)
    return prev_ids, cur_ids


def compute_matches(similarity_tab, weight_tab, max_dist = 100., max_vis = 100., bipart_match_algo = 'hungarian'):

    # matches structure keeps track of which of the current boxes matches to
    # which box in the previous frame. If any idx remains -1, it will be set
    # as a new track.

    C = np.average(np.array(similarity_tab), axis = 0, weights=weight_tab).transpose()
    C_dist = np.array(similarity_tab[0]).transpose()
    C_vis = np.array(similarity_tab[1]).transpose()

    matches = -np.ones((C.shape[1],), dtype=np.int32)

    if bipart_match_algo == 'hungarian':
        prev_inds, next_inds = scipy_opt.linear_sum_assignment(C)
    elif bipart_match_algo == 'greedy':
        prev_inds, next_inds = bipartite_matching_greedy(C)
    else:
        raise NotImplementedError('Unknown matching algo: {}'.format(
            bipart_match_algo))
    assert(len(prev_inds) == len(next_inds))

    for i in range(len(prev_inds)):
        cost = C[prev_inds[i], next_inds[i]]
        dist = C_dist[prev_inds[i], next_inds[i]]
        vis = C_vis[prev_inds[i], next_inds[i]]
        if dist < max_dist and vis < max_vis :
            matches[next_inds[i]] = prev_inds[i]
        else :
            matches[next_inds[i]] = -1
    return matches
