import sys
from tkinter import Frame
sys.path.append('../other_utils/metrics/')
from pascalvoc import compute_metrics
import cv2
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import torchvision.utils
from faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from RPN import RPN
from faster_rcnn import FasterRCNN, fasterrcnn_resnet18_fpn, fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator
from soccer_dataset import SoccerDataset
import torch
import utils
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import shutil
import numpy as np
from PIL import Image
import time
import pathlib

def enlargeBbox(bbox, scale, image_shape):
    min_x, min_y, max_x, max_y = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()
    margin_x = int(0.5 * scale[0] * (max_x - min_x))
    margin_y = int(0.5 * scale[1] * (max_y - min_y))
    min_x -= margin_x
    max_x += margin_x
    min_y -= margin_y
    max_y += margin_y
    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(image_shape[2], max_x)
    max_y = min(image_shape[1], max_y)
    bbox_enlarged = [min_x, min_y, max_x, max_y]
    return bbox_enlarged


def get_model_detection(model_name, weight_loss, backbone_name, pretrained,
                        pretrained_backbone, original, detection_score_thres, use_soft_nms,
                        anchor_sizes=[32, 64, 128, 256, 512], n_channel_backbone=5, use_context=False,
                        nms_thres=0.4, use_track_branch=False):

    num_classes = 2

    if model_name == 'frcnn_fpn':
        if backbone_name == 'resnet18':
            model = fasterrcnn_resnet18_fpn(weight_loss=weight_loss,
                                            pretrained_backbone=pretrained_backbone, num_classes=2,
                                            detection_score_thres=detection_score_thres, anchor_sizes=anchor_sizes,
                                            n_channel_backbone=n_channel_backbone, use_soft_nms=use_soft_nms,
                                            nms_thres=nms_thres,
                                            use_context=use_context,
                                            use_track_branch=use_track_branch)
        if backbone_name == 'resnet50':
            model = fasterrcnn_resnet50_fpn(weight_loss=weight_loss,
                                            pretrained=pretrained, pretrained_backbone=pretrained_backbone,
                                            num_classes=2, anchor_sizes=anchor_sizes,
                                            detection_score_thres=detection_score_thres,
                                            n_channel_backbone=n_channel_backbone, use_soft_nms=use_soft_nms,
                                            nms_thres=nms_thres,
                                            use_context=use_context, use_track_branch=use_track_branch)
        if not original:
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    else:
        'model not available'

    return model


def get_transform():
    transforms = [T.ToTensor()]
    return T.Compose(transforms)

def NMS(boxes, overlapThresh = 0.6):
    # Return an empty list, if no boxes given
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]  # x coordinate of the top-left corner
    y1 = boxes[:, 1]  # y coordinate of the top-left corner
    x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
    y2 = boxes[:, 3]  # y coordinate of the bottom-right corner
    # Compute the area of the bounding boxes and sort the bounding
    # Boxes by the bottom-right y-coordinate of the bounding box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) # We add 1, because the pixel at the start as well as at the end counts
    # The indices of all boxes at start. We will redundant indices one by one.
    indices = np.arange(len(x1))
    for i,box in enumerate(boxes):
        # Create temporary indices  
        temp_indices = indices[indices!=i]
        # Find out the coordinates of the intersection box
        xx1 = np.maximum(box[0], boxes[temp_indices,0])
        yy1 = np.maximum(box[1], boxes[temp_indices,1])
        xx2 = np.minimum(box[2], boxes[temp_indices,2])
        yy2 = np.minimum(box[3], boxes[temp_indices,3])
        # Find out the width and the height of the intersection box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / areas[temp_indices]
        # if the actual boungding box has an overlap bigger than treshold with any other box, remove it's index  
        if np.any(overlap) > overlapThresh:
            indices = indices[indices != i]
    #return only the boxes at the remaining indices
    return boxes[indices].astype(int)

def eval_image(frame, image, model, device, birdeye=False, topfield=None, op=None, detector=None):
    model.eval()
    with torch.no_grad():
        trans = T.ToTensor()
        image = trans(image)
        image = image.to(device)
        image = [image]
        detection = model(image)

        if detector is not None:
            bboxes = detection[0][0]['boxes'].detach().cpu().numpy()
            #print(bboxes.shape)
            #bboxes = NMS(bboxes)
            #print(bboxes.shape)
            #print()
            teams = detector.getTeams(frame, bboxes)
            referee = [teams['referee']]
            masry = teams['masry']
            pyramids = teams['pyramids']
            for i in referee:
                cv2.rectangle(frame,
                            (int(i[0]), int(i[1]), int(i[2] - i[0]), int(i[3] - i[1])),
                            (0, 0, 255), 2)
                if birdeye:
                        w = float((i[2] - i[0])/2)
                        point = np.array([[float((i[2] - w)), float(i[3])]], np.float32).reshape(1, -1, 2)
                        point = cv2.perspectiveTransform(point, op.M)
                        point = np.int32(point)
                        point = (point[0][0][0], point[0][0][1])
                        cv2.circle(topfield, point, 10, (0, 0, 255), -1)
            
            for i in masry:
                cv2.rectangle(frame,
                            (int(i[0]), int(i[1]), int(i[2] - i[0]), int(i[3] - i[1])),
                            (255, 0, 0), 2)
                if birdeye:
                        w = float((i[2] - i[0])/2)
                        point = np.array([[float((i[2] - w)), float(i[3])]], np.float32).reshape(1, -1, 2)
                        point = cv2.perspectiveTransform(point, op.M)
                        point = np.int32(point)
                        point = (point[0][0][0], point[0][0][1])
                        cv2.circle(topfield, point, 10, (255, 0, 0), -1)
            
            for i in pyramids:
                cv2.rectangle(frame,
                            (int(i[0]), int(i[1]), int(i[2] - i[0]), int(i[3] - i[1])),
                            (0, 0, 0), 2)
                if birdeye:
                        w = float((i[2] - i[0])/2)
                        point = np.array([[float((i[2] - w)), float(i[3])]], np.float32).reshape(1, -1, 2)
                        point = cv2.perspectiveTransform(point, op.M)
                        point = np.int32(point)
                        point = (point[0][0][0], point[0][0][1])
                        cv2.circle(topfield, point, 10, (0, 0, 0), -1)

            return frame, topfield
            
        else:
            show = cv2.cvtColor(np.array(image[0].cpu()).transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
            for d, bbox in enumerate(detection[0][0]['boxes']):
                    cv2.rectangle(show,
                                (int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])),
                                (255, 255, 255), 1)

                    if birdeye:
                        w = float((bbox[2] - bbox[0])/2)
                        point = np.array([[float((bbox[2] - w)), float(bbox[3])]], np.float32).reshape(1, -1, 2)
                        point = cv2.perspectiveTransform(point, op.M)
                        point = np.int32(point)
                        point = (point[0][0][0], point[0][0][1])
                        cv2.circle(topfield, point, 10, (0, 0, 255), -1)
            show = show*255
            return show, topfield

def eval(model, test_loader, n_test, test_bs, device, n_update, original_model,
         writer=None, first_iter=0, get_image=False,show_GT_label=False):
    
    rescale_bbox = [0., 0.]

    parts = list(pathlib.Path().absolute().parts)[:-2]
    parts.append('data')
    data_path = pathlib.Path(*parts)
    intermediate_path = os.path.join(data_path, 'intermediate', 'fasterRCNN_test')
    if not os.path.exists(intermediate_path):
        os.mkdir(intermediate_path)

    gt_txt_path = os.path.join(intermediate_path, 'GT')
    det_txt_path = os.path.join(intermediate_path, 'det')
    if os.path.exists(det_txt_path):
        shutil.rmtree(det_txt_path)
    os.mkdir(det_txt_path)
    if os.path.exists(gt_txt_path):
        shutil.rmtree(gt_txt_path)
    os.mkdir(gt_txt_path)

    test_loader_iterator = iter(test_loader)
    n_iter = min(n_test // test_bs, len(test_loader))
    model.eval()
    
    annotation_list = []
    detection_list = []
    scores_list = []
    images_list=[]

    time_list = []

    with torch.no_grad():
        
        for i in range(n_iter):

            for j in range(first_iter + 1):
                images, targets = next(test_loader_iterator)
            
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            t1 = time.time()
            detections = model(images)
            t2 = time.time()
            time_list.append(t2 - t1)

            if original_model:
                for j in range(len(images)):
                    annotation = targets[j]["boxes"].to(torch.device("cpu"))
                    detection = []
                    score = []
                    detection_all = detections[0][j]['boxes'].to(torch.device("cpu"))
                    label_all = detections[0][j]['labels'].to(torch.device("cpu"))
                    score_all = detections[0][j]['scores'].to(torch.device("cpu"))
                    for d, det in enumerate(detection_all):
                        if label_all[d] == 1:
                            detection.append(detection_all[d])
                            score.append(score_all[d])
                    annotation_list.append(annotation)
                    detection_list.append(detection)
                    scores_list.append(score)
                    images_list.append(images[j])

            else:
                for j in range(len(images)):
                    annotation = targets[j]["boxes"].to(torch.device("cpu"))
                    detection = detections[0][j]['boxes'].to(torch.device("cpu"))
                    score = detections[0][j]['scores'].to(torch.device("cpu"))
                    annotation_list.append(annotation)
                    detection_list.append(detection)
                    scores_list.append(score)
                    images_list.append(images[j])

            image_shape = images[0].shape
        
        
        

    for i in range(len(annotation_list)):
        with open(os.path.join(gt_txt_path, str(i) + '.txt'), 'w') as GT_file:
            with open(os.path.join(det_txt_path, str(i) + '.txt'), 'w') as det_file:
                annotation_boxes = annotation_list[i]
                det_boxes = detection_list[i]
                det_scores = scores_list[i]
                for bbox in annotation_boxes:
                    bbox = np.array(enlargeBbox(bbox, rescale_bbox, image_shape))
                    GT_file.write('person ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + ' ' +
                                  str(bbox[3]) + '\n')
                for j, bbox in enumerate(det_boxes):
                    det_score = det_scores[j].item()
                    det_file.write('person ' + str(det_score) + ' ' + str(bbox[0].item()) + ' ' + str(bbox[1].item()) +
                                   ' ' + str(bbox[2].item()) + ' ' + str(bbox[3].item()) + '\n')

    currentPath = os.path.dirname(os.path.abspath(__file__))
    mAP = compute_metrics(gt_txt_path, det_txt_path, iouThreshold=0.5, showPlot=False)
    os.chdir(currentPath)
    if writer is not None or get_image:
        show_images = [cv2.cvtColor(np.array(image.cpu()).transpose(1, 2, 0), cv2.COLOR_RGB2BGR) for image in images_list]
        out_tensor = []
        out_array = []
        for j in range(len(show_images)):
            detection = detection_list[j]
            #label = detections[0][j]['labels'].to(torch.device("cpu"))
            annotation_boxes = annotation_list[j]
            if show_GT_label :
                for d, bbox in enumerate(annotation_boxes):
                    bbox = np.array(enlargeBbox(bbox, rescale_bbox, image_shape))
                    cv2.rectangle(show_images[j],
                                (int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])),
                                (0, 0, 255), 2)
            for d, bbox in enumerate(detection):
                #if label[d] == 1:
                cv2.rectangle(show_images[j],
                            (int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])),
                            (255, 255, 255), 2)
            out_array.append(show_images[j])
            if writer is not None:
                shape = show_images[j].shape
                new_shape = (int(shape[1] / 4.), int(shape[0] / 4.))
                show_image = cv2.resize(show_images[j], dsize=new_shape, interpolation=cv2.INTER_CUBIC)
                show_image = cv2.cvtColor(show_image, cv2.COLOR_BGR2RGB)
                show_image = Image.fromarray((show_image * 255).astype(np.uint8))
                out_tensor.append(T.ToTensor()(show_image))
        if writer is not None:
            img_grid = torchvision.utils.make_grid(out_tensor)
            writer.add_image('val/imgs', img_grid, n_update)

    if get_image:
        return mAP, out_array
    else:
        return mAP, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='test', help='name of the experiment')
    parser.add_argument('--test_dataset_name', type=str, default='TV_soccer')
    parser.add_argument('--test_bs', type=int, default=4)
    parser.add_argument('--n_test', type=int, default=400)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--model_name', type=str, default='frcnn_fpn')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--scale_transform_test', type=float, default=1.)
    parser.add_argument('--detection_score_thres', type=float, default=0.9)
    parser.add_argument('--eval_original', dest='eval_original', action='store_true')
    parser.set_defaults(eval_original=False)
    parser.add_argument('--n_channel_backbone', type=int, default=5)
    parser.add_argument('--min_anchor', type=int, default=16)
    parser.add_argument('--no_use_soft_nms', dest='use_soft_nms', action='store_false')
    parser.set_defaults(use_soft_nms=True)
    parser.add_argument('--no_use_context', dest='use_context', action='store_false')
    parser.set_defaults(use_context=True)
    parser.add_argument('--use_field_detection', dest='use_field_detection', action='store_true')
    parser.set_defaults(use_field_detection=False)
    parser.add_argument('--use_track_branch', dest='use_track_branch', action='store_true')
    parser.set_defaults(use_track_branch=False)
    parser.add_argument('--save_visualization', dest='save_visualization', action='store_true')
    parser.set_defaults(save_visualization=False)
    parser.add_argument('--show_GT_label', dest='show_GT_label', action='store_true')
    parser.set_defaults(show_GT_label=False)
    args = parser.parse_args()
    args.detection_score_thres
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2

    parts = list(pathlib.Path().absolute().parts)[:-2]
    parts.append('data')
    data_path = pathlib.Path(*parts)

    path = os.path.join(data_path, args.test_dataset_name)
    test_image_files = os.path.join(path, 'test_frame_list.txt')
    test_annotation_files = os.path.join(path, 'test_annotation_list.txt')
    
    dataset_test = SoccerDataset(
        test_image_files=test_image_files,
        test_annotation_files=test_annotation_files,
        transform=get_transform(),
        train=False,
        test_dataset_name=args.test_dataset_name,
        scale_transform=args.scale_transform_test,
        use_field_detection=args.use_field_detection)

    test_loader = DataLoader(
        dataset_test, batch_size=args.test_bs, shuffle=False, num_workers=args.num_workers,
        collate_fn=utils.collate_fn)
    
    writer = SummaryWriter("runs/" + args.name)

    if args.min_anchor == 8:
        anchor_sizes = [8, 16, 32, 64, 128]
    elif args.min_anchor == 16:
        anchor_sizes = [16, 32, 64, 128, 256]
    else:
        anchor_sizes = [32, 64, 128, 256, 512]

    model = get_model_detection(args.model_name, False, args.backbone,
                                False, False, False, args.detection_score_thres,
                                args.use_soft_nms, anchor_sizes, args.n_channel_backbone,
                                args.use_context, use_track_branch=args.use_track_branch)
    model.to(device)
    
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))

    if args.eval_original:
        model = get_model_detection(args.model_name, False, args.backbone,
                                    True, True, True, args.detection_score_thres, args.use_soft_nms)
        model.to(device)

    
    t1 = time.time()
    current_mAP, images = eval(model, test_loader, args.n_test, args.test_bs,
                    device, 0, args.eval_original, get_image=args.save_visualization,
                            show_GT_label = args.show_GT_label)
    t2 = time.time()
    print('evalutation time (sec) : ', str(int(t2 - t1)))
    dir_path = os.path.dirname(os.path.realpath(__file__))
    vis_path = 'results'
    os.path.join(dir_path,vis_path)
    if not os.path.exists(vis_path):
        os.mkdir(vis_path)
    vis_path = os.path.join(vis_path,args.name)
    if not os.path.exists(vis_path):
        os.mkdir(vis_path)
    vis_path = os.path.join(vis_path,args.test_dataset_name)
    if not os.path.exists(vis_path):
        os.mkdir(vis_path)
    for i,im in enumerate(images):
        cv2.imwrite(os.path.join(vis_path,str(i)+'.png'),im*255)