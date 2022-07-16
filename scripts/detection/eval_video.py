from tkinter.messagebox import NO
from xmlrpc.client import boolean
import cv2
import os
import numpy as np
import torch
import time
import argparse
from eval_fasterRCNN import get_model_detection, eval_image
from TopView import PerspectiveTransformation
from Detector import Detector


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='frcnn_fpn')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--detection_score_thres', type=float, default=0.9)
    parser.add_argument('--eval_original', dest='eval_original', action='store_true')
    parser.set_defaults(eval_original=False)
    parser.add_argument('--n_channel_backbone', type=int, default=5)
    parser.add_argument('--min_anchor', type=int, default=16)
    parser.add_argument('--no_use_soft_nms', dest='use_soft_nms', action='store_false')
    parser.set_defaults(use_soft_nms=True)
    parser.add_argument('--no_use_context', dest='use_context', action='store_false')
    parser.set_defaults(use_context=True)
    parser.add_argument('--use_track_branch', dest='use_track_branch', action='store_true')
    parser.set_defaults(use_track_branch=False)
    parser.add_argument('--birdfield', type=str, default='topfield.jpg')
    parser.add_argument('--birdout', type=str, default='birdEye.mp4')
    parser.add_argument('--frontout', type=str, default='frontEye.mp4')
    parser.add_argument('--invideo', type=str, default='invideo.mp4')
    parser.add_argument('--birdeye', dest='birdeye', action='store_true')
    parser.set_defaults(birdeye=False)
    parser.add_argument('--classify', dest='classify', action='store_true')
    parser.set_defaults(classify=False)
    parser.add_argument('--fps', type=float, default=29.0)
    args = parser.parse_args()
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2
    
    res_path = 'results/'
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    out_path = res_path + 'out/'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    data_path = 'data/'
    if not os.path.exists(data_path):
        os.mkdir(data_path)

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

    #videocapture
    cap = cv2.VideoCapture(data_path + args.invideo)
    ret, frame = cap.read()
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    #cap_front = cv2.VideoWriter(out_path + args.frontout, fourcc, args.fps, (4184,630))
    cap_front = cv2.VideoWriter(out_path + args.frontout, fourcc, args.fps, (frame.shape[1], frame.shape[0]))

    if args.birdeye:
        op = PerspectiveTransformation()
        topfield = cv2.imread(args.birdfield)
        fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        cap_bird = cv2.VideoWriter(out_path + args.birdout, fourcc, args.fps, (1500, 1000))
    else:
        topfield = None
        op = None

    t1 = time.time()
    if args.classify:
        classifier = Detector()
    else:
        classifier = None
    
    while True:
        topfield_copy = np.copy(topfield)
        ret, frame = cap.read()
        if ret:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            front_out, bird_out = eval_image(frame, image, model, device, args.birdeye, topfield_copy, op, classifier)
            front_out = np.uint8(front_out)
            cap_front.write(front_out)

            if args.birdeye:
                bird_out = np.uint8(bird_out)
                cap_bird.write(bird_out)
        else:
            break


    t2 = time.time()
    print('evalutation time (sec) : ', str(int(t2 - t1)))
    cap.release()
    cap_front.release()
