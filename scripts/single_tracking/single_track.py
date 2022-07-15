import cv2
import numpy as np
import matplotlib.pyplot as plt
from Single_Tracker import Tracker
import pandas as pd
import argparse
from tqdm import tqdm

""""
Script for single-player tracking.
Usage:
python single_track.py INPUT_PATH OUTPUT_PATH [-p PLAYER_NAME]
"""

#ARGPARSING
parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, help='input video path')
parser.add_argument('output', type=str, help='output video path')
parser.add_argument('-p', type=str, help="player's name", default='player')
args = parser.parse_args()

vid = cv2.VideoCapture(args.input)
#GET METADATA
frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
fps = vid.get(cv2.CAP_PROP_FPS)
frame = vid.read()[1]
shape = (frame.shape[1], frame.shape[0])
_, frame = vid.read()

#VIDEOWRITER
output = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'XVID'), fps, shape)

tracker = Tracker(args.p)
tracker.calcHist(frame)

for i in tqdm(range(frame_count-1),
            desc='Stitching..', ascii=False, ncols=75):
    _, frame = vid.read()
    track_window , back_proj = tracker.apply(frame, register=i%10==0)
    x,y,w,h = track_window
    cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
    cv2.imshow('Match', frame)
    output.write(frame)
    k = cv2.waitKey(1)
    if k == 69 or k == 101:
        tracker.calcHist(frame)
    if k == 27: break

tracker.save()
vid.release()
output.release()
