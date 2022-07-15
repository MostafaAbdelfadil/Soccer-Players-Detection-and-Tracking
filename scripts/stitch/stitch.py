from pickle import FALSE
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from stitcherClass import Stitcher

"""
Script for video stitching.
Usage:
python stitch.py LEFT_VIDEO_PATH MID_VIDEO_PATH [--third RIGHT_VIDEO_PATH] OUTPUT_PATH
"""

#ARGPARSING
parser = argparse.ArgumentParser()
parser.add_argument('first', type=str, help='first video path')
parser.add_argument('second', type=str, help='second video path')
parser.add_argument('--third', type=str, help='third video path')
parser.add_argument('output', type=str, help='output path')
args = parser.parse_args()

#LOADING VIDS
vids = [cv2.VideoCapture(args.first), cv2.VideoCapture(args.second)]
if args.third:
    vids.append(cv2.VideoCapture(args.third))

for vid in vids:
    if not vid.isOpened():
        print('error loading videos')

#GET METADATA
frame_count = int(vids[0].get(cv2.CAP_PROP_FRAME_COUNT)) - 1
fps = vids[0].get(cv2.CAP_PROP_FPS)
imageStitcher = Stitcher()
frames = [vid.read()[1] for vid in vids]
stitchedImage = imageStitcher.stitch(frames)
height = stitchedImage.shape[0]
width = stitchedImage.shape[1]
shape = (width, height)

#VIDEOWRITER
output = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'XVID'), fps, shape)
output.write(stitchedImage)


#STITCHING
for i in tqdm(range(frame_count-1),
            desc='Stitching..', ascii=False, ncols=75):
            frames = [vid.read()[1] for vid in vids]
            stitchedImage = imageStitcher.stitch(frames)
            output.write(stitchedImage)

#RELEASE
for vid in vids:
    vid.release()
output.release()