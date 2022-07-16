# Soccer-Players-Detection-and-Tracking
the purpose of this application is to apply deep learning model pretrained on soccer players to detect only field soccer players and track them during the game.

<hr>

![ezgif com-gif-maker](https://drive.google.com/file/d/1YRaa89wkuPItIhEvSVgLi15FOmiAIlXv/view?usp=sharing)


## Prerequisites
```
PyTorch >= 1
CUDA >= 11
Torchvision >= 0.1.0
OpenCV >= 4.0.0
Pillow >= 7.1.0
scipy >= 1.7.0
motmetrics
``` 
<hr>


## Checkpoints Preparation

You can download all checkpoints from this [Drive](https://drive.google.com/drive/folders/1L2CSvFteLeZD6vdDem2ghCFGtAmzgslo?fbclid=IwAR22xmtNc-DA-SMzwyJ9iRtnmavv7kfqo3ocWz11coHt93Z4E4_VCoQutgY)

```
Soccer-Players-Detection-and-Tracking
├── checkpoints_runs
│   ├── player_det_resnet18_student.path
│   ├── player_det_resnet18_student.path
├── scripts
```
<hr>

## Testing 

**Download video**
You can download invideo.mp4 from this [Drive](https://drive.google.com/file/d/1-cxoZq6cBC6irxhHCqEZIGzXqrjSz-jr/view?usp=sharing)
You can download invideo.mp4 from this [Drive](https://drive.google.com/file/d/1YRaa89wkuPItIhEvSVgLi15FOmiAIlXv/view?usp=sharing)

```
Soccer-Players-Detection-and-Tracking
├── checkpoints_runs
├── scripts
│   ├──detection
│   │  ├──data
│   │  │  ├── invideo.mp4
│   │  │  ├── euro.mp4 
```
<hr>

**Player detection**


The script eval_video.py enables to get the detection result of the model, to do so put your test video in scripts/detection/data/<your_video.mp4>, '--birdeye' to get the birdeye view, '--classify' to apply classification. out video will be saved in the folder 'script/detection/results/out'.
'--classify' and 'birdeye' is valid only on our video from stitching 'invideo.mp4'
```
cd scripts/detection
python eval_video.py --invideo <video_name.mp4> --birdeye --classify --backbone resnet18 --checkpoint ../../checkpoints_runs/player_det_resnet50_teacher.pth

```
output video will be in 'birdEye.mp4' if '--birdeye' and in 'frontEye.mp4' by default. Also, you can specify the output name using '--birdout' and '--frontout'.
**Player tracking**

The code for tracking is based on the [Extending IOU Based Multi-Object Tracking by Visual Information]

* you can test your own video using the command below or using our videos above.

```
cd scripts/tracking
python main_tracking.py --visualize --input_path <your_video.mp4> --output_path <destination_folder>
```

output video will be 'result/output_tracking/out.mp4' by default or you can specify your own out folder path using '--output_path' by providing the folder without the file name.
**Single player tracking**

* you can test your own video using the command below or using our videos above.

```
cd scripts/single_tracking
python single_track.py INPUT_VIDEO_PATH OUTPUT_VIDEO_PATH [-p PLAYER_NAME]
```
Provide it with the input video path, the output video path and optionally the player’s name.

Then when the video finishes it will save three files, a JSON file containing the recorded 2D positions of the tracked player, an image showing the heatmap, and the output video showing the bounding box of the tracked player across all frames.

If the tracker loses the player press the key ‘e’ to reselect the player. And to exit the tracking press the ESC key.
