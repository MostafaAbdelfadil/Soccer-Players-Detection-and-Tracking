# Soccer-Players-Detection-and-Tracking
the purpose of this application is to apply deep learning model pretrained on soccer players to detect only field soccer players and track them during the game.

## Prerequisites
```
PyTorch >= 1
CUDA >= 11
Torchvision >= 0.1.0
OpenCV >= 4.0.0
Pillow >= 7.1.0
scipy >= 1.7.0
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
'--classify' is valid only on our video from stitching 'invideo.mp4'
```
cd scripts/detection
python eval_video.py --birdeye --classify --backbone resnet18 --checkpoint ../../checkpoints_runs/player_det_resnet50_teacher.pth

```

**Player tracking**

The code for tracking is based on the [Extending IOU Based Multi-Object Tracking by Visual Information]

* you can test your own video using the command below or using our videos above.

```
cd scripts/tracking
python main_tracking --visualize --input_path <your_video.mp4>
```

