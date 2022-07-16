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

**Download data**

**Player detection**


The script eval_video.py enables to get the detection result of the model, to do so put your test video in scripts/detection/data/<your_video.mp4>, '--birdeye' to get the birdeye view, '--classify' to apply classification. Images will be saved in the folder 'script/detection/results/out'.
```
cd scripts/detection
python eval_video.py --birdeye --classify --backbone resnet18 --checkpoint ../../checkpoints_runs/player_det_resnet50_teacher.pth

```

**Player tracking**

The code for tracking is based on the [LightTrack](https://github.com/Guanghan/lighttrack) code. 

* First clone the [LightTrack](https://github.com/Guanghan/lighttrack) repository in 'script/other_utils' 
* Change the visualizer code of the LightTrack code with the visualizer folder given in 'my_utils' : 

```
cd script
cp -r my_utils/visualizer other_utils/lighttrack/
```

* Realize tracking on the dataset of your choice. Only the ISSIA evaluation dataset contains tracking ground-truth information. 
Use the argument --use_GT_position is order to realize tracking on ground-truth player position data. Without this flag, the code will use the detection model decribed above.

```
python main_tracking --data_name issia --visualize --write_video --output_path ../../results
```

