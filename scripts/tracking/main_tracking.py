import sys
import torchvision
import os
import torch
from tracking_utils import track
from natsort import natsorted, ns
import numpy as np
from argparse import ArgumentParser
sys.path.append('../detection')
from faster_rcnn import FastRCNNPredictor
from faster_rcnn import fasterrcnn_resnet18_fpn, fasterrcnn_resnet50_fpn, fasterrcnn_detnet59_fpn, fasterrcnn_resnet101_fpn, fasterrcnn_resnet34_fpn
def get_model_detection(model_name, weight_loss, backbone, pretrained,
    pretrained_backbone, original, detection_score_thres, use_non_player,
    use_soft_nms, anchor_sizes=[32,64,128,256,512], n_channel_backbone = 5,
    use_context=False, nms_thres = 0.4, use_track_branch = False):

        if use_non_player :
            num_classes = 3
        else :
            num_classes = 2

        if model_name == 'frcnn_fpn' :
            if backbone == 'resnet18' :
                model = fasterrcnn_resnet18_fpn(weight_loss=weight_loss,
                pretrained_backbone = pretrained_backbone, num_classes = 2,
                detection_score_thres = detection_score_thres, anchor_sizes = anchor_sizes,
                n_channel_backbone = n_channel_backbone, box_nms_thresh = nms_thres, use_soft_nms = use_soft_nms,
                use_context = use_context, use_track_branch = use_track_branch)
            if backbone == 'resnet50' :
                model = fasterrcnn_resnet50_fpn(weight_loss=weight_loss,
                pretrained = pretrained, pretrained_backbone = pretrained_backbone,
                num_classes=2, anchor_sizes = anchor_sizes, detection_score_thres = detection_score_thres,
                n_channel_backbone = n_channel_backbone, box_nms_thresh = nms_thres, use_soft_nms = use_soft_nms,
                use_context = use_context, use_track_branch = use_track_branch)
            if not original :
                in_features = model.roi_heads.box_predictor.cls_score.in_features
                model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        else :
            'model not available'
        return model

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--data_name', type=str, default='issia')
    parser.add_argument('--use_GT_position', dest='use_GT_position', action='store_true')
    parser.set_defaults(use_GT_position=False)
    parser.add_argument('--rescale_img_factor', type=float, default=1.0)
    parser.add_argument('--model_name', type=str, default='frcnn_fpn')
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--checkpoint', type=str, default='../../checkpoints_runs/player_det_resnet18_student.pth')
    parser.add_argument('--detection_score_thres', type=float, default=0.8)
    parser.add_argument('--no_use_context', dest='use_context', action='store_false')
    parser.set_defaults(use_context=True)
    parser.add_argument('--no_use_soft_nms', dest='use_soft_nms', action='store_false')
    parser.set_defaults(use_soft_nms=True)
    parser.add_argument('--nms_thres', type=float, default=0.4)
    parser.add_argument('--anchor_sizes', type=int, nargs='+', default=[32, 64, 128, 256, 512])
    parser.add_argument('--use_track_branch_model', dest='use_track_branch_model', action='store_true')
    parser.set_defaults(use_track_branch_model=False)
    parser.add_argument('--use_track_branch_embed', dest='use_track_branch_embed', action='store_true')
    parser.set_defaults(use_track_branch_embed=False)
    parser.add_argument('--pose_model', type=str, default='mobile-deconv')
    parser.add_argument('--keyframe_interval', type=int, default=1)
    parser.add_argument('--no_use_IOU', dest='use_IOU', action='store_false')
    parser.set_defaults(use_IOU=True)
    parser.add_argument('--spacial_iou_thresh', type=float, default=0.5)
    parser.add_argument('--no_use_features', dest='use_features', action='store_false')
    parser.set_defaults(use_features=True)
    parser.add_argument('--no_use_visual_feat', dest='use_visual_feat', action='store_false')
    parser.set_defaults(use_visual_feat=True)
    parser.add_argument('--visual_feat_model_name', type=str, default='faster-rcnn')
    parser.add_argument('--imagenet_model', dest='imagenet_model', action='store_false')
    parser.set_defaults(imagenet_model=True)
    parser.add_argument('--weight_loss', dest='weight_loss', action='store_true')
    parser.set_defaults(weight_loss=False)
    parser.add_argument('--w_spacial', type=float, default=0.97)
    parser.add_argument('--w_visual', type=float, default=0.03)
    parser.add_argument('--visual_metric', type=str, default='l2')
    parser.add_argument('--use_filter_tracks', dest='use_filter_tracks', action='store_true')
    parser.set_defaults(use_filter_tracks=False)
    parser.add_argument('--thres_count_ids', type=int, default=2)
    parser.add_argument('--use_ReID_module', dest='use_ReID_module', action='store_true')
    parser.set_defaults(use_ReID_module=False)
    parser.add_argument('--max_vis_reID', type=int, default=4)
    parser.add_argument('--max_vis_feat', type=int, default=4)
    parser.add_argument('--N_past_to_keep_reID', type=int, default=3)
    parser.add_argument('--N_past_to_keep', type=int, default=1)
    parser.add_argument('--N_frame_lost_keep', type=int, default=10)
    parser.add_argument('--visualize', dest='visualize', action='store_true')
    parser.set_defaults(visualize=False)
    parser.add_argument('--output_path', type=str, default='./')
    parser.add_argument('--input_path', type=str, default='./test.mp4')
    hparams = parser.parse_args()

    hparams.current_model_detection = None
    hparams.flag_method = True
    if not hparams.use_visual_feat:
        hparams.w_visual = 0
    
    if hparams.visual_feat_model_name == 'faster-rcnn':
        hparams.imagenet_model = False

    max_dist_factor_feat = 32 * (1 / hparams.rescale_img_factor)
    max_dist_factor_reID = max_dist_factor_feat / 4
    if not hparams.use_GT_position:
        if hparams.current_model_detection is None:
            #from train_tracker import get_model_detection
            model_detection = get_model_detection(hparams.model_name, hparams.weight_loss, hparams.backbone, False,
                                                  False, False, hparams.detection_score_thres, False,
                                                  hparams.use_soft_nms, anchor_sizes=hparams.anchor_sizes, use_context=hparams.use_context,
                                                  nms_thres=hparams.nms_thres, use_track_branch=hparams.use_track_branch_model)
            model_detection.load_state_dict(torch.load(hparams.checkpoint))
            model_detection.to(torch.device('cuda'))
            model_detection.eval()
        else:
            model_detection = hparams.current_model_detection
    else:
        model_detection = None

    if hparams.use_visual_feat:
        if hparams.visual_feat_model_name == 'faster-rcnn':
            if hparams.current_model_detection is None:
               # from train_tracker import get_model_detection
                visual_feat_model = get_model_detection(hparams.model_name, hparams.weight_loss, hparams.backbone, False,
                                                        False, False, hparams.detection_score_thres, False,
                                                        hparams.use_soft_nms, anchor_sizes=hparams.anchor_sizes,
                                                        use_context=hparams.use_context, nms_thres=hparams.nms_thres,
                                                        use_track_branch=hparams.use_track_branch_model)
                visual_feat_model.load_state_dict(torch.load(hparams.checkpoint))
                visual_feat_model.to(torch.device('cuda'))
            else:
                visual_feat_model = hparams.current_model_detection
            visual_feat_model.eval()
            layer = visual_feat_model._modules.get('fc7')

        elif hparams.visual_feat_model_name == 'resnet50':
            visual_feat_model = torchvision.models.resnet50(pretrained=True)
            visual_feat_model.to(torch.device('cuda'))
            visual_feat_model.eval()
            layer = visual_feat_model._modules.get('avgpool')
        elif hparams.visual_feat_model_name == 'vgg19':
            visual_feat_model = torchvision.models.vgg19(pretrained=True)
            visual_feat_model.to(torch.device('cuda'))
            visual_feat_model.eval()
            layer = visual_feat_model._modules.get('avgpool')
        else:
            print(' visual feature model does not exist')
            use_visual_feat = False
    else:
        visual_feat_model = None
        layer = None
    base_dir = hparams.output_path + '/result'
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    output_folder = os.path.join(base_dir, 'output_tracking')
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    output_video_path = os.path.join(output_folder, "out.mp4")

    input_path=hparams.input_path
    out =track( model_detection, visual_feat_model, layer,
                           hparams.rescale_img_factor,
                          input_path,output_video_path, hparams.use_features,
                          hparams.w_spacial, hparams.w_visual, hparams.use_IOU, hparams.spacial_iou_thresh,
                          hparams.detection_score_thres,  hparams.use_visual_feat, hparams.imagenet_model,
                           hparams.use_GT_position, hparams.flag_method,
                          hparams.keyframe_interval, hparams.visualize,
                          hparams.use_filter_tracks, hparams.thres_count_ids, hparams.visual_metric,
                          hparams.N_frame_lost_keep, hparams.N_past_to_keep, hparams.use_ReID_module,
                          hparams.N_past_to_keep_reID,hparams.max_vis_feat, max_dist_factor_feat, hparams.max_vis_reID,
                          max_dist_factor_reID,
                          hparams.use_track_branch_embed)








