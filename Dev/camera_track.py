from __future__ import division, print_function, absolute_import
import os, sys, cv2, time, warnings, random
import numpy as np
import os.path as osp
from copy import deepcopy
from base_camera import BaseCamera
import torch.nn.functional as F

import torch, torchvision
import torch.nn as nn
from torch2trt import TRTModule

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer

from fastreid.engine import DefaultPredictor
from fastreid.config import get_cfg
from fastreid.modeling.meta_arch import build_model
from fastreid.utils.checkpoint import Checkpointer

warnings.filterwarnings('ignore')

def cfg_setting(args, gpu_number):
    cfg = get_cfg()
    cfg.merge_from_file(args.configfile)
    # if args.trt:
    #     cfg.MODEL.WEIGHTS = args.reid_trtfile
    # else:
    #     cfg.MODEL.WEIGHTS = args.reid_weightfile
    cfg.MODEL.WEIGHTS = args.reid_weightfile
    cfg.MODEL.DEVICE = f'cuda:{gpu_number}'
    cfg.freeze()
    return cfg

class FeatureExtracter(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = DefaultPredictor(cfg)

    def inference(self, batched_image, batch_size=None):
        if batch_size is None:
            predictions = self.model(batched_image)
        else:
            predictions = self.model(batched_image)

        return predictions

class TRTFeatureExtracter(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = TRTModule()
        self.model.load_state_dict(torch.load(cfg.MODEL.WEIGHTS))

    def inference(self, batched_image, batch_size=64):
        if batched_image.shape[0] > batch_size:
            predictions = []
            complete_batch = batched_image.shape[0]//batch_size
            remain = batched_image.shape[0] % batch_size
            for i in range(complete_batch):
                predictions.append(self.model(batched_image[i*batch_size: i*batch_size+batch_size]))
            predictions.append(self.model(batched_image[batch_size*complete_batch: batch_size*complete_batch+remain]))
            predictions = torch.cat(predictions, dim=0)
        else:
            predictions = self.model(batched_image)

        return predictions

class Camera(BaseCamera):
    def __init__(self, feed_type, device_number, gpu_number, args):
        unique_name = (feed_type, device_number)
        BaseCamera.cfg = cfg_setting(args, gpu_number)
        BaseCamera.timers[unique_name] = [Timer(), Timer()]

        ### region mask ###
        if args.mask:
            mask_npy = np.load(f'./region_mask/Cam{device_number}.npy')
        else:
            mask_npy = None
        
        ### create tracker
        tracker = BYTETracker(BaseCamera.yolo_exp, device_number=device_number, use_feat=True, frame_rate=60, region_mask=mask_npy)
        BaseCamera.trackers[unique_name] = tracker
        print(f'[Device {device_number}] Create Tracker.')

        # ### create extracter
        # if not args.trt:
        #     extractor = FeatureExtracter(BaseCamera.cfg)
        #     BaseCamera.extractors[unique_name] = extractor
        #     print(f'[Device {device_number}] Create Re-ID extractor on GPU{gpu_number}.')
        extractor = FeatureExtracter(BaseCamera.cfg)
        BaseCamera.extractors[unique_name] = extractor
        print(f'[Device {device_number}] Create Re-ID extractor on GPU{gpu_number}.')
        
        super(Camera, self).__init__(feed_type, device_number)

    @classmethod
    def track_frames(self, unique_name):
        device_number = unique_name[1]
        exp = BaseCamera.yolo_exp
        args = BaseCamera.args
        cfg = BaseCamera.cfg

        # if args.trt:
        #     reid_model = TRTFeatureExtracter(BaseCamera.cfg)
        #     print(f'[Device {device_number}] Create TRT-Extractor.')
        # else:        
        #     reid_model = BaseCamera.extractors[unique_name]
        reid_model = BaseCamera.extractors[unique_name]
        tracker = BaseCamera.trackers[unique_name]
        timer_reid, timer_tracker = BaseCamera.timers[unique_name]
        
        # wait for server_thread to create
        get_feed_from = ('yolo', device_number)
        while get_feed_from not in BaseCamera.frame: 
            time.sleep(0)

        num_process_frames = 1
        while True:
            frame, fid, boxes, _, duration = BaseCamera.get_frame(get_feed_from)
            if fid is None :
                yield None, None, None, None, None
                
            assert fid == num_process_frames
            num_process_frames += 1
            
            # add detections to batched images
            reid_imgs = []
            online_targets = []
            timer_reid.tic()
            if boxes.shape[0] != 0:
                for idx, box in enumerate(boxes):
                    x1,y1,x2,y2 = tuple(map(int, box[:4]))
                    tgt = cv2.resize(frame[y1:y2+1, x1:x2+1, :], (cfg.INPUT.SIZE_TEST[1], cfg.INPUT.SIZE_TEST[0]))
                    reid_imgs.append(tgt)

                # Forward those images to update the image features
                reid_imgs = torch.from_numpy(np.array(reid_imgs)).float().to(cfg.MODEL.DEVICE)
                reid_imgs = reid_imgs.permute(0,3,1,2)
                predictions = reid_model.inference(reid_imgs)
                predictions = F.normalize(predictions, dim=1).cpu()
                timer_reid.toc()
                # print(predictions.shape) # 1024-dim

                timer_tracker.tic()
                online_tracks = tracker.update(boxes, predictions, exp.test_size)
                # Choose good online targets
                for t in online_tracks:
                    _,_,w,h = t.tlwh
                    not_vertical = w / h > exp.asp_ratio_thres
                    if w * h > exp.min_box_area and not not_vertical:
                        online_targets.append(t) 
                        timer_tracker.toc()
            else:
                timer_reid.toc()
                timer_tracker.tic()
                tracker.update_with_no_boxes()
                timer_tracker.toc()
            
            if fid % args.print_interval == 0:
                print('[Device {}] Tracking Thread frame {:5d} ({:.2f} fps) | DET:{:.5f}, REID:{:.5f}, Track:{:.5f}, box_num:{}'
                .format(device_number, fid, 1./max(1e-5, timer_reid.average_time+timer_tracker.average_time+duration), duration, timer_reid.average_time, timer_tracker.average_time, boxes.shape[0]))
            
            
            if args.visualize:
                plot_tlwhs = [t.tlwh for t in online_targets]
                online_ids = [t.track_id for t in online_targets]
                online_scores = [t.score for t in online_targets]
                online_global_ids = ['' for t in online_targets]
                frame = plot_tracking(
                    frame, plot_tlwhs, online_ids, scores = online_scores,
                    frame_id = fid, global_IDs = online_global_ids)

            # print('#'*80)
            # print(np.array([t.tlwh for t in online_targets]))
            # print(np.array([t.mb for t in online_targets]))
            # print(np.array([t.track_id for t in online_targets]))


            yield frame, fid, online_targets, None, [duration, timer_reid.average_time, timer_tracker.average_time]
