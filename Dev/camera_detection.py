from __future__ import division, print_function, absolute_import
import os, cv2, time, warnings, errno
import os.path as osp
import numpy as np
from copy import deepcopy
from pathlib import Path

from base_camera import BaseCamera

import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from torch2trt import TRTModule

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer

warnings.filterwarnings('ignore')

class Predictor(object):
    def __init__(self, model, exp, device, args, fp16=False):
        self.model = model
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.rgb_means = (0.485,0.456,0.406)
        self.std = (0.229,0.224,0.225)

    def inference(self, img, timer):
        img, _ = preproc(img, self.test_size, self.rgb_means, self.std)
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img).cpu().float().contiguous()
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)

        return outputs

class TRTPredictor(object):
    def __init__(self, model, engine_path, exp, device, args):
        ### create pytorch model
        self.exp = exp
        #### TRT create engine ####
        self.engine_path = engine_path
        self.device = device
        self.model = TRTModule()
        self.model.load_state_dict(torch.load(self.engine_path))

        # for decode
        self.hw = deepcopy(model.head.hw)
        self.strides = deepcopy(model.head.strides)
        ######################
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.rgb_means = (0.485,0.456,0.406)
        self.std = (0.229,0.224,0.225)

        # if self.exp.deepsort:
        #     if self.exp.sct_trt:
        #         self.sct_featext = TRTModule()
        #         self.sct_featext.load_state_dict(torch.load(exp.sct_reid_path))
        #     else:
        #         self.sct_featext = Baseline(exp.sct_reid_path,'after',exp.sct_model).eval()
        #         if exp.fp16:
        #             self.sct_featext = self.sct_featext.half()
        #         self.sct_featext.to('cuda')
        # else:
        #     self.sct_featext = None
        self.sct_featext = None

    def decode_outputs(self, outputs, dtype, device):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype).to(device)
        strides = torch.cat(strides, dim=1).type(dtype).to(device)
        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    def inference(self, img, timer):
        # preprocess
        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            outputs = self.decode_outputs(outputs, dtype=outputs.cpu().type(), device=outputs.device).cpu().contiguous()
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        
        return outputs
    
    def __del__(self):
        """Free CUDA memories and context."""
        del self.model

class Camera(BaseCamera):
    def __init__(self, feed_type, device_number, file, gpu_number, args):
        unique_name = (feed_type, device_number)

        BaseCamera.height = args.height
        BaseCamera.width = args.width
        exp = get_exp(args.exp_file, args.exp_name)
        BaseCamera.yolo_exp = exp
        BaseCamera.args = args

        BaseCamera.det_gpu_number[unique_name] = gpu_number
        BaseCamera.timers[unique_name] = Timer()

        roi = cv2.imread(f'../../test/Cam{device_number}/roi.jpg', 0)
        # roi = cv2.resize(roi, (exp.test_size[1], exp.test_size[0]), interpolation=cv2.INTER_NEAREST) # (608, 1088)or (800, 1440)
        BaseCamera.rois[device_number] = roi

        ### Video capturer ###
        if os.path.isfile(file):
            cap = cv2.VideoCapture(file)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file)

        cam_name = Path(file).stem
        BaseCamera.video_capturers[unique_name] = cap
        print(f'[Device {device_number}] Start video streaming for {cam_name}...')

        ### create YoloX predictor for each camera ###
        if not args.trt:
            detector = exp.get_model().to(f"cuda:{gpu_number}").eval()
            detector.load_state_dict(torch.load(args.det_weightfile, map_location='cpu')['model'])
            if exp.fuse: 
                detector = fuse_model(detector)
            if exp.fp16:
                detector = detector.half()

            predictor = Predictor(detector, exp, gpu_number, args, fp16=exp.fp16)
            BaseCamera.detectors[unique_name] = predictor
            print(f'[Device {device_number}] Create Detector on GPU{gpu_number}')
        
        super(Camera, self).__init__(feed_type, device_number)

    @classmethod
    def yolo_frames(self, unique_name):
        device_number = unique_name[1]
        exp = BaseCamera.yolo_exp
        exp.fuse = False
        args = BaseCamera.args

        gpu_number = BaseCamera.det_gpu_number[unique_name]
        cap = BaseCamera.video_capturers[unique_name]
        timer = BaseCamera.timers[unique_name]
        roi = BaseCamera.rois[device_number]
        
        if args.trt: # create fake model for self.hw, self.strides in model.head
            assert not exp.fuse
            assert os.path.exists(args.det_trtfile)

            model = exp.get_model().eval()
            model.head.decode_in_inference = False
            model(torch.ones((1, 3, exp.test_size[0], exp.test_size[1])))
            # create TRT predictor
            predictor = TRTPredictor(model, args.det_trtfile, exp, gpu_number, args) 
            del model
            print(f'[Device {device_number}] Create TRT-Detector.')
            
            BaseCamera.detectors[unique_name] = predictor
        else:
            predictor = BaseCamera.detectors[unique_name]

        # Go! #
        fid = 0
        while True:
            ret, frame = cap.read()
            if ret:
                fid += 1
                frame = cv2.resize(frame, (exp.test_size[1], exp.test_size[0])) # TODO:turn width and height with yolox input_size, then remove the preprocessing part (resize) of the yolox
            else:
                yield None, None, None, None, None
            
            boxes = predictor.inference(frame, timer)
            timer.toc()

            # boxes contains: (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            if boxes[0] == None:
                boxes = np.array([])
            else:
                boxes = boxes[0][:, :5].numpy() # x1, y1, x2, y2, score
                boxes = boxes[boxes[:, 4] > 0.1, :] # keep the boxes with specific confidence scores
                
                # Remove the bounding boxes with ROI #
                mb = np.stack(((boxes[:,0]+boxes[:,2])/2, boxes[:,3]), axis=1).astype(np.int_) # (N, 2)
                mb[:, 0] = np.clip(mb[:, 0]*(2704/args.width) , 0, 2703)
                mb[:, 1] = np.clip(mb[:, 1]*(1520/args.height), 0, 1519)
                boxes = boxes[roi[mb[:, 1], mb[:, 0]] >= 128, :]

                boxes[:, 0] = np.clip(boxes[:, 0], 0, exp.test_size[1]-1)
                boxes[:, 1] = np.clip(boxes[:, 1], 0, exp.test_size[0]-1)
                boxes[:, 2] = np.clip(boxes[:, 2], 0, exp.test_size[1]-1)
                boxes[:, 3] = np.clip(boxes[:, 3], 0, exp.test_size[0]-1)

            # if fid % args.print_interval == 0:
            #     print('[Device {}] Detection Thread frame {:5d} ({:.2f} fps) | DET:{:.5f}'
            #     .format(device_number, fid, 1./max(1e-5, timer.average_time), timer.average_time))
            
            yield frame, fid, boxes, None, timer.average_time


# Max_Det in testing set #
# yolox_s  {1: 78, 2: 35, 3: 32, 4: 23, 5: 18, 6: 20, 7: 15, 8: 22, 9: 58, 10: 46, 11: 28}
# yolox_m  {1: 20, 2: 20, 3: 23, 4: 14, 5: 15, 6: 14, 7: 10, 8: 16, 9: 15, 10: 14, 11: 15}