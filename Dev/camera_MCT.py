from __future__ import division, print_function, absolute_import
import os, sys, cv2, time, warnings
import numpy as np
import os.path as osp
from copy import deepcopy
from base_camera import BaseCamera
import torch.nn.functional as F

from yolox.utils.visualize import plot_tracking
from yolox.tracking_utils.timer import Timer

warnings.filterwarnings('ignore')

class Camera(BaseCamera):
    def __init__(self, feed_type, setting, args):
        gateway_number = setting['host_number']
        unique_name = (feed_type, gateway_number)
        BaseCamera.setting = setting
        BaseCamera.MCT_server.update_camera(setting, args)
        BaseCamera.timers[unique_name] = Timer()

        super(Camera, self).__init__(feed_type, gateway_number)

    @classmethod
    def MCT_frames(self, unique_name):
        gateway_number = unique_name[1]
        setting = BaseCamera.setting
        args = BaseCamera.args
        
        client_cnt = len(setting['client_number'])
        get_feed_from = [('track', d) for d in setting['client_number']]
        tracker = [BaseCamera.trackers[g] for g in get_feed_from]
        timer = BaseCamera.timers[unique_name]
        
        # wait for server_thread to create
        check = [g not in BaseCamera.frame for g in get_feed_from]
        while any(check): 
            time.sleep(0)

        num_process_frames = 1
        while True:
            ### Get data from buffers ###
            data = [BaseCamera.get_frame(g) for g in get_feed_from]
            assert len(data) == client_cnt
            
            ### End Check ###
            fid_list = [d[1] for d in data]
            if None in fid_list:
                yield None, None, None, None, None
                
            ### synchronization ###
            fid = fid_list[0]
            assert len(fid_list) == fid_list.count(fid)
            assert fid == num_process_frames
            num_process_frames += 1
            

            #######################################################
            #################### Main function ####################
            #######################################################
            ### Get enter and leave tracklets from each tracker ###
            enter_list = []
            leave_list = []
            for i in range(client_cnt):
                enter_list.extend(tracker[i].get_enter_tracks())
                leave_list.extend(tracker[i].get_exit_tracks())

            ### Matching process ###
            timer.tic()
            BaseCamera.MCT_server.push_to_memory(fid, enter_list, leave_list)
            if fid % args.exchange_interval == 0: 
                BaseCamera.MCT_server.P2P(mode='before')
            if fid % args.exchange_interval == 0:
                BaseCamera.MCT_server.update()
            if fid % args.exchange_interval == 0:
                BaseCamera.MCT_server.P2P(mode='after')
            timer.toc()
            
            ### Compute processing time ###
            duration_list = [d[4] for d in data]
            total_time = [sum(d) for d in duration_list]
            duration = duration_list[total_time.index(max(total_time))]

            if fid % args.print_interval == 0:
                print('Tracking Thread frame {} ({:.2f} fps) | DET:{:.6f}, REID:{:.6f}, Track:{:.6f}, MCT:{:.6f}'
                .format(fid, 1./max(1e-5, timer.average_time+sum(duration)), duration[0], duration[1], duration[2], timer.average_time))
            

            frame_list = [d[0] for d in data]
            online_targets = [d[2] for d in data]
            # if args.visualize:
            #     for f, tgt in zip(frame_list, online_targets):
            #         plot_tlwhs = [t.tlwh for t in tgt]
            #         online_ids = [t.track_id for t in tgt]
            #         online_scores = [t.score for t in tgt]
            #         online_global_ids = ['' for t in tgt]
            #         f = plot_tracking(
            #             f, plot_tlwhs, online_ids, scores = online_scores,
            #             frame_id = fid, global_IDs = online_global_ids)


            yield frame_list, fid, online_targets, None, None
