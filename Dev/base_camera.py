import sys, cv2, time, threading
import imagezmq
import numpy as np
from collections import deque,defaultdict
import torch
try:
    from greenlet import getcurrent as get_ident
except ImportError:
    try:
        from thread import get_ident
    except ImportError:
        from _thread import get_ident

from yolox.tracker.matching import linear_assignment

class MultiCam_Server(object):
    def __init__(self):
        """
        enter_dict:
            new track enter in the camera view, like the role of new detections in ByteTracker
            when matching with leave_dict, track_id needs to be changed to the track_id matched with
        leave_dict:
            old track leave the camera view, like the role of lost tracks in ByteTrack

        """
        self.args = None

        self.reid_thres = 0.3
        self.enter_last_time = 5*60
        self.leave_last_time = 45*60 # 1min = 3600 frames # it should depends on the traveling time!!!

        self.fid = 0
        self.sending = None

        self.enter_dict = []

        self.leave_feat = None
        self.leave_pid = []
        self.leave_end = []
        self.leave_dev = [] 
        self.leave_reg = [] # region_number

        self.new_leave_feat = None
        self.new_leave_pid = []
        self.new_leave_end = []
        self.new_leave_dev = []
        self.new_leave_reg = []

        self.other_leave_feat = None
        self.other_leave_pid = []
        self.other_leave_end = []
        self.other_leave_dev = []
        self.other_leave_reg = []

        self.match_pid = []

        self.windows = []

    """ update camera information """
    def update_camera(self, setting, args):
        self.args = args

        self.receivers = []
        for ip in setting['open_port'].values():
            self.receivers.append(imagezmq.ImageHub(open_port=ip))

        self.senders = []
        self.target_num = []
        for gate_num, ip in setting['connect_to'].items():
            self.senders.append(imagezmq.ImageSender(connect_to=ip))
            self.target_num.append(int(gate_num[-1]))

        self.gid = setting['host_number']
        self.target_reg = list(setting['transmit'].values())

        if args.window:
            self.windows = np.load('./travel_time.npy')
            # distribution = distribution / max(distribution) #TODO: How to make it to possibilities

    """ push track infomation to MCT server """
    def push_to_memory(self, fid, enter, leave):
        assert self.fid + 1 == fid, f'MCT_server:{self.fid} ,tracking_thread: {fid}'
        self.fid += 1
        
        if enter: # save all info in STrack
            self.enter_dict.extend(enter) 
        if leave: 
            feats = np.asarray(self.get_feats(leave, mode='leave'))
            self.new_leave_feat = feats if self.new_leave_feat is None else np.concatenate((self.new_leave_feat, feats), axis=0)
            for t in leave:
                self.new_leave_pid.append(t.track_id)
                self.new_leave_end.append(t.end_frame)
                self.new_leave_dev.append(t.device_number)
                self.new_leave_reg.append(t.leave_region)

    """ transceive between gateways """
    def transmit(self, sender, target_reg, mode='before'):
        # prepare data
        if mode == 'before':
            if len(self.new_leave_pid) > 0:
                pids = np.asarray(self.new_leave_pid)[:, np.newaxis]
                ends = np.asarray(self.new_leave_end)[:, np.newaxis]
                devs = np.asarray(self.new_leave_dev)[:, np.newaxis]
                regs = np.asarray(self.new_leave_reg)[:, np.newaxis]
                self.sending = np.concatenate((pids, ends, devs, regs, self.new_leave_feat), axis=1)

                if self.args.mask: # only send possible info to other gateways
                    idx = np.asarray(target_reg == regs)
                    idx = idx.reshape(self.sending.shape[0])
                    self.sending = self.sending[idx, :]
                    if len(self.sending) == 0: self.sending = None # change it to count true 

                # update
                self.leave_feat = self.new_leave_feat if self.leave_feat is None else np.concatenate((self.leave_feat, self.new_leave_feat), axis=0)
                self.leave_pid.extend(self.new_leave_pid)
                self.leave_end.extend(self.new_leave_end)
                self.leave_dev.extend(self.new_leave_dev)
                self.leave_reg.extend(self.new_leave_reg)

                self.new_leave_feat = None
                self.new_leave_pid = []
                self.new_leave_end = []
                self.new_leave_dev = []
                self.new_leave_reg = []

        elif len(self.match_pid) > 0:
            self.sending = np.array(self.match_pid)
            self.match_pid = []
        
        # sending data
        if self.sending is not None:
            sender.send_image(self.fid, self.sending)

        # send a break signal
        sender.send_image(0, np.array(0))
    
    def receive(self, receiver, mode='before'):
        while True:
            msg, info = receiver.recv_image()

            # check whether is the target gateways
            receiver.send_reply(b'OK')

            if msg == 0:
                break
            else:
                assert msg == self.fid, f'FID:{self.fid}, others:{msg}' # syn between gateways
                if mode == 'before':
                    assert info.shape[1] == 516 #512+1+1+1+1
                    feats = np.asarray(info[:, 4:])
                    self.other_leave_feat = feats if self.other_leave_feat is None else np.concatenate((self.other_leave_feat, feats), axis=0)
                    self.other_leave_pid.extend(info[:, 0].astype(np.int_).tolist())
                    self.other_leave_end.extend(info[:, 1].astype(np.int_).tolist())
                    self.other_leave_dev.extend(info[:, 2].astype(np.int_).tolist())
                    self.other_leave_reg.extend(info[:, 3].astype(np.int_).tolist())
                else:
                    if len(self.other_leave_pid) > 0:
                        info = info.tolist()
                        keep_idx = [idx for idx, p in enumerate(self.other_leave_pid) if p not in info]

                        self.other_leave_pid  = self.keep_list_index(self.other_leave_pid, keep_idx)
                        self.other_leave_end  = self.keep_list_index(self.other_leave_end, keep_idx)
                        self.other_leave_dev  = self.keep_list_index(self.other_leave_dev, keep_idx)
                        self.other_leave_reg  = self.keep_list_index(self.other_leave_reg, keep_idx)
                        self.other_leave_feat = None if len(keep_idx) == 0 else np.take(self.other_leave_feat, keep_idx, axis=0)
                                   
    def P2P(self, mode='before'): # point-to-point
        """ mode: before/after update stage """
        for receiver, sender, tgt, tgt_reg in zip(self.receivers, self.senders, self.target_num, self.target_reg):
            if tgt > self.gid:
                print(f'[Gateway {self.gid}: {mode}] Connecting with Gateway{tgt}...')
                self.receive(receiver, mode=mode)
                print(f'Receive Stage Done')
                self.transmit(sender, tgt_reg, mode=mode)
                print(f'Transmit Stage Done')
            else:
                print(f'[Gateway {self.gid}: {mode}] Connecting with Gateway{tgt}...')
                self.transmit(sender, tgt_reg, mode=mode)
                print(f'Transmit Stage Done')
                self.receive(receiver, mode=mode)
                print(f'Receive Stage Done')

        self.sending = None

    """ update MC Tracker """
    def update(self):
        # matching
        leave_cnt = len(self.leave_pid) + len(self.new_leave_pid) + len(self.other_leave_pid)
        if len(self.enter_dict) > 0 and leave_cnt > 0:
            enters = self.get_feats(self.enter_dict, mode='enter')
            leaves = self.get_leave_feats()

            dists = self.embedding_distance(leaves, enters)
            dists = self.condition(dists, leave_cnt)
        
            matches, u_leaves, u_enters = linear_assignment(dists, self.reid_thres)

            match_list = []
            leave_pids = self.leave_pid + self.new_leave_pid + self.other_leave_pid
            # leave_regs = self.leave_reg + self.new_leave_reg + self.other_leave_reg

            for i_leave, i_enter in matches:
                # update the STrack information: track_id, and feats
                STrack_enter = self.enter_dict[i_enter]
                STrack_enter.track_id = leave_pids[i_leave]
                STrack_enter.update_mct_feature(leaves[i_leave])
                match_list.append(i_leave)

            # remove the matching pools and cands from the dict()
            self.keep(match_list, u_leaves, u_enters)
        
        # remove the pools and cands last for a long time
        self.remove()
    
    def keep(self, match_list, u_leaves, u_enters):
        # tell other gateways to delete the cands, only sent STracks are needed to be collected
        self.match_pid.extend([self.leave_pid[m] for m in match_list if m < len(self.leave_pid)])
        # self.match_pid.extend([self.leave_pid[m] for m in match_list])

        self.enter_dict = self.keep_list_index(self.enter_dict, u_enters)

        ### can rewrite better ??? ###
        other_idx = [u - len(self.leave_pid) - len(self.new_leave_pid) for u in u_leaves if u >= len(self.leave_pid) + len(self.new_leave_pid)]
        remains = [u for u in u_leaves if u < len(self.leave_pid) + len(self.new_leave_pid)]
        new_idx = [u - len(self.leave_pid) for u in remains if u >= len(self.leave_pid)]
        ori_idx = [u for u in remains if u < len(self.leave_pid)]

        self.leave_pid  = self.keep_list_index(self.leave_pid,  ori_idx)
        self.leave_end  = self.keep_list_index(self.leave_end,  ori_idx)
        self.leave_dev  = self.keep_list_index(self.leave_dev,  ori_idx)
        self.leave_reg  = self.keep_list_index(self.leave_reg,  ori_idx)
        self.leave_feat = None if len(ori_idx) == 0 else np.take(self.leave_feat, ori_idx, axis=0)
        self.new_leave_pid  = self.keep_list_index(self.new_leave_pid,  new_idx)
        self.new_leave_end  = self.keep_list_index(self.new_leave_end,  new_idx)
        self.new_leave_dev  = self.keep_list_index(self.new_leave_dev,  new_idx)
        self.new_leave_reg  = self.keep_list_index(self.new_leave_reg,  new_idx)
        self.new_leave_feat = None if len(new_idx) == 0 else np.take(self.new_leave_feat, new_idx, axis=0)
        self.other_leave_pid  = self.keep_list_index(self.other_leave_pid,  other_idx)
        self.other_leave_end  = self.keep_list_index(self.other_leave_end,  other_idx)
        self.other_leave_dev  = self.keep_list_index(self.other_leave_dev,  other_idx)
        self.other_leave_reg  = self.keep_list_index(self.other_leave_reg,  other_idx)
        self.other_leave_feat = None if len(other_idx) == 0 else np.take(self.other_leave_feat, other_idx, axis=0)

    def keep_list_index(self, l, idx):
        return [i for j,i in enumerate(l) if j in idx]

    def remove(self):
        if len(self.enter_dict) > 0 :
            keep_idx = []
            for idx, t in enumerate(self.enter_dict):
                if self.fid - t.start_frame <= self.enter_last_time:
                    keep_idx.append(idx)
            self.enter_dict = self.keep_list_index(self.enter_dict, keep_idx)

        if len(self.leave_pid) > 0 :
            keep_idx = []
            for idx, t in enumerate(self.leave_end):
                if self.fid - t <= self.leave_last_time:
                    keep_idx.append(idx)
            self.leave_pid  = self.keep_list_index(self.leave_pid,  keep_idx)
            self.leave_end  = self.keep_list_index(self.leave_end,  keep_idx)
            self.leave_dev  = self.keep_list_index(self.leave_dev,  keep_idx)
            self.leave_reg  = self.keep_list_index(self.leave_reg,  keep_idx)
            self.leave_feat = None if len(keep_idx) == 0 else np.take(self.leave_feat, keep_idx, axis=0)
        
        if len(self.new_leave_pid) > 0 :
            keep_idx = []
            for idx, t in enumerate(self.new_leave_end):
                if self.fid - t <= self.leave_last_time:
                    keep_idx.append(idx)
            self.new_leave_pid  = self.keep_list_index(self.new_leave_pid,  keep_idx)
            self.new_leave_end  = self.keep_list_index(self.new_leave_end,  keep_idx)
            self.new_leave_dev  = self.keep_list_index(self.new_leave_dev,  keep_idx)
            self.new_leave_reg  = self.keep_list_index(self.new_leave_reg,  keep_idx)
            self.new_leave_feat = None if len(keep_idx) == 0 else np.take(self.new_leave_feat, keep_idx, axis=0)
        
        if len(self.other_leave_pid) > 0 :
            keep_idx = []
            for idx, t in enumerate(self.other_leave_end):
                if self.fid - t <= self.leave_last_time:
                    keep_idx.append(idx)
            self.other_leave_pid  = self.keep_list_index(self.other_leave_pid,  keep_idx)
            self.other_leave_end  = self.keep_list_index(self.other_leave_end,  keep_idx)
            self.other_leave_dev  = self.keep_list_index(self.other_leave_dev,  keep_idx)
            self.other_leave_reg  = self.keep_list_index(self.other_leave_reg,  keep_idx)
            self.other_leave_feat = None if len(keep_idx) == 0 else np.take(self.other_leave_feat, keep_idx, axis=0)
            
    def get_feats(self, STrack, mode='enter'):
        feat = []
        for t in STrack:
            if mode=='enter':
                feat.append(t.sct_feat)
            else:
                feat.append(t.mct_feat)
        return torch.stack(feat, dim=0)
    
    def get_leave_feats(self):
        cands = None
        cnt = [len(self.leave_pid), len(self.new_leave_pid), len(self.other_leave_pid)]
        for idx, c in enumerate(cnt):
            if c != 0:
                if idx == 0:
                    cands = self.leave_feat
                elif idx == 1:
                    cands = self.new_leave_feat if cands is None else np.concatenate((cands, self.new_leave_feat), axis=0)
                elif idx == 2:
                    cands = self.other_leave_feat if cands is None else np.concatenate((cands, self.other_leave_feat), axis=0)
        return cands

    def embedding_distance(self, feata, featb, metric='cosine'):
        cost_matrix = np.zeros((len(feata), len(featb)), dtype=np.float)
        if cost_matrix.size == 0:
            return cost_matrix

        feata = np.asarray(feata)
        featb = np.asarray(featb)
        
        cost_matrix = np.maximum(0.0, 1-np.matmul(feata,featb.T))
        return cost_matrix

    def condition(self, dists, leave_cnt):
        # temporal
        pool_str = np.array([t.start_frame for t in self.enter_dict]) # start
        cand_end = np.array(self.leave_end + self.new_leave_end + self.other_leave_end)
        pool_str = np.tile(pool_str, (leave_cnt, 1)) # start
        cand_end = np.tile(cand_end, (len(self.enter_dict), 1)).T

        # region
        pool_reg = np.array([t.enter_region for t in self.enter_dict])
        cand_reg = np.array(self.leave_reg + self.new_leave_reg + self.other_leave_reg)
        pool_reg = np.tile(pool_reg, (leave_cnt, 1))
        cand_reg = np.tile(cand_reg, (len(self.enter_dict), 1)).T

        # device
        pool_dev = np.array([t.device_number for t in self.enter_dict])
        cand_dev = np.array(self.leave_dev + self.new_leave_dev + self.other_leave_dev)
        pool_dev = np.tile(pool_dev, (leave_cnt, 1))
        cand_dev = np.tile(cand_dev, (len(self.enter_dict), 1)).T

        mask = np.zeros(dists.shape, dtype=bool)
        if self.args.mask: # target diff camera but same region
            mask = np.logical_or(pool_reg != cand_reg, mask) # diff. region -> max cost
            mask = np.logical_or(pool_dev == cand_dev, mask) # same camera -> max cost
        if self.args.window: # two different type of time window: possibilities & hard time window 
            mask = mask # time offset within time window 
        if self.args.overlap:
            mask = np.logical_or(pool_str <= cand_end, mask) # timeline overlapping -> max cost

        dists[mask] = 1 # max_cost
        return dists


class BaseCamera:
    threads = {}  # background thread that reads frames from camera
    frame = {}  # current frame is stored here by background thread
    thread_condition = {}

    # camera_server
    video_capturers ={}
    rois = {}
    height = None
    width = None

    # yolo_server
    detectors = {}
    det_gpu_number ={}
    timers = {}
    args = None
    yolo_exp = None

    # track_server
    trackers = {}

    # reid_server
    extractors = {}
    cfg = None

    # MCT server
    MCT_server = MultiCam_Server()
    setting = None
    timing = []

    def __init__(self, feed_type, device_number):
        """Start the background camera thread if it isn't running yet."""
        self.unique_name = (feed_type, device_number)
        BaseCamera.frame[self.unique_name] = None
        BaseCamera.thread_condition[self.unique_name] = True

        if self.unique_name not in BaseCamera.threads:
            # start background frame thread
            BaseCamera.threads[self.unique_name] = threading.Thread(target=self._thread,
                                                                    args=(feed_type, device_number))
    
    @classmethod
    def get_FPS(cls):
        return (len(BaseCamera.timing)-1)/(BaseCamera.timing[-1]-BaseCamera.timing[0])

    @classmethod
    def get_frame(cls, unique_name):
        """Return the current camera frame."""
        # wait for a signal from the camera thread
        while BaseCamera.frame[unique_name] is None:
            if BaseCamera.thread_condition[unique_name]: # wait
                time.sleep(0)
            else: # file end
                return (None, None, None, None, None)
        data = BaseCamera.frame[unique_name]
        BaseCamera.frame[unique_name] = None

        return data

    @classmethod
    def track_frames(unique_name):
        """"Generator that returns frames from the camera."""
        raise RuntimeError('Must be implemented by subclasses')

    @classmethod
    def yolo_frames(unique_name):
        """"Generator that returns frames from the camera."""
        raise RuntimeError('Must be implemented by subclasses')

    @classmethod
    def MCT_frames(unique_name):
        """"Generator that returns frames from the camera."""
        raise RuntimeError('Must be implemented by subclasses')

    @classmethod
    def track_thread(cls, unique_name):
        device_number = unique_name[1]
        frames_iterator = cls.track_frames(unique_name)

        for frame, fid, online_targets, matching, duration in frames_iterator:
            if fid is None:
                frames_iterator.close()
                BaseCamera.thread_condition[unique_name] = False
                print(f'Stopping tracking thread for device {device_number} due to ending of the video.')

            # wait for next thread to take the frame
            while BaseCamera.frame[unique_name] is not None: 
                time.sleep(0)

            BaseCamera.frame[unique_name] = (frame, fid, online_targets, matching, duration)

    @classmethod
    def yolo_thread(cls, unique_name):
        device_number = unique_name[1]
        frames_iterator = cls.yolo_frames(unique_name)

        for frame, fid, boxes, _, duration in frames_iterator:
            if fid is None:
                frames_iterator.close()
                BaseCamera.thread_condition[unique_name] = False
                print(f'Stopping detection thread for device {device_number} due to ending of the video.')

            # wait for next thread to take the frame
            while BaseCamera.frame[unique_name] is not None: 
                time.sleep(0)

            BaseCamera.frame[unique_name] = (frame, fid, boxes, _, duration)
    
    @classmethod
    def MCT_thread(cls, unique_name):
        device_number = unique_name[1]
        frames_iterator = cls.MCT_frames(unique_name)

        for frame, fid, online_targets, _, _ in frames_iterator:
            if fid is None:
                frames_iterator.close()
                BaseCamera.thread_condition[unique_name] = False
                print(f'Stopping MCT thread for Gateway{device_number} due to ending of the video.')

            # wait for next thread to take the frame
            while BaseCamera.frame[unique_name] is not None: 
                time.sleep(0)

            BaseCamera.frame[unique_name] = (frame, fid, online_targets, None, None)
            BaseCamera.timing.append(time.time())
            
            

    @classmethod
    def _thread(cls, feed_type, device_number):
        unique_name = (feed_type, device_number)

        if feed_type == 'yolo':
            print('Starting Detection thread for device {}.'.format(device_number))
            cls.yolo_thread(unique_name)

        elif feed_type == 'track':
            print('Starting Tracking thread for device {}.'.format(device_number))
            cls.track_thread(unique_name)

        elif feed_type == 'MCT':
            print('Starting MCT thread for device {}.'.format(device_number))
            cls.MCT_thread(unique_name)
            
        BaseCamera.threads[unique_name] = None
 
 