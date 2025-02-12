import numpy as np
from collections import deque
import os, sys
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState
from collections import deque, defaultdict
from torch2trt import TRTModule
import pycuda.autoinit

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, device_number, init_feat=None, use_feat=True, init_region=None):
        # the element need to be transimited between Gateways:
        # STrack.mct_feat, STrack.device_number, STrack.start_frame, STrack.end_frame, STracks.track_id

        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.device_number = device_number
        self.score = score
        self.tracklet_len = 0
        self.use_feat = use_feat
        self.direction = [0.0, 0.0]

        self.enter_region = init_region
        self.leave_region = 0
        
        # for feats
        if use_feat:
            assert init_feat is not None
            self.sct_feat = init_feat
            self.mct_feat = None
            self.updated_num = 0 # help to get the new_enter STracks
            self.push = False

    def update_sct_feature(self, feat, score=None):
        if score >= 0.5:
            self.sct_feat = 0.9*self.sct_feat + 0.1*feat # TODO:try other parameters, e.g. lower score lower impact of updated feats
            self.sct_feat = F.normalize(self.sct_feat, dim=0)
        self.updated_num += 1

    def update_mct_feature(self, feat):
        feat = torch.from_numpy(feat)
        if self.mct_feat is None:
            self.mct_feat = feat
        else:
            self.mct_feat = 0.5*self.mct_feat + 0.5*feat
        self.mct_feat = F.normalize(self.mct_feat, dim=0)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def next_id(self):
        # return the id with device_number: ID 1 in Cam1 --> 101, ID 20 in Cam10 --> 2010
        BaseTrack._count += 1
        return BaseTrack._count*100 + self.device_number

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 1
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.frame_id = frame_id
        self.tracklet_len = 1
        self.direction = [0.0, 0.0]

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if new_id:
            self.track_id = self.next_id()
        if self.use_feat:
            self.update_sct_feature(new_track.sct_feat, score=new_track.score)

    def update(self, new_track, frame_id): # TODO: difference between reactivate and update???
        self.frame_id = frame_id
        self.tracklet_len += 1

        self.direction = self.mean[:2].copy() - new_track.cxcy

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if self.use_feat:
            self.update_sct_feature(new_track.sct_feat, score=new_track.score)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        ret[2:] += ret[:2] #xyxy
        ret[0] = limit(ret[0], 0, 1440)
        ret[1] = limit(ret[1], 0, 800)
        ret[2] = limit(ret[2], 0, 1440)
        ret[3] = limit(ret[3], 0, 800)
        ret[2:] -= ret[:2]
        return ret

    @property
    # @jit(nopython=True)
    def cxcy(self):
        ret = self._tlwh.copy()
        ret[:2] += ret[2:]/2
        return ret[:2]

    @property
    # @jit(nopython=True)
    def cxcywh(self):
        ret = self._tlwh.copy()
        ret[:2] += ret[2:]/2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret
    
    @property
    # @jit(nopython=True)
    def mb(self):
        """Convert bounding box to format `(middle x, max y)` (middle bottom point)
        """
        ret = self.tlwh.copy()
        ret[0] = ret[0] + ret[2]*0.5
        ret[1] = ret[1] + ret[3]
        return ret[:2]

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret
    
    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_mb(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[0] = (ret[0] + ret[2])/2
        ret[1] = ret[3]
        return ret[:2]

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args, device_number, use_feat=False, frame_rate=60, region_mask=None):
        #####
        # use_feat:
        #     features are already generated, choose whether use_feat in association step, 
        #     using features may cost more time
        #####

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks    = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.leaved_stracks  = []  # type: list[STrack]
        self.last_num_leaved = 0

        self.frame_id = 0
        self.args = args
        self.use_feat = use_feat
        self.device_number = device_number
        self.region_mask = region_mask

        self.kalman_filter = KalmanFilter()


        # self-defined parameters
        self.det_thresh = args.track_thresh + 0.1 # 0.5+0.1
        self.max_time_lost = 1.5*60  # unit: frame
        self.init_enter_track = 15

    def get_exit_tracks(self):
        new_exit = []
        for t in self.leaved_stracks:
            # mct_feat may contain feature information in other cams, but sct_feat only contains the information in this cam
            t.mct_feat = t.sct_feat
            if self.region_mask is not None:
                t.leave_region = self.region_mask[int(t.mb[1]*(1520/800)), int(t.mb[0]*(2704/1440))]
                if t.leave_region > 11: # local or mask
                    continue
                # print(f'[Leave] {t.track_id}: ', [int(t.mb[1]*(1520/800)), int(t.mb[0]*(2704/1440))], t.mb, t.tlwh, t.leave_region)
            new_exit.append(t)
        self.leaved_stracks = []
        return new_exit

    def get_enter_tracks(self):
        new_enter = []
        for t in self.tracked_stracks:
            if t.updated_num == self.init_enter_track and t.push == False:
                t.push = True
                if self.region_mask is not None:
                    if t.enter_region > 11: # local or mask -> do not pass
                        continue
                    # print(f'[Enter] {t.track_id}: ', [int(t.mb[1]*(1520/800)), int(t.mb[0]*(2704/1440))], t.mb, t.tlwh, t.enter_region)
                new_enter.append(t)
        return new_enter

    def update_with_no_boxes(self):
        self.frame_id += 1
        lost_stracks    = []
        removed_stracks = []
        leaved_stracks  = []
        tracked_stracks = []

        for track in self.tracked_stracks:
            if not track.is_activated: # unconfirmed stracks which contains only one frame
                track.mark_removed() 
                removed_stracks.append(track)
            else:
                tracked_stracks.append(track)
        
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)
        for track in strack_pool:
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                leaved_stracks.append(track)

        self.tracked_stracks = {} # nothing, all tracked_stracks are turned into removed or lost
        
        self.removed_stracks.extend(removed_stracks)
        self.leaved_stracks.extend(leaved_stracks)
        
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.leaved_stracks)
        
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)

    def update(self, output_results, predictions, img_size):
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        leaved_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]

        ''' Split the detections with confidence scores '''
        remain_inds = scores > self.args.track_thresh
        inds_low = scores > self.args.low_thresh
        inds_high = scores < self.args.track_thresh
        inds_second = np.logical_and(inds_low, inds_high) # 0.1 < score < 0.5

        # high score detections
        dets = bboxes[remain_inds]        
        feats = predictions[remain_inds]
        scores_keep = scores[remain_inds]

        # low score detections
        dets_second = bboxes[inds_second] # low score detections
        feats_second = predictions[inds_second]
        scores_second = scores[inds_second]

        ''' Generate STracks with new detections '''
        detections = []
        if len(dets) > 0:
            for (tlbr, s, f) in zip(dets, scores_keep, feats):
                mb = STrack.tlbr_to_mb(tlbr)
                init_region = self.region_mask[int(mb[1]*(1520/800)), int(mb[0]*(2704/1440))] if self.region_mask is not None else None
                detections.append(STrack(STrack.tlbr_to_tlwh(tlbr), s, self.device_number, init_feat=f, init_region=init_region))

        ''' Step1 : Add newly detected tracklets to tracked_stracks '''
        #liu : put activated tracks to "track_stracks",
        #liu : put non-activated tracks to "uncomfirmed" (two consecutive tracked will be activated)
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        
        ''' Step 2: First association, with high score detection boxes '''
        #liu : "strack_pool" contains activated tracks and lost tracks, may contain duplication
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # if self.use_feat:
        #     dists = matching.embedding_distance(strack_pool, detections)
        #     dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        #     # dists = matching.fuse_iou(dists, strack_pool, detections, iou_alpha=0.5)
        #     matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.3) # TODO: check the parameter, smaller is better
        # else:        
        dists = matching.iou_distance(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            # liu: if the matched track has been tracked, update and append to "activated_stracks"
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            # liu : if the matched track is lost, reactivtate and append to "refind_stracks"
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)



        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        detections_second = []
        if len(dets_second) > 0:
            for (tlbr, s, f) in zip(dets_second, scores_second, feats_second):
                mb = STrack.tlbr_to_mb(tlbr)
                init_region = self.region_mask[int(mb[1]*(1520/800)), int(mb[0]*(2704/1440))] if self.region_mask is not None else None
                detections_second.append(STrack(STrack.tlbr_to_tlwh(tlbr), s, self.device_number, init_feat=f, init_region=init_region))

        # liu : "r_tracked_stracks" filter the unmatched tracks with tracked state: lost state will not join in the second matching
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            # liu: if the matched track has been tracked, update and append to "activated_stracks"
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            # liu : if the matched track is lost, reactivtate and append to "refind_stracks"
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)


        # ''''''''''''''''''''''''''''''''''''''''''''' Extra stage based on CYK's idea '''''''''''''''''''''''''''''''''''''''''''''''''''''
        # lost_strack_pool = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Lost] 
        # lost_track_index = [i for i in u_track if strack_pool[i].state == TrackState.Lost]
        # detections = [detections[i] for i in u_detection] 
        # dists = feat_dists[lost_track_index, :][:, u_detection]
        # dists = matching.fuse_velocity(dists, lost_strack_pool, detections, self.frame_id)
        # matches, u_track_lost, u_detection = matching.linear_assignment(dists, thresh=0.5)
        # for itracked, idet in matches:
        #     track = lost_strack_pool[itracked]
        #     det = detections[idet]
        #     track.re_activate(det, self.frame_id, new_id=False)
        #     refind_stracks.append(track)

        ######
        # 'activated_stracks' contains tracked stracks updated with high & low detections
        # 'refind_stracks' contains lost stracks reactivated with high & low detectins
        ######

        # liu : Mark the finally unmatched stracks to lost track
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]  # u_detection : high score but unmatched
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        
        # liu :  update the unconfirmed tracks to activated. (an activated track should contains consecutive bboxes)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        # liu : remove frame with only one high boxes
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:  # 0.6
                continue
            track.activate(self.kalman_filter, self.frame_id) # unconfirmed
            activated_stracks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                leaved_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        
        self.removed_stracks.extend(removed_stracks)
        self.leaved_stracks.extend(leaved_stracks)
        
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.leaved_stracks)
        
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb


def get_region(mask, tlwh):
    x = int(tlwh[0] + tlwh[2] * 0.5)
    y = int(tlwh[1] + tlwh[3])
    return mask[x,y]

def limit(value, min_v, max_v):
    if value < min_v:
        return 0
    elif value >= max_v:
        return max_v-1
    else:
        return value
