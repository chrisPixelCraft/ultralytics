import os, sys, cv2, json, time, errno, socket, argparse
from threading import Thread
from importlib import import_module

import camera_detection
import camera_track
import camera_MCT

import random, torch
import numpy as np

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_yolo_streamer(setting, args):
    yolo_streamer = []
    for c, d in zip(setting['client_name'], setting['client_number']):
        video_pth = f'../../test/{c}/{c}.MP4'
        yolo_streamer.append(camera_detection.Camera('yolo', d, video_pth, 0, args))
    return yolo_streamer

def create_track_streamer(setting, args):
    track_streamer = []
    for c, d in zip(setting['client_name'], setting['client_number']):
        track_streamer.append(camera_track.Camera('track', d, 0, args))
    return track_streamer

def get_det_results(setting, streamer, feed_type, visualize=False, write_result=False):
    if visualize:
        os.makedirs('./image_output', exist_ok=True)
        for c in setting['client_number']:
            os.makedirs(f'./image_output/Cam{c}',  exist_ok=True)
    
    if write_result:
        os.makedirs('./text_output', exist_ok=True)

    idx = 1
    while True:         
        for i, c in enumerate(setting['client_number']):
            frame, fid, boxes, _, _= streamer[i].get_frame((feed_type, c))
            if fid is None: break
            assert idx == fid, f'Idx:{idx}, Fid:{fid}'

            if write_result and len(boxes) > 0:
                with open(f'./text_output/Cam{c}.txt', 'a') as fout:
                    for t in boxes:
                        fout.write(f'{fid} {t[0]} {t[1]} {t[2]} {t[3]} {t[4]}\n') # x, y, x, y, score 
            if visualize:
                for f, c in zip(frames, setting['client_number']):
                    cv2.imwrite(f'./image_output/Cam{c}/img{fid:06d}.jpg', f)
        if fid is None:
            break
        # print(idx)
        idx += 1

def get_track_results(args, setting, streamer, feed_type, visualize=False, write_result=False):
    if visualize:
        os.makedirs('./video_output', exist_ok=True)
        writers = []
        for c in setting['client_number']:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writers.append(cv2.VideoWriter(f'./video_output/Cam{c}.mp4', fourcc, 60.0, (1440,  800)))
    
    if write_result:
        os.makedirs(f'./{args.write_dir}', exist_ok=True)

    idx = 1
    while True:         
        for i, c in enumerate(setting['client_number']):
            frame, fid, online_targets, _, _= streamer[i].get_frame((feed_type, c))
            if fid is None: break
            assert idx == fid, f'Idx:{idx}, Fid:{fid}'

            if visualize:
                writers[i].write(frame)

            if write_result and len(online_targets) > 0:
                with open(f'./{args.write_dir}/Cam{c}.txt', 'a') as fout:
                    for t in online_targets: 
                        x, y, w, h = list(map(int, t.tlwh))
                        fout.write(f'{c} {t.track_id} {fid} {x} {y} {w} {h} {t.score:.5f} -1 -1\n') # cid, pid, fid, x, y, w, h, c(int) 
            
        if fid is None:
            break
        idx += 1
    
    if visualize:
        for w in writers:
            w.release()

def get_MCT_results(args, setting, streamer, feed_type, visualize=False, write_result=False):
    if write_result:
        os.makedirs(f'./{args.write_dir}', exist_ok=True)
    
    idx = 1
    while True:         
        frames, fid, online_targets, _, _= streamer.get_frame((feed_type, setting['host_number']))
        if fid is None: 
            FPS = streamer.get_FPS()
            print(f'Inside Average FPS: {FPS:.3f}')
            break
        assert len(online_targets) == len(setting['client_number'])
        assert idx == fid, f'Idx:{idx}, Fid:{fid}'
        # print(idx)
        idx += 1

        if write_result:
            for targets, c in zip(online_targets, setting['client_number']):
                if len(targets) == 0:
                    continue
                else:
                    with open(f'./{args.write_dir}/Cam{c}.txt', 'a') as fout:
                        for t in targets:
                            x, y, w, h = list(map(int, t.tlwh))
                            fout.write(f'{c} {t.track_id} {fid} {x} {y} {w} {h} -1 -1\n') # cid, pid, fid, x, y, w, h (int) 
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--det' , action='store_true')
    parser.add_argument('-R', '--reid', action='store_true')
    parser.add_argument('-M', '--mct' , action='store_true')
    parser.add_argument('-T', '--trt' , action='store_true')

    """Gateway Setting"""
    parser.add_argument('-J', '--json', type=str, default='./setting_Gateway4.json')

    """Detecion Setting"""
    # setting file path and name
    parser.add_argument('-expn', '--exp_name', type=str, default=None)
    parser.add_argument('-expf', '--exp_file', type=str, default='yolox_exps/example/mot/yolox_s_mix_det.py')
    # weight file path of deteciton model
    parser.add_argument('-DW', '--det_weightfile', default='yolox_weight/bytetrack_s_NTU.pth.tar')
    parser.add_argument('-DT', '--det_trtfile', default='yolox_weight/bytetrack_s_NTU_trt.pth')
    # height and weight of input streaming
    # yolox_m (800, 1440)
    # yolox_s (608, 1088)
    parser.add_argument('--height',type=int, default=608)
    parser.add_argument('--width' ,type=int, default=1088)

    """ReID Setting """
    # config file path of reid model
    parser.add_argument('--configfile', default='reid_weight/cheat.yaml')
    parser.add_argument('-RW', '--reid_weightfile', default='reid_weight/cheat.pth')
    parser.add_argument('-RT', '--reid_trtfile', default='reid_weight/cheat_trt.pth')

    """ MCT Setting """
    parser.add_argument('--mask' , action='store_true') # region_mask
    parser.add_argument('--window' , action='store_true') # traveling time window
    parser.add_argument('--overlap' , action='store_true') # time overlapping check
    parser.add_argument('--exchange_interval' , type=int, default=180)

    """Output setting"""
    parser.add_argument('-V', '--visualize', action='store_true')
    parser.add_argument('-W', '--write', action='store_true')
    parser.add_argument('--write_dir', default='text_output')
    parser.add_argument('--print_interval', type=int, default=500)
    args = parser.parse_args()

    # setting 
    setting = json.load(open(args.json))

    setup_seed(20221219)

    if args.det and args.reid and args.mct: #----------Video + SCT + ReID + MCT----------#:
        # create threads
        yolo_streamer  = create_yolo_streamer(setting, args)
        track_streamer = create_track_streamer(setting, args)
        MCT_streamer = camera_MCT.Camera('MCT', setting, args)
        
        # start threads
        if args.trt:
            for i, d in enumerate(setting['client_number']):
                yolo_streamer[i].threads[('yolo', d)].start()
                track_streamer[i].threads[('track', d)].start()
                while not ('yolo', d) in yolo_streamer[i].detectors and not ('track', d) in track_streamer[i].trackers and not ('track', d) in track_streamer[i].extractors:
                    time.sleep(0)
            MCT_streamer.threads[('MCT', setting['host_number'])].start()
        else:
            for i, d in enumerate(setting['client_number']):
                while not ('yolo', d) in yolo_streamer[i].detectors and not ('track', d) in track_streamer[i].trackers and not ('track', d) in track_streamer[i].extractors:
                    time.sleep(0)
                yolo_streamer[i].threads[('yolo', d)].start()
                track_streamer[i].threads[('track', d)].start()
            MCT_streamer.threads[('MCT', setting['host_number'])].start()
        
        # get the results of threads
        get_MCT_results(args, setting, MCT_streamer, 'MCT', args.visualize, args.write)

    elif args.det and args.reid: #----------Video + SCT + ReiD----------#
        # create threads
        yolo_streamer  = create_yolo_streamer(setting, args)
        track_streamer = create_track_streamer(setting, args)
        
        # start threads
        if args.trt:
            for i, d in enumerate(setting['client_number']):
                yolo_streamer[i].threads[('yolo', d)].start()
                while not ('yolo', d) in yolo_streamer[i].detectors and not ('track', d) in track_streamer[i].trackers and not ('track', i) in track_streamer[i].extractors:
                    time.sleep(0)
                track_streamer[i].threads[('track', d)].start()
        else:
            for i, d in enumerate(setting['client_number']):
                while not ('yolo', d) in yolo_streamer[i].detectors and not ('track', d) in track_streamer[i].trackers and not ('track', i) in track_streamer[i].extractors:
                    time.sleep(0)
                yolo_streamer[i].threads[('yolo', d)].start()
                track_streamer[i].threads[('track', d)].start()
        
        # get the results of threads
        get_track_results(args, setting, track_streamer, 'track', args.visualize, args.write)

    elif args.det: #----------Video + SCT----------#
        # create threads
        yolo_streamer  = create_yolo_streamer(setting, args)
        
        # start threads
        if args.trt:
            for i, d in enumerate(setting['client_number']):
                yolo_streamer[i].threads[('yolo', d)].start()
                while not ('yolo', d) in yolo_streamer[i].detectors:
                    time.sleep(0)
        else:
            for i, d in enumerate(setting['client_number']):
                while not ('yolo', d) in yolo_streamer[i].detectors:
                    time.sleep(0)
                yolo_streamer[i].threads[('yolo', d)].start()
        
        # get the results of threads
        get_det_results(args, setting, yolo_streamer, 'yolo', args.visualize, args.write)

    

    
    
