import os, sys, argparse, glob
import cv2
import numpy as np 

"""
remove the bounding box whose bottom-middle point is in the region of interest (ROI) in tracking results
"""
def xywh2xyxy(pts):
    if pts.ndim == 2: # multi points
        pts[:, 2] = pts[:, 0] + pts[:, 2]
        pts[:, 3] = pts[:, 1] + pts[:, 3]
    elif pts.ndim == 1: # single point
        pts[2] = pts[0] + pts[2]
        pts[3] = pts[1] + pts[3]
    return pts
def xyxy2mb(pts): # xyxy 2 middle-bottom point
    if pts.ndim == 2: # multi points
        x = (pts[:, 2] + pts[:, 0])/2
        y = pts[:, 3]
    elif pts.ndim == 1: # single point
        x = (pts[2] + pts[0])/2
        y = pts[3] 
    return np.stack((x,y), axis=1)
def clip(pts, w, h):
    if pts.ndim == 2: # multi points
        pts[:, 0] = np.clip(pts[:, 0], 0, w-1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h-1)
        pts[:, 2] = np.clip(pts[:, 2], 0, w-1)
        pts[:, 3] = np.clip(pts[:, 3], 0, h-1)
    elif pts.ndim == 1: # single point
        pts[0] = np.clip(pts[0], 0, w-1)
        pts[1] = np.clip(pts[1], 0, h-1)
        pts[2] = np.clip(pts[2], 0, w-1)
        pts[3] = np.clip(pts[3], 0, h-1)
    return pts

def trackNroi(args):
    CAM = args.camid
    TEXT = os.path.join(args.input_dir, CAM, 'results.txt')
    MASK = os.path.join(args.input_msk, f'{CAM}_15_roi.jpg')
    OUTPUT_PATH = os.path.join(args.output_dir, f'{CAM}_results.txt')
    print('#'*30)
    print(f'Processing with {CAM}....')

    mask = cv2.imread(MASK)
    mask = mask[:,:,0] > 0
    h, w = mask.shape

    tracks = np.loadtxt(TEXT, delimiter=',')
    old = tracks.shape[0]
    tlwh = tracks.copy()[:,2:6]
    tlwh = xywh2xyxy(tlwh)
    tlwh = clip(tlwh, w, h)
    tlwh = xyxy2mb(tlwh)
    tlwh = np.around(tlwh, decimals=0).astype(np.int32)

    mb = mask[tlwh[:,1], tlwh[:,0]]
    tracks = tracks[mb]
    new = tracks.shape[0]
    print(f'{old} -> {new} bboxes')

    ##################################################
    # fids = tracks[:,0]
    # cnt = 0
    # previous = 0
    # for i in range(1,7):
    #     top = i * 9000 -1 
    #     num = np.count_nonzero(fids <= top)
    #     final = num - previous
    #     print(f' {i}: {final}')
    #     previous = num
    #     cnt += final
    ##################################################

    with open(OUTPUT_PATH, 'w') as f:
        for k in range(tracks.shape[0]):
            fid = int(tracks[k][0])
            pid = int(tracks[k][1])
            tlwh = tracks[k][2:6]
            score = tracks[k][6]

            f.write(
                f"{fid},{pid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{score:.2f},-1,-1,-1\n"
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',
                        help='input text file')
    parser.add_argument('--input_msk',
                        help='input mask file (roi.jpg)')
    parser.add_argument('--output_dir', default='./',
                        help='output text path')
    parser.add_argument('--camid', 
                        help='camera ID')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    trackNroi(args)
