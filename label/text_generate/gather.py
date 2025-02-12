import os, sys, argparse, glob
import cv2
import numpy as np 
from tqdm import trange


# split a video tracking results into per image text file of LabelImg format
# LabelImg format: ID, cx, cy, w, h & classes.txt
def split(matrix, id_list, output_dir):
    # output text file
    for i in trange(matrix.shape[0]):
        fid, pid, cx, cy, w, h = matrix[i]
        fid, pid = int(fid)+1, int(pid)
        cx, cy, w, h = float(cx), float(cy), float(w), float(h)

        fn = os.path.join(output_dir, 'img{}.txt'.format(str(fid).zfill(6)))

        with open(fn, "a") as fout:
            fout.write("{:d} {:6f} {:.6f} {:.6f} {:.6f}\n".format(id_list.index(pid), cx, cy, w, h))
    # output classes.txt
    with open(os.path.join(output_dir, 'classes.txt'), "w") as fout:
        for i in id_list:
            fout.write("{:d}\n".format(i))
    

def transfer(args, tracks):
    tracks[:, 2] = (tracks[:, 2] + 0.5*tracks[:, 4]) / args.W
    tracks[:, 3] = (tracks[:, 3] + 0.5*tracks[:, 5]) / args.H
    tracks[:, 4] = tracks[:, 4] / args.W
    tracks[:, 5] = tracks[:, 5] / args.H
    return tracks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_pth', default='/home/chenyukai/NTU-dataset/label/ByteTrack',
                        help='input path to tracking text file')
    parser.add_argument('--output_pth', default='./',
                        help='output path')
    parser.add_argument('--camid', 
                        help='camera ID')
    parser.add_argument('--H', type=int, default=1520,
                        help='output path')
    parser.add_argument('--W', type=int, default=2704,
                        help='output path')
    
    args = parser.parse_args()
    # directory
    CAM = args.camid
    print('#'*30)
    print(f'Processing with {CAM}...')
    TXT = os.path.join(args.input_pth, f'{CAM}_results.txt')
    DIR = os.path.join(args.output_pth, CAM)
    os.makedirs(DIR, exist_ok=True)
    tracks = np.loadtxt(TXT, delimiter=',', dtype='float', usecols=(0,1,2,3,4,5))
    tracks = transfer(args, tracks)
    id_list = np.unique(tracks[:, 1]).astype(int).tolist()
    split(tracks, id_list, DIR)