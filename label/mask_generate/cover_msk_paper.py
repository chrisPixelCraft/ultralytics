import os, sys, argparse, glob
import cv2
import numpy as np 
from tqdm import tqdm

"""
plot the mask on frames for labeling convinience
"""

def cover_msk(args):
    INPUT_PTH = args.input_dir
    MASK = args.input_msk
    OUTPUT_PTH = args.output_dir
    BLEND = 0.45
    filename = INPUT_PTH.split('/')[-1].split('.')[0]

    msk = cv2.imread(MASK, 0)
    msk = msk == 0 # black
    cover = np.zeros((msk.shape[0], msk.shape[1], 3))
    cover[msk, :] = [0, 0 ,255]

    img = cv2.imread(INPUT_PTH)
    # img = img * (1-BLEND) + cover * BLEND
    img = img  + cover * BLEND
    img = np.clip(img, 0, 255)
    img = cv2.resize(img, (676, 380))
    cv2.imwrite(os.path.join(OUTPUT_PTH, f'{filename}_msk.jpg'), img)

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',
                        help='where the frames are located')
    parser.add_argument('--input_msk',
                        help='input mask file (roi.jpg)')
    parser.add_argument('--output_dir', default='./',
                        help='output path')
    
    args = parser.parse_args()
    cover_msk(args)
