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

    img_list = sorted(glob.glob(os.path.join(INPUT_PTH, '*.jpg')))

    msk = cv2.imread(MASK, 0)
    msk = msk == 0 # black
    cover = np.zeros((msk.shape[0], msk.shape[1], 3))
    cover[msk, :] = [0, 0 ,255]

    for img_pth in tqdm(img_list):
        fn = img_pth.split('/')[-1].split('.')[0]
        img = cv2.imread(img_pth)
        # img = img * (1-BLEND) + cover * BLEND
        img = img  + cover * BLEND
        img = np.clip(img, 0, 255)
        cv2.imwrite(os.path.join(OUTPUT_PTH, '{}.jpg'.format(fn)), img)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',
                        help='where the frames are located')
    parser.add_argument('--input_msk',
                        help='input mask file (roi.jpg)')
    parser.add_argument('--output_dir', default='./',
                        help='output path')
    
    args = parser.parse_args()
    print('Input" {}'.format(args.input_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    cover_msk(args)
