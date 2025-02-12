import os, sys, argparse, glob
import cv2
import numpy as np 

# mask: [0,0,128]->[0,0,0], background: [0,0,0]->[255,255,255]


def transfer_mask(INPUT_PATH, OUTPUT_PATH):
    img = cv2.imread(INPUT_PATH)
    fn = INPUT_PATH.split('/')[-1].split('.')[0]

    img[img[:,:,2]==0] = [255, 255, 255]
    img[img[:,:,2]==128] = [0, 0, 0]

    cv2.imwrite(os.path.join(OUTPUT_PATH, '{}_roi.jpg'.format(fn)), img)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        help='input mask (image or directory)')
    parser.add_argument('--output_dir', default='./',
                        help='output path')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # directory
    if os.path.isdir(args.input):
        img_list = sorted(glob.glob(os.path.join(args.input, '*.png')))
        for img in img_list:
            transfer_mask(img, args.output_dir)

    # single image
    else:
        transfer_mask(args.input, args.output_dir)