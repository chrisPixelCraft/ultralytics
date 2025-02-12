import os
import argparse
from glob import glob
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('CAM',
                    help='cam name')
args = parser.parse_args()


split_number = 9000 
cam = args.CAM

print(f'Processing with {cam}')
img_src = len(glob(os.path.join('../image', cam, '*.jpg')))
img_dst = len(glob(os.path.join('./', cam, '*/*.jpg')))
assert img_src == 54000
assert img_src == img_dst
print(img_src, img_dst)


txt_src = len(glob(os.path.join('../text_generation/split', cam, '*.txt'))) -1 #classes.txt
txt_dst = len(glob(os.path.join('./', cam, '*/*.txt')))
print(txt_src, txt_dst)
assert txt_src == txt_dst
