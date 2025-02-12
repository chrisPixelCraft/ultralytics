import os
import argparse
from glob import glob
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('CAM',
                    help='cam name')
args = parser.parse_args()
cam = args.CAM

split_number = 9000 # 15 fps for 10 mins

print(f'Processing with {cam}')
img_pth = os.path.join('../image', cam, '*.jpg')
img_list = sorted(glob(img_pth))
assert len(img_list) == 54000

# create directory
os.makedirs('./{}'.format(cam), exist_ok=True)
for i in range(1,7):
    os.makedirs(os.path.join('./', cam, '{}_0{}'.format(cam, i)), exist_ok=True)

# mv
for img in tqdm(img_list):
    img_num = int(img.split('/')[-1].split('.')[0][3:]) - 1
    img_name = img.split('/')[-1]
    dir_num = (img_num // split_number) + 1
    img_src = img
    img_dst = os.path.join('./', cam, '{}_0{}'.format(cam, dir_num), img_name)

    txt_src = img.replace('.jpg', '.txt').replace('image', 'text_generation/split') 
    txt_name = txt_src.split('/')[-1]
    txt_dst = os.path.join('./', cam, '{}_0{}'.format(cam, dir_num), txt_name)

    os.system('cp {} {}'.format(img_src, img_dst))
    if os.path.isfile(txt_src):
        os.system('cp {} {}'.format(txt_src, txt_dst))