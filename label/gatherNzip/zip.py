import os
import argparse
from glob import glob
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('CAM',
                    help='cam name')
args = parser.parse_args()

cam = args.CAM

print(f'Processing with {cam}')

cls_txt = os.path.join('../text_generation/split', cam, 'classes.txt')
for d in tqdm(os.listdir(os.path.join('./', cam))):
    # mv classes.txt
    tgt = os.path.join('./', cam, d)
    cls_tgt = os.path.join(tgt, 'classes.txt')
    os.system(f'cp {cls_txt} {cls_tgt}')

    # zip
    os.system(f'zip -r ./{cam}/{d}.zip {tgt}')
    # print(f'zip -r ./{cam}/{d}.zip {tgt}')