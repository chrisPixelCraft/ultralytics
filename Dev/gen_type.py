import os, sys, cv2, warnings, argparse, json
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from torchvision.models import resnet50

warnings.filterwarnings('ignore')

def gen_type(args, target):
    os.makedirs(f'./{args.target_dir}/status', exist_ok=True)

    model = resnet50()
    model.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
    model.load_state_dict(torch.load('./reid_weight/Ped_Bike.pth'))
    model.eval()
    model.to('cuda')

    for t in target:
        status = {}
        videopath = f'../../test/{t}/{t}.MP4'
        cap = cv2.VideoCapture(videopath)

        detfile = f'./{args.target_dir}/{t}.txt'
        all_boxes = np.loadtxt(detfile, delimiter=' ', usecols=(0,1,2,3,4,5,6,7)) # c, p, f, x, y, w, h, conf

        fid = 0
        while True:
            ret, frame = cap.read()
            if ret:
                fid += 1
                frame = cv2.resize(frame, (1440, 800))
            else: break
            
            boxes = all_boxes[all_boxes[:, 2]==fid]
            
            # add detections to batched images
            if boxes.shape[0] != 0:
                for idx, box in enumerate(boxes):
                    x,y,w,h = tuple(map(int, box[3:7]))
                    p = int(box[1])
                    tgt = cv2.resize(frame[y:y+h+1, x:x+w+1, :], (256,512))
                    imgs = torch.from_numpy(np.array(tgt)).float().to('cuda')
                    imgs = imgs.permute(2,0,1).unsqueeze(0)
                    outputs = model(imgs)
                    print(outputs)
                    _, predicted = outputs.max(1)
                    print(predicted)
                    if predicted == 0:
                        sys.exit()
                    

                    if p not in status.keys():
                        status[p] = []
                    status[p].append(predicted[0].item())

        data = json.dumps(status, indent=2)
        with open(f'./{args.target_dir}/status/{t}.json', 'w', newline='\n') as f:
            f.write(data)

def vote(args, target):
    for t in target:
        status = {}
        data = json.load(open(f'./{args.target_dir}/status/{t}.json'))
        for pid, stat in tqdm(data.items()):
            num = len(stat)
            ped_num = stat.count(1)
            bik_num = stat.count(0)
            assert num == ped_num + bik_num

            if ped_num >= 0.95 * num:
                status[pid] = [1, 1]
            elif bik_num >= 0.95 * num:
                status[pid] = [0, 0]
            else:
                status[pid] = []
                select = int(num * 0.15)
                head = stat[:select+1]
                tail = stat[-select:]
                if head.count(1) > head.count(0):
                    status[pid].append(1)
                else:
                    status[pid].append(0)
                if tail.count(1) > tail.count(0):
                    status[pid].append(1)
                else:
                    status[pid].append(0)

        final = json.dumps(status, indent=2)
        with open(f'./{args.target_dir}/status_final/{t}.json', 'w', newline='\n') as f:
            f.write(final)
    





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir', default='v17_output_v5')
    args = parser.parse_args()

    target = ['Cam1', 'Cam2', 'Cam3', 'Cam4', 'Cam5', 'Cam6', 'Cam7', 'Cam8', 'Cam9', 'Cam10', 'Cam11']
    print(target)
    # gen_type(args, target)
    vote(args, target)