import os
from glob import glob
import numpy as np





with open('bboxes.txt', 'w') as f:
    for d in sorted(glob('./afterByte/*.txt')):
        f.write('='*20+'\n')
        f.write(d.split('/')[-1].split('.')[0].split('_')[0]+'\n')
        fids = np.loadtxt(d, delimiter=',', dtype='int', usecols=(0))

        cnt = 0
        previous = 0
        for i in range(1,7):
            top = i * 9000 -1 
            num = np.count_nonzero(fids <= top)
            final = num - previous
            previous = num
            f.write(f' {i}: {final}\n')
            cnt += final
        assert cnt == len(fids)
        f.write(f' Total: {cnt}')
        f.write('\n')













