import os
from glob import glob
import numpy as np


dirs = [name for name in sorted(os.listdir('./')) if os.path.isdir(os.path.join('./', name)) ]


with open('bboxes.txt', 'w') as f:
    for d in dirs:
        f.write('='*20+'\n')
        f.write(d+'\n')
        text_file = os.path.join('./', d, 'results.txt')
        fids = np.loadtxt(text_file, delimiter=',', dtype='int', usecols=(0))

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













