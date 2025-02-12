# encoding: utf-8
"""
@author:  sherlock (changed by Nir)
@contact: sherlockliao01@gmail.com
"""


import glob
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class NTUMTMC(ImageDataset):
    _junk_pids = [0, -1]
    dataset_dir = ''
    dataset_url = '' 
    dataset_name = "NTUMTMC"

    def __init__(self, root='datasets', **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'NTUMTMC-ReID')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"NTUMTMC-ReID".')

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')

        self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        self.extra_gallery = False

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]

        self.check_before_run(required_files)

        train = lambda: self.process_dir(self.train_dir)
        query = lambda: self.process_dir(self.query_dir, is_train=False)
        gallery = lambda: self.process_dir(self.gallery_dir, is_train=False)
                        #   (self.process_dir(self.extra_gallery_dir, is_train=False) if self.extra_gallery else [])

        super(NTUMTMC, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        # print('#'*30)
        # print(dir_path)
        # print('#'*30)

        img_paths = glob.glob(osp.join(dir_path, '*.bmp'))

        data = []
        for img_path in img_paths:
            pid = int(img_path.split('/')[-1].split('_')[0])
            if pid == -1:
                continue  # junk images are just ignored
            camid = int(img_path.split('/')[-1].split('_')[1][3:])
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))
        # print(len(data))
        return data
