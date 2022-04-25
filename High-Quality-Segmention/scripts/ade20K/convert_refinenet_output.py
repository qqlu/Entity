import os
import sys

import cv2
import numpy as np

import h5py

im_root = sys.argv[1]
mask_root = sys.argv[2]

im_list = os.listdir(im_root)
im_list = [f for f in im_list]

for im_name in im_list:
    im = cv2.imread(os.path.join(im_root, im_name))
    h, w, _ = im.shape
    print(h, w)

    mat_name = im_name.replace('.jpg', '.mat')
    with h5py.File(os.path.join(mask_root, mat_name), 'r') as mat:

        seg = mat['data_obj']['mask_data']
        seg = np.array(seg).T

        seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_LINEAR)

        seg_name = im_name.replace('.jpg', '.png')
        cv2.imwrite(os.path.join(mask_root, seg_name), seg)
