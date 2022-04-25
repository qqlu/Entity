import os
import sys

import cv2
import numpy as np

import h5py

classes = {
    'aeroplane': 1, 
    'bicycle': 2, 
    'bird': 3, 
    'boat': 4, 
    'bottle': 5, 
    'bus': 6, 
    'car': 7, 
    'cat': 8, 
    'chair': 9, 
    'cow': 10, 
    'diningtable': 11, 
    'dog': 12, 
    'horse': 13, 
    'motorbike': 14, 
    'person': 15, 
    'pottedplant': 16, 
    'sheep': 17, 
    'sofa': 18, 
    'train': 19, 
    'tv': 20, 
}

root = sys.argv[1]

im_list = os.listdir(root)
im_list = [f for f in im_list if '_im.jpg' in f]

for im_name in im_list:
    im = cv2.imread(os.path.join(root, im_name))
    h, w, _ = im.shape
    print(h, w)

    mat_name = im_name.replace('_im.jpg', '.mat')

    with h5py.File(os.path.join(root, mat_name), 'r') as mat:

        seg = mat['data_obj']['mask_data']
        seg = np.array(seg).T

        for k, v in classes.items():
            if k in mat_name:
                selected_class = v
                print(mat_name, ', Selected: ', k, v)
                break

        seg_class = (seg==selected_class).astype('float32')
        seg_class = cv2.resize(seg_class, (w, h), interpolation=cv2.INTER_CUBIC)

        seg_class = (seg_class>0.5).astype('uint8') * 255

        seg_name = im_name.replace('_im.jpg', '_seg.png')
        cv2.imwrite(os.path.join(root, seg_name), seg_class)
