import os
import sys

import cv2
import numpy as np

root = sys.argv[1]

im_list = os.listdir(root)
im_list = [f for f in im_list if '_im.png' in f]

for im_name in im_list:
    im = cv2.imread(os.path.join(root, im_name))
    h, w, _ = im.shape
    print(h, w)

    seg_name = im_name.replace('_im.png', '_seg.png')
    seg = cv2.imread(os.path.join(root, seg_name))

    selected_class = int(im_name[-9:-7])
    print(np.unique(seg), selected_class)

    seg_class = (seg==selected_class).astype('float32')
    seg_class = cv2.resize(seg_class, (w, h), interpolation=cv2.INTER_CUBIC)

    seg_class = (seg_class>0.5).astype('uint8') * 255

    cv2.imwrite(os.path.join(root, seg_name), seg_class)
