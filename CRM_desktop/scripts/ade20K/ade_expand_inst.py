import os
import sys
from shutil import copyfile

from PIL import Image
import numpy as np
import cv2

import progressbar

img_path = sys.argv[1]
gt_path = sys.argv[2]
seg_path = sys.argv[3]
out_path = sys.argv[4]

seg_list = os.listdir(seg_path)

os.makedirs(out_path, exist_ok=True)

def get_disk_kernel(size):
    r = size // 2

    y, x = np.ogrid[-r:size-r, -r:size-r]
    mask = x*x + y*y <= r*r

    array = np.zeros((size, size)).astype('uint8')
    array[mask] = 1
    return array

disc_kernel = get_disk_kernel(10)
for im_idx, seg in enumerate(progressbar.progressbar(seg_list)):
    name = os.path.basename(seg)[:-4]

    im_full_path = os.path.join(img_path, name+'.jpg')

    seg_img = Image.open(os.path.join(seg_path, seg)).convert('L')
    gt_img = Image.open(os.path.join(gt_path, seg)).convert('L')

    seg_img = np.array(seg_img)
    gt_img = np.array(gt_img)

    seg_classes = np.unique(seg_img)
    gt_classes = np.unique(gt_img)

    all_classes = np.union1d(seg_classes, gt_classes)

    seg_written = False
    for c in all_classes:
        class_seg = (seg_img == c).astype('uint8')
        class_gt  = (gt_img == c).astype('uint8')

        # Remove small overall parts
        if class_seg.sum() <= 32*32:
            continue

        class_seg_dilated = cv2.dilate(class_seg, disc_kernel)
        _, components_map = cv2.connectedComponents(class_seg_dilated, connectivity=8)
        components = np.unique(components_map)
        components = components[components!=0] # Remove zero, the background class

        for comp in components:
            comp_map = (components_map == comp).astype('uint8')
            # Similar to a closing operator, we don't want to include extra regions
            comp_map = cv2.erode(comp_map, disc_kernel)

            if comp_map.sum() <= 32*32:
                continue

            # Masking
            comp_seg = (comp_map * class_seg) * 255
            comp_gt  = (comp_map * class_gt) * 255

            seg_written = True
            cv2.imwrite(os.path.join(out_path, name + '_%d.%d_seg.png' % (c, comp)), comp_seg)
            cv2.imwrite(os.path.join(out_path, name + '_%d.%d_gt.png' % (c, comp)), comp_gt)

    if seg_written:
        copyfile(im_full_path, os.path.join(out_path, name + '_im.jpg'))
