import os
import sys

import cv2

root = sys.argv[1]

mask_list = os.listdir(root)

for mask_name in mask_list:
    mask = cv2.imread(os.path.join(root, mask_name))
    cv2.imwrite(os.path.join(root, mask_name), mask+1)
