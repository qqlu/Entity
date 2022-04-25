import cv2
import sys

im = cv2.imread(sys.argv[1])
if len(im.shape) > 2:
    im = im.sum(2)
    im = (im > 0).astype('uint8') * 255

    cv2.imwrite(sys.argv[1], im)

