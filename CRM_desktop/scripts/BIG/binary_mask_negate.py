import cv2
import sys

imA = cv2.imread(sys.argv[1])
imB = cv2.imread(sys.argv[2])
out = sys.argv[3]


imC = imA - imB
cv2.imwrite(out, imC)

