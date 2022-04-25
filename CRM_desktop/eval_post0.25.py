import numpy as np
from PIL import Image
import progressbar

from util.compute_boundary_acc import compute_boundary_acc
from util.file_buffer import FileBuffer

from argparse import ArgumentParser
import os
import re


parser = ArgumentParser()

parser.add_argument('--dir', help='Directory with image, gt, and mask')

parser.add_argument('--output', help='Output of temp results',
        default=None)

args = parser.parse_args()

def get_iu(seg, gt):
    intersection = np.count_nonzero(seg & gt)
    union = np.count_nonzero(seg | gt)
    
    return intersection, union 

total_new_i = 0
total_new_u = 0
total_old_i = 0
total_old_u = 0

total_old_correct_pixels = 0
total_new_correct_pixels = 0
total_num_pixels = 0

total_num_images = 0
total_seg_acc = 0
total_mask_acc = 0

small_objects = 0

all_h = 0
all_w = 0
all_max = 0

all_gts = [gt for gt in os.listdir(args.dir) if '_gt.png' in gt]
file_buffer = FileBuffer(os.path.join(args.dir, 'results_post0.25.txt'))

if args.output is not None:
    os.makedirs(args.output, exist_ok=True)

for gt_name in progressbar.progressbar(all_gts):
    
    gt = np.array(Image.open(os.path.join(args.dir, gt_name)
                                ).convert('L'))

    seg = np.array(Image.open(os.path.join(args.dir, gt_name.replace('_gt', '_seg'))
                                ).convert('L'))

    mask_im = Image.open(os.path.join(args.dir, "0.25_"+(gt_name.replace('_gt', '_mask')))
                                ).convert('L')
    mask = seg.copy()
    this_class = int(re.search(r'\d+', gt_name[::-1]).group()[::-1]) - 1

    rmin = cmin = 0
    rmax, cmax = seg.shape

    all_h += rmax
    all_w += cmax
    all_max += max(rmax, cmax)

    mask_h, mask_w = mask.shape
    if mask_h != cmax:
        mask = np.array(mask_im.resize((cmax, rmax), Image.BILINEAR))

    if seg.sum() < 32*32:
        # Reject small objects, just copy input
        small_objects += 1
    else:
        if (cmax==cmin) or (rmax==rmin):
            # Should not happen. Check the input in this case.
            print(gt_name, this_class)
            continue
        class_mask_prob = np.array(mask_im.resize((cmax-cmin, rmax-rmin), Image.BILINEAR))
        mask[rmin:rmax, cmin:cmax] = class_mask_prob

    """
    Compute IoU and boundary accuracy
    """
    gt = gt > 128
    seg = seg > 128
    mask = mask > 128

    old_i, old_u = get_iu(gt, seg)
    new_i, new_u = get_iu(gt, mask)

    total_new_i += new_i
    total_new_u += new_u
    total_old_i += old_i
    total_old_u += old_u

    seg_acc, mask_acc = compute_boundary_acc(gt, seg, mask)
    total_seg_acc += seg_acc
    total_mask_acc += mask_acc
    total_num_images += 1
    
    if args.output is not None:
        gt = Image.fromarray(gt)
        seg = Image.fromarray(seg)
        mask = Image.fromarray(mask)

        gt.save(os.path.join(args.output, gt_name))
        seg.save(os.path.join(args.output, gt_name.replace('_gt.png', '_seg.png')))
        mask.save(os.path.join(args.output, gt_name.replace('_gt.png', '_mask.png')))

new_iou = total_new_i/total_new_u
old_iou = total_old_i/total_old_u
new_mba = total_mask_acc/total_num_images
old_mba = total_seg_acc/total_num_images

file_buffer.write('New IoU  : ', new_iou)
file_buffer.write('Old IoU  : ', old_iou)
file_buffer.write('IoU Delta: ', new_iou-old_iou)

file_buffer.write('New mBA  : ', new_mba)
file_buffer.write('Old mBA  : ', old_mba)
file_buffer.write('mBA Delta: ', new_mba-old_mba)

file_buffer.write('Avg. H+W  : ', (all_h+all_w)/total_num_images)
file_buffer.write('Avg. Max(H,W) : ', all_max/total_num_images)

file_buffer.write('Number of small objects: ', small_objects)
