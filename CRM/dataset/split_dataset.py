import os
from torch.utils.data.dataset import Dataset
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import progressbar

from dataset.make_bb_trans import *
import util.boundary_modification as boundary_modification

seg_normalization = transforms.Normalize(
                mean=[0.5],
                std=[0.5]
            )

class SplitTransformDataset(Dataset):
    def __init__(self, root, in_memory=False, need_name=False, perturb=True, img_suffix='_im.jpg'):
        self.root = root
        self.need_name = need_name
        self.in_memory = in_memory
        self.perturb = perturb
        self.img_suffix = img_suffix

        imgs = os.listdir(self.root)

        self.im_list = [im for im in imgs if '_im' in im]
        self.gt_list = [im for im in imgs if '_gt' in im]

        print('%d ground truths found' % len(self.gt_list))

        if perturb:
            # Make up some transforms
            self.im_transform = transforms.Compose([
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                transforms.RandomGrayscale(),
                # transforms.Resize((224, 224), interpolation=Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            # Make up some transforms
            self.im_transform = transforms.Compose([
                # transforms.Resize((224, 224), interpolation=Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

        self.gt_transform = transforms.Compose([
            # transforms.Resize((224, 224), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

        self.seg_transform = transforms.Compose([
            # transforms.Resize((224, 224), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            seg_normalization,
        ])

        # Map ground truths to images
        self.gt_to_im = []
        for im in self.gt_list:
            # Find the second last underscore and remove from there to get basename
            end_idx = im[:-8].rfind('_')
            self.gt_to_im.append(im[:end_idx])

        if self.in_memory:
            self.images = {}
            for im in progressbar.progressbar(self.im_list):
                # Remove img_suffix, indexing might be faster but well..
                self.images[im.replace(self.img_suffix, '')] = Image.open(self.join_path(im)).convert('RGB')
            print('Images loaded to memory.')

            self.gts = []
            for im in progressbar.progressbar(self.gt_list):
                self.gts.append(Image.open(self.join_path(im)).convert('L'))
            print('Ground truths loaded to memory')

            if not self.perturb:
                self.segs = []
                for im in progressbar.progressbar(self.gt_list):
                    self.segs.append(Image.open(self.join_path(im.replace('_gt', '_seg'))).convert('L'))
                print('Input segmentations loaded to memory')

    def join_path(self, im):
        return os.path.join(self.root, im)

    def __getitem__(self, idx):
        if self.in_memory:
            gt = self.gts[idx]
            im = self.images[self.gt_to_im[idx]]
            if not self.perturb:
                seg = self.segs[idx]
        else:
            gt = Image.open(self.join_path(self.gt_list[idx])).convert('L')
            im = Image.open(self.join_path(self.gt_to_im[idx]+self.img_suffix)).convert('RGB')
            if not self.perturb:
                seg = Image.open(self.join_path(self.gt_list[idx].replace('_gt', '_seg'))).convert('L')

        # Get bounding box from ground truth
        if self.perturb:
            im_width, im_height = gt.size # PIL inverted width/height
            try:
                bb_pos = get_bb_position(np.array(gt))
                bb_pos = mod_bb(*bb_pos, im_height, im_width, 0.1, 0.1)
                rmin, rmax, cmin, cmax = scale_bb_by(*bb_pos, im_height, im_width, 0.25, 0.25)
            except:
                print('Failed to get bounding box')
                rmin = cmin = 0
                rmax = im_height
                cmax = im_width
        else:
            im_width, im_height = seg.size # PIL inverted width/height
            try:
                bb_pos = get_bb_position(np.array(seg))
                rmin, rmax, cmin, cmax = scale_bb_by(*bb_pos, im_height, im_width, 0.25, 0.25)
            except:
                print('Failed to get bounding box')
                rmin = cmin = 0
                rmax = im_height
                cmax = im_width

        # If no GT then we ha ha ha
        if (rmax-rmin==0 or cmax-cmin==0):
            print('No GT, no cropping is done.')
            crop_lambda = lambda x: x
        else:
            crop_lambda = lambda x: transforms.functional.crop(x, rmin, cmin, rmax-rmin, cmax-cmin)

        im = crop_lambda(im)
        gt = crop_lambda(gt)

        if self.perturb:
            iou_max = 1.0
            iou_min = 0.7
            iou_target = np.random.rand()*(iou_max-iou_min) + iou_min
            seg = boundary_modification.modify_boundary((np.array(gt)>0.5).astype('uint8')*255, iou_target=iou_target)
            seg = Image.fromarray(seg)
        else:
            seg = crop_lambda(seg)

        im = self.im_transform(im)
        gt = self.gt_transform(gt)
        seg = self.seg_transform(seg)

        if self.need_name:
            return im, seg, gt, os.path.basename(self.gt_list[idx][:-7])
        else:
            return im, seg, gt

    def __len__(self):
        return len(self.gt_list)
