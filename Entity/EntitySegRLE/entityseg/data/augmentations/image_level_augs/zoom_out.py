import copy
import math
import torch
import random
import numpy as np
from detectron2.structures.boxes import Boxes
from detectron2.structures.masks import PolygonMasks
from detectron2.augmentations.image_level_augs.scale_jitter import scale_jitter
from detectron2.augmentations.vis import _vis


class Zoom_out(object):
    def __init__(self, ratio=1.0, img_pool_size=10, iou_threshold=0.5, size_divisible=2):
        self.ratio = ratio
        self.img_pool = []
        self.img_pool_size = img_pool_size
        self.iou_threshold =iou_threshold
        self.size_divisible = size_divisible

    def __call__(self, tensor, target):
        if self.ratio >= 1.0:
            return tensor, target

        self.img_pool.append({'tensor': tensor, 'target': target})

        if len(self.img_pool) > self.img_pool_size:
            self.img_pool.pop(0)

        if len(self.img_pool) < 4:
            return tensor, target

        use_mask = ('gt_masks' in target)

        bbox = target['gt_boxes']
        classes = target['gt_classes']
        masks = target['gt_masks'] if use_mask else None

        c, h, w = tensor.shape
        h = int(math.ceil(h / self.size_divisible) * self.size_divisible)
        w = int(math.ceil(w / self.size_divisible) * self.size_divisible)

        new_h, new_w = int(self.ratio * h), int(self.ratio * w)
        in_tensor, in_bbox, in_mask = scale_jitter(tensor, bbox, self.ratio, (new_h, new_w), masks)

        pad_imgs = random.sample(self.img_pool, 3)
        pad_tensors, pad_bboxes, pad_masks = [], [], []
        for img in pad_imgs:
            pad_tensor, pad_bbox, pad_mask = scale_jitter(img['tensor'], img['target']['gt_boxes'], self.ratio, (new_h, new_w), img['target']['gt_masks'] if use_mask else None)
            pad_tensors.append(pad_tensor)
            pad_bboxes.append(pad_bbox)
            pad_masks.append(pad_mask)

        crop_boxes = [(new_h, w-new_w), (h-new_h, new_w), (h-new_h, w-new_w)]

        tensor_out = in_tensor.new(*(c, h, w)).zero_()
        tensor_out[:c, :new_h, :new_w].copy_(in_tensor)
        tensor_out[:c, :new_h, new_w:].copy_(pad_tensors[0][:c, :crop_boxes[0][0], :crop_boxes[0][1]])
        tensor_out[:c, new_h:, :new_w].copy_(pad_tensors[1][:c, :crop_boxes[1][0], :crop_boxes[1][1]])
        tensor_out[:c, new_h:, new_w:].copy_(pad_tensors[2][:c, :crop_boxes[2][0], :crop_boxes[2][1]])

        crop_bboxes, crop_classes, crop_masks = [], [], []
        for i, pad_bbox in enumerate(pad_bboxes):
            crop_bbox = copy.deepcopy(pad_bbox)
            crop_bbox.clip(crop_boxes[i])
            ious = crop_bbox.area() / pad_bbox.area()
            inds = ious >= self.iou_threshold
            crop_bbox = crop_bbox[inds]
            crop_bboxes.append(crop_bbox)
            crop_classes.append(pad_imgs[i]['target']['gt_classes'][inds])
            if use_mask:
                crop_masks.append([mask for j, mask in enumerate(pad_masks[i]) if inds[j]])

        offsets_box = [torch.Tensor([0.0,0.0,0.0,0.0]), torch.Tensor([new_w, 0.0, new_w, 0.0]), torch.Tensor([0.0, new_h, 0.0, new_h]), torch.Tensor([new_w, new_h, new_w, new_h])]
        offsets_mask = [[0.0, 0.0],  [0.0, new_w], [new_h, 0], [new_h, new_w]]
        bbox_out = Boxes(torch.cat([target.tensor + offsets_box[i] for i, target in enumerate([in_bbox] + crop_bboxes)], dim=0))
        classes_out = torch.cat([classes] + crop_classes, dim=0)
        target_out = {'gt_boxes': bbox_out, 'gt_classes': classes_out}

        if use_mask:
            masks_out = []
            for i, crop_mask in enumerate([in_mask]+crop_masks):
                mask_out = []
                for polys in crop_mask:
                    poly_out = []
                    for poly in polys:
                        poly_new = copy.deepcopy(poly)
                        poly_new[0::2] = poly_new[0::2] + offsets_mask[i][1]
                        poly_new[1::2] = poly_new[1::2] + offsets_mask[i][0]
                        poly_out.append(poly_new)
                    mask_out.append(poly_out)

                masks_out += mask_out
            masks_out = PolygonMasks(masks_out)
            target_out['gt_masks'] = masks_out

        return tensor_out, target_out
