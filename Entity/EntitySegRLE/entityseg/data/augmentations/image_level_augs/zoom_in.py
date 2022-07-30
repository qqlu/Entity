import copy
import torch
import numpy as np
from detectron2.structures.masks import PolygonMasks
from detectron2.augmentations.image_level_augs.scale_jitter import scale_jitter, _crop_boxes


class Zoom_in(object):
    def __init__(self, ratio=1.0, iou_threshold=0.5):
        self.ratio = ratio
        self.iou_threshold = iou_threshold

    def __call__(self, tensor, target):
        if self.ratio <= 1.0:
            return tensor, target
        bbox = target['gt_boxes']
        classes = target['gt_classes']

        h, w = tensor.shape[1], tensor.shape[2]
        new_h, new_w = int(h * self.ratio), int(w * self.ratio)

        use_mask = ('gt_masks' in target)

        original_bbox = copy.deepcopy(bbox)
        enlarged_tensor, enlarged_bbox, enlarged_mask = scale_jitter(tensor, bbox, self.ratio, mask=target['gt_masks'] if use_mask else None)
        original_enlarged_bbox = copy.deepcopy(enlarged_bbox)

        crop_x, crop_y = np.random.randint(0, new_h - h), np.random.randint(0, new_w - w)
        crop_box = (crop_y, crop_x, crop_y + w, crop_x + h)
        cropped_tensor = enlarged_tensor[:, crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
        cropped_bbox = _crop_boxes(enlarged_bbox, crop_box)
        ious = cropped_bbox.area() / original_enlarged_bbox.area()
        inds = ious >= self.iou_threshold
        cropped_bbox = cropped_bbox[inds]
        cropped_classes = classes[inds]

        if len(cropped_bbox) > 0:
            tensor_out = cropped_tensor
            target_out = {'gt_boxes': cropped_bbox, 'gt_classes': cropped_classes}
        else:
            tensor_out = tensor
            target_out = {'gt_boxes': original_bbox, 'gt_classes': classes}

        if use_mask:
            cropped_masks = []
            for j, polys in enumerate(enlarged_mask):
                poly_out = []
                for poly in polys:
                    if len(cropped_bbox) > 0:
                        poly_new = copy.deepcopy(poly)
                        poly_new[0::2] = poly_new[0::2] - crop_y
                        poly_new[1::2] = poly_new[1::2] - crop_x
                        poly_out.append(poly_new)
                    else:
                        poly_out.append(poly)
                if len(cropped_bbox) == 0:
                    cropped_masks.append(poly_out)
                elif inds[j]:
                    cropped_masks.append(poly_out)
            target_out['gt_masks'] = PolygonMasks(cropped_masks)
            if len(target_out['gt_boxes']) > len(target_out['gt_masks']):
                from IPython import embed; embed()

        return tensor_out, target_out
