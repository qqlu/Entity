import copy
import torch
from detectron2.structures.boxes import Boxes
from detectron2.structures.masks import PolygonMasks

def scale_jitter(tensor, target, jitter_factor, jitter_size=None, mask=None):
    if jitter_size is None:
        _, h, w = tensor.shape
        new_h, new_w = int(h * jitter_factor), int(w * jitter_factor)
        jitter_factor_x = jitter_factor_y = jitter_factor
    else:
        new_h, new_w = jitter_size
        _, h, w = tensor.shape
        jitter_factor_y, jitter_factor_x = new_h/h, new_w/w
    tensor_out = torch.nn.functional.interpolate(tensor.unsqueeze(0), size=(new_h, new_w), mode='nearest').squeeze(0)
    target_out = copy.deepcopy(target)
    target_mask = []

    if 'gt_masks' in target:
        mask = target['gt_masks']

    if mask is not None:
        for polys in mask.polygons:
            new_polys = copy.deepcopy(polys)
            for p in new_polys:
                p[0::2] *= jitter_factor_x
                p[1::2] *= jitter_factor_y
            target_mask.append(new_polys)

    if isinstance(target, dict):
        target_out['gt_boxes'].scale(jitter_factor_x, jitter_factor_y)
        if 'gt_masks' in target:
            target_out['gt_masks'] = PolygonMasks(target_mask)
    elif isinstance(target, Boxes):
        target_out.scale(jitter_factor_x, jitter_factor_y)
    else:
        raise ValueError('Unsupported target %s'%str(target))

    return tensor_out, target_out, target_mask


def _crop_boxes(gt_boxes, crop_box):
    xmin, ymin, xmax, ymax = gt_boxes.tensor.split(1, dim=-1)

    assert torch.isfinite(gt_boxes.tensor).all(), "Box tensor contains infinite or NaN!"

    w, h = crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]
    cropped_xmin = (xmin - crop_box[0]).clamp(min=0, max=w)
    cropped_ymin = (ymin - crop_box[1]).clamp(min=0, max=h)
    cropped_xmax = (xmax - crop_box[0]).clamp(min=0, max=w)
    cropped_ymax = (ymax - crop_box[1]).clamp(min=0, max=h)

    cropped_box = torch.cat((cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1)
    gt_boxes.tensor = cropped_box
    return gt_boxes
