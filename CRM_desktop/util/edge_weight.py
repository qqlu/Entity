import cv2
import numpy as np

import torch
import torch.nn.functional as F

def edge(masks, thickness=5):
    masks = masks.cpu().detach().numpy().astype(np.uint8)
    bounds = []
    # dd_s = time.time()
    for mask in masks:
        mask = np.pad(mask[0], thickness, 'constant', constant_values=0)
        mask_sobel_x = cv2.Sobel(mask, cv2.CV_16S, 1, 0)
        mask_sobel_y = cv2.Sobel(mask, cv2.CV_16S, 0, 1)
        abs_x = cv2.convertScaleAbs(mask_sobel_x)
        abs_y = cv2.convertScaleAbs(mask_sobel_y)
        bound = cv2.addWeighted(abs_x,0.5,abs_y,0.5,0)
        mask = mask[thickness:-thickness, thickness:-thickness]
        bound = bound[thickness:-thickness, thickness:-thickness]
        bound = (bound>0).astype(np.uint8)
        bounds.append(bound[np.newaxis, :, :])
    # dd_e = time.time()
    # ee_s = time.time()
    bounds = np.concatenate(bounds, axis=0)
    # bounds = bounds == 1
    # index = np.where()
    # ee_e = time.time()
    # print("dd:{}, ee:{}".format(float(dd_e-dd_s), float(ee_e-ee_s)))
    return bounds

def mask_losses(mask_logits, gt_masks, mask_weight=1.0, edge_weight=1.0):
    if len(gt_masks) == 0:
        return mask_logits.sum()*0

    mask_side_len = 224
    ## gt_masks shape: N*28*28
    mask_weights = torch.full_like(gt_masks.squeeze(1), mask_weight).float().detach()
    
    if edge_weight > 1.0:
        edges = edge(gt_masks)
        edges = torch.Tensor(edges).cuda()
        index = (edges==1)
        mask_weights[index] = edge_weight

    gt_masks = gt_masks.view(-1, mask_side_len*mask_side_len).to(dtype=torch.float32)
    mask_logits = mask_logits.view(-1, mask_side_len*mask_side_len).to(dtype=torch.float32)
    mask_weights = mask_weights.view(-1, mask_side_len*mask_side_len)

    mask_loss = F.binary_cross_entropy_with_logits(mask_logits, gt_masks, weight=mask_weights)
    return mask_loss
