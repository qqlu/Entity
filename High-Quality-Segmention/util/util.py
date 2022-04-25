from torch.nn import functional as F

def compute_tensor_iu(seg, gt):
    seg = seg.squeeze(1)
    gt = gt.squeeze(1)
    
    intersection = (seg & gt).float().sum()
    union = (seg | gt).float().sum()

    return intersection, union

def compute_tensor_iou(seg, gt):
    seg = seg.squeeze(1)
    gt = gt.squeeze(1)
    
    intersection = (seg & gt).float().sum((1, 2))
    union = (seg | gt).float().sum((1, 2))
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return iou 

def resize_min_side(im, size, method):
    h, w = im.shape[-2:]
    min_side = min(h, w)
    ratio = size / min_side
    if method == 'bilinear':
        return F.interpolate(im, scale_factor=ratio, mode=method, align_corners=False)
    else:
        return F.interpolate(im, scale_factor=ratio, mode=method)

def resize_max_side(im, size, method):
    h, w = im.shape[-2:]
    max_side = max(h, w)
    ratio = size / max_side
    if method in ['bilinear', 'bicubic']:
        return F.interpolate(im, scale_factor=ratio, mode=method, align_corners=False)
    else:
        return F.interpolate(im, scale_factor=ratio, mode=method)
