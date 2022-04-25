import torch
import torch.nn.functional as F

from util.util import resize_max_side


def safe_forward(model, im, seg, inter_s8=None, inter_s4=None):
    """
    Slightly pads the input image such that its length is a multiple of 8
    """
    b, _, ph, pw = seg.shape
    if (ph % 8 != 0) or (pw % 8 != 0):
        newH = ((ph//8+1)*8)
        newW = ((pw//8+1)*8)
        p_im = torch.zeros(b, 3, newH, newW).cuda()
        p_seg = torch.zeros(b, 1, newH, newW).cuda() - 1

        p_im[:,:,0:ph,0:pw] = im
        p_seg[:,:,0:ph,0:pw] = seg
        im = p_im
        seg = p_seg

        if inter_s8 is not None:
            p_inter_s8 = torch.zeros(b, 1, newH, newW).cuda() - 1
            p_inter_s8[:,:,0:ph,0:pw] = inter_s8
            inter_s8 = p_inter_s8
        if inter_s4 is not None:
            p_inter_s4 = torch.zeros(b, 1, newH, newW).cuda() - 1
            p_inter_s4[:,:,0:ph,0:pw] = inter_s4
            inter_s4 = p_inter_s4

    images = model(im, seg, inter_s8, inter_s4)
    return_im = {}

    for key in ['pred_224', 'pred_28_3', 'pred_56_2']:
        return_im[key] = images[key][:,:,0:ph,0:pw]
    del images

    return return_im

def process_high_res_im(model, im, seg, para, name=None, aggre_device='cpu:0', coord=None, cell=None):

    im = im.to(aggre_device)
    seg = seg.to(aggre_device)

    images = model(im, seg, coord, cell)

    import pdb; pdb.set_trace()
    if para['clear']:
        torch.cuda.empty_cache()

    return images


def process_im_single_pass(model, im, seg, min_size, para):
    """
    A single pass version, aka global step only.
    """

    max_size = para['L']

    _, _, h, w = im.shape
    if max(h, w) < min_size:
        im = resize_max_side(im, min_size, 'bicubic')
        seg = resize_max_side(seg, min_size, 'bilinear')

    if max(h, w) > max_size:
        im = resize_max_side(im, max_size, 'area')
        seg = resize_max_side(seg, max_size, 'area')

    images = safe_forward(model, im, seg)

    if max(h, w) < min_size:
        images['pred_224'] = F.interpolate(images['pred_224'], size=(h, w), mode='area')
    elif max(h, w) > max_size:
        images['pred_224'] = F.interpolate(images['pred_224'], size=(h, w), mode='bilinear', align_corners=False)

    return images
