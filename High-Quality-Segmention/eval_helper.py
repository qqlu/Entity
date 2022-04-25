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

def process_high_res_im(model, im, seg, para, name=None, aggre_device='cpu:0'):

    im = im.to(aggre_device)
    seg = seg.to(aggre_device)

    max_L = para['L']
    stride = para['stride']

    _, _, h, w = seg.shape

    """
    Global Step
    """
    if max(h, w) > max_L:
        im_small = resize_max_side(im, max_L, 'area')
        seg_small = resize_max_side(seg, max_L, 'area')
    else:
        im_small = im
        seg_small = seg

    images = safe_forward(model, im_small, seg_small)

    pred_224 = images['pred_224'].to(aggre_device)
    pred_56 = images['pred_56_2'].to(aggre_device)

    # del images
    if para['clear']:
        torch.cuda.empty_cache()
    
    """
    Local step
    """

    for new_size in [max(h, w)]:
        im_small = resize_max_side(im, new_size, 'area')
        seg_small = resize_max_side(seg, new_size, 'area')
        _, _, h, w = seg_small.shape

        combined_224 = torch.zeros_like(seg_small)
        combined_weight = torch.zeros_like(seg_small)

        r_pred_224 = (F.interpolate(pred_224, size=(h, w), mode='bilinear', align_corners=False)>0.5).float()*2-1
        r_pred_56 = F.interpolate(pred_56, size=(h, w), mode='bilinear', align_corners=False)*2-1

        padding = 16
        step_size = stride - padding*2
        step_len  = max_L

        used_start_idx = {}
        for x_idx in range((w)//step_size+1):
            for y_idx in range((h)//step_size+1):

                start_x = x_idx * step_size
                start_y = y_idx * step_size
                end_x = start_x + step_len
                end_y = start_y + step_len

                # Shift when required
                if end_y > h:
                    end_y = h
                    start_y = h - step_len
                if end_x > w:
                    end_x = w
                    start_x = w - step_len

                # Bound x/y range
                start_x = max(0, start_x)
                start_y = max(0, start_y)
                end_x = min(w, end_x)
                end_y = min(h, end_y)

                # The same crop might appear twice due to bounding/shifting
                start_idx = start_y*w + start_x
                if start_idx in used_start_idx:
                    continue
                else:
                    used_start_idx[start_idx] = True
                
                # Take crop
                im_part = im_small[:,:,start_y:end_y, start_x:end_x]
                seg_224_part = r_pred_224[:,:,start_y:end_y, start_x:end_x]
                seg_56_part = r_pred_56[:,:,start_y:end_y, start_x:end_x]

                # Skip when it is not an interesting crop anyway
                seg_part_norm = (seg_224_part>0).float()
                high_thres = 0.9
                low_thres = 0.1
                if (seg_part_norm.mean() > high_thres) or (seg_part_norm.mean() < low_thres):
                    continue
                grid_images = safe_forward(model, im_part, seg_224_part, seg_56_part)
                grid_pred_224 = grid_images['pred_224'].to(aggre_device)

                # Padding
                pred_sx = pred_sy = 0
                pred_ex = step_len
                pred_ey = step_len

                if start_x != 0:
                    start_x += padding
                    pred_sx += padding
                if start_y != 0:
                    start_y += padding
                    pred_sy += padding
                if end_x != w:
                    end_x -= padding
                    pred_ex -= padding
                if end_y != h:
                    end_y -= padding
                    pred_ey -= padding

                combined_224[:,:,start_y:end_y, start_x:end_x] += grid_pred_224[:,:,pred_sy:pred_ey,pred_sx:pred_ex]

                del grid_pred_224

                if para['clear']:
                    torch.cuda.empty_cache()

                # Used for averaging
                combined_weight[:,:,start_y:end_y, start_x:end_x] += 1

        # Final full resolution output
        seg_norm = (r_pred_224/2+0.5)
        pred_224 = combined_224 / combined_weight
        pred_224 = torch.where(combined_weight==0, seg_norm, pred_224)

    _, _, h, w = seg.shape
    images = {}
    images['pred_224'] = F.interpolate(pred_224, size=(h, w), mode='bilinear', align_corners=False)

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
