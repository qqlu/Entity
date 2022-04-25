import numpy as np

def is_bb_overlap(rmin, rmax, cmin, cmax, 
                    crmin, crmax, ccmin, ccmax):
    
    is_y_overlap = (rmax > crmin) and (crmax > rmin)
    is_x_overlap = (cmax > ccmin) and (ccmax > cmin)

    return is_x_overlap and is_y_overlap

def get_bb_position(mask):
    mask = mask > 0.5
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # y_min, y_max, x_min, x_max
    return rmin, rmax, cmin, cmax

def scale_bb_by(rmin, rmax, cmin, cmax, im_height, im_width, h_scale, w_scale):
    height = rmax - rmin
    width = cmax - cmin

    rmin -= h_scale * height / 2
    rmax += h_scale * height / 2
    cmin -= w_scale * width / 2
    cmax += w_scale * width / 2

    rmin = int(max(0, rmin))
    rmax = int(min(im_height-1, rmax))
    cmin = int(max(0, cmin))
    cmax = int(min(im_width-1, cmax))

    # Prevent negative width/height
    rmax = max(rmin, rmax)
    cmax = max(cmin, cmax)

    return rmin, rmax, cmin, cmax 
