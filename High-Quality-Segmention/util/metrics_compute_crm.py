import torch.nn.functional as F

from util.util import compute_tensor_iu

def get_new_iou_hook(values, size):
    return 'iou/new_iou_%s'%size, values['iou/new_i_%s'%size]/values['iou/new_u_%s'%size]

def get_orig_iou_hook(values):
    return 'iou/orig_iou', values['iou/orig_i']/values['iou/orig_u']

def get_iou_gain(values, size):
    return 'iou/iou_gain_%s'%size, values['iou/new_iou_%s'%size] - values['iou/orig_iou']

iou_hooks_to_be_used = [
        get_orig_iou_hook,
        lambda x: get_new_iou_hook(x, '224'), lambda x: get_iou_gain(x, '224'),
    ]

iou_hooks_final_only = [
    get_orig_iou_hook,
    lambda x: get_new_iou_hook(x, '224'), lambda x: get_iou_gain(x, '224'),
]

# Compute common loss and metric for generator only
def compute_loss_and_metrics(images, para, detailed=True, need_loss=True, has_lower_res=True):

    """
    This part compute loss and metrics for the generator
    """

    loss_and_metrics = {}

    gt = images['gt']
    seg = images['seg']

    pred_224 = images['pred_224']

    if need_loss:
        # Loss weights
        ce_weights = para['ce_weight']
        l1_weights = para['l1_weight']
        l2_weights = para['l2_weight']

        # temp holder for losses at different scale
        ce_loss = 0 
        l1_loss = 0 
        l2_loss = 0 
        loss = 0 

        ce_loss = F.binary_cross_entropy_with_logits(images['out_224'], (gt>0.5).float())
        l1_loss = F.l1_loss(pred_224, gt)
        l2_loss = F.mse_loss(pred_224, gt)

        loss_and_metrics['grad_loss'] = F.l1_loss(images['gt_sobel'], images['pred_sobel'])

        # Weighted loss for different levels
        loss = ce_loss * ce_weights + l1_loss * l1_weights + l2_loss * l2_weights
        
        loss += loss_and_metrics['grad_loss'] * para['grad_weight']

    """
    Compute IOU stats
    """
    orig_total_i, orig_total_u = compute_tensor_iu(seg>0.5, gt>0.5)
    loss_and_metrics['iou/orig_i'] = orig_total_i
    loss_and_metrics['iou/orig_u'] = orig_total_u

    new_total_i, new_total_u = compute_tensor_iu(pred_224>0.5, gt>0.5)
    loss_and_metrics['iou/new_i_224'] = new_total_i
    loss_and_metrics['iou/new_u_224'] = new_total_u
        
    """
    All done.
    Now gather everything in a dict for logging
    """

    if need_loss:
        loss_and_metrics['total_loss'] = 0
        loss_and_metrics['ce_loss'] = ce_loss
        loss_and_metrics['l1_loss'] = l1_loss
        loss_and_metrics['l2_loss'] = l2_loss
        loss_and_metrics['loss'] = loss

        loss_and_metrics['total_loss'] += loss

    return loss_and_metrics

