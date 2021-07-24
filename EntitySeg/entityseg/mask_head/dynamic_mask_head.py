import torch
from torch.nn import functional as F
from torch import nn

from ..det_head.utils.comm import compute_locations, aligned_bilinear
from fvcore.nn import sigmoid_focal_loss_jit
from .utils import sigmoid_focal_loss_boundary, sigmoid_focal_loss_boundary_jit
import pdb

def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss

def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts)

    return weight_splits, bias_splits

def build_dynamic_mask_head(cfg):
    return DynamicMaskHead(cfg)

class DynamicMaskHead(nn.Module):
    def __init__(self, cfg):
        super(DynamicMaskHead, self).__init__()
        self.num_layers = cfg.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS
        self.channels = cfg.MODEL.CONDINST.MASK_HEAD.CHANNELS
        self.in_channels = cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS
        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE
        self.disable_rel_coords = cfg.MODEL.CONDINST.MASK_HEAD.DISABLE_REL_COORDS
        self.cluster_weight = cfg.MODEL.CONDINST.MASK_HEAD.CLUSTER_WEIGHT

        soi = [64,128,256,512,1024]
        # self.register_buffer("sizes_of_interest", torch.tensor(soi + [soi[-1] * 2]))
        self.register_buffer("sizes_of_interest", torch.tensor(soi))

        weight_nums, bias_nums = [], []
        for l in range(self.num_layers):
            if l == 0:
                if not self.disable_rel_coords:
                    weight_nums.append((self.in_channels + 2) * self.channels)
                else:
                    weight_nums.append(self.in_channels * self.channels)
                bias_nums.append(self.channels)
            elif l == self.num_layers - 1:
                weight_nums.append(self.channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.channels * self.channels)
                bias_nums.append(self.channels)
        
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

        stable_conv_1 = nn.Sequential(nn.Conv2d(10,8,kernel_size=3, stride=1, padding=1),nn.ReLU())
        torch.nn.init.normal_(stable_conv_1[0].weight, std=0.01)
        torch.nn.init.constant_(stable_conv_1[0].bias, 0)

        stable_conv_2 = nn.Sequential(nn.Conv2d(8,8,kernel_size=3, stride=1, padding=1),nn.ReLU())
        torch.nn.init.normal_(stable_conv_2[0].weight, std=0.01)
        torch.nn.init.constant_(stable_conv_2[0].bias, 0)

        stable_conv_3 = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(stable_conv_3.weight, std=0.01)
        torch.nn.init.constant_(stable_conv_3.bias, 0)
        self.stable = nn.ModuleList([stable_conv_1, stable_conv_2, stable_conv_3])

        self.general_choose = cfg.MODEL.CONDINST.MASK_HEAD.DYNAMIC
        self.general_choose_weight = cfg.MODEL.CONDINST.MASK_HEAD.DYNAMIC_WEIGHT
        self.key_weight = dict()
        for key, value in zip(self.general_choose, self.general_choose_weight):
            self.key_weight[key]=value


    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        mid_features = []
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(x, w, bias=b, stride=1, padding=0, groups=num_insts)
            if i < n_layers - 1:
                x = F.relu(x)
            mid_features.append(x)
        return x, mid_features

    def mask_heads_forward_split(self, features, weight, bias, num_insts, has_relu=True):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        # n_layers = len(weights)
        x = features
        x = F.conv2d(x, weight, bias=bias, stride=1, padding=0, groups=num_insts)
        if has_relu:
            x = F.relu(x)
        return x

    def mask_heads_forward_with_coords_test(self, mask_feats, mask_feat_stride, instances):
        locations = compute_locations(mask_feats.size(2), mask_feats.size(3), stride=mask_feat_stride, device=mask_feats.device)
        n_inst = len(instances)

        im_inds = instances.im_inds
        mask_head_params = instances.mask_head_params
        
        N, _, H, W = mask_feats.size()

        if not self.disable_rel_coords:
            instance_locations = instances.locations
            relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
            relative_coords = relative_coords.permute(0, 2, 1).float()
            soi = self.sizes_of_interest.float()[instances.fpn_levels]
            relative_coords = relative_coords / soi.reshape(-1, 1, 1)
            relative_coords = relative_coords.to(dtype=mask_feats.dtype)

            mask_head_inputs = torch.cat([
                relative_coords, mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)
            ], dim=1)
        else:
            mask_head_inputs = mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)

        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)

        weights, biases = parse_dynamic_params(
            mask_head_params, self.channels,
            self.weight_nums, self.bias_nums
        )

        mask_logits, mid_features = self.mask_heads_forward(mask_head_inputs, weights, biases, n_inst)

        mask_logits = mask_logits.reshape(-1, 1, H, W)

        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0
        mask_logits = aligned_bilinear(mask_logits, int(mask_feat_stride / self.mask_out_stride))

        return mask_logits.sigmoid()

    def mask_heads_forward_with_coords(self, mask_feats, mask_feat_stride, instances, gt_bitmasks, ignore_maps):
        locations = compute_locations(mask_feats.size(2), mask_feats.size(3), stride=mask_feat_stride, device=mask_feats.device)
        n_inst = len(instances)

        im_inds = instances.im_inds
        mask_head_params = instances.mask_head_params

        # clusters
        gt_inds = instances.gt_inds
        instance_locations = instances.locations
        fpn_levels = instances.fpn_levels

        clusters_ids = []
        clusters_imgids = []
        clusters_gt_masks = []
        gt_unique_inds = torch.unique(gt_inds)
        for gt_ind in gt_unique_inds:
            gt_ind = int(gt_ind)
            clusters_gt_masks.append(gt_bitmasks[gt_ind])
            im_ind = int(torch.unique(im_inds[(gt_inds == gt_ind)]))
            clusters_ids.append(gt_ind)
            clusters_imgids.append(im_ind)

        clusters_ids = torch.tensor(clusters_ids).cuda()
        clusters_imgids = torch.tensor(clusters_imgids)
        clusters_gt_masks = torch.stack(clusters_gt_masks, dim=0)
        n_clusters = len(clusters_ids)

        N, _, H, W = mask_feats.size()

        if not self.disable_rel_coords:
            instance_locations = instances.locations
            relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
            relative_coords = relative_coords.permute(0, 2, 1).float()
            soi = self.sizes_of_interest.float()[instances.fpn_levels]
            relative_coords = relative_coords / soi.reshape(-1, 1, 1)
            relative_coords = relative_coords.to(dtype=mask_feats.dtype)
            mask_head_inputs = torch.cat([relative_coords, mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)], dim=1)
        else:
            mask_head_inputs = mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)

        # mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)
        mask_head_inputs = mask_head_inputs.reshape(n_inst, self.in_channels+2, H, W)
        weights, biases = parse_dynamic_params(mask_head_params, self.channels, self.weight_nums, self.bias_nums)

        feature0  = self.stable[0](mask_head_inputs)
        feature1  = self.mask_heads_forward_split(mask_head_inputs.reshape(1, -1, H, W), weights[0], biases[0], n_inst).reshape(n_inst, -1, H, W)

        feature00 = self.stable[1](feature0)
        feature01 = self.mask_heads_forward_split(feature0.reshape(1, -1, H, W), weights[1], biases[1], n_inst).reshape(n_inst, -1, H, W)
        feature10 = self.stable[1](feature1)
        feature11 = self.mask_heads_forward_split(feature1.reshape(1, -1, H, W), weights[1], biases[1], n_inst).reshape(n_inst, -1, H, W)

        feature001 = self.mask_heads_forward_split(feature00.reshape(1, -1, H, W), weights[2], biases[2], n_inst, has_relu=False).reshape(n_inst, -1, H, W)
        feature010 = self.stable[2](feature01)
        feature011 = self.mask_heads_forward_split(feature01.reshape(1, -1, H, W), weights[2], biases[2], n_inst, has_relu=False).reshape(n_inst, -1, H, W)
        
        feature100 = self.stable[2](feature10)
        feature101 = self.mask_heads_forward_split(feature10.reshape(1, -1, H, W), weights[2], biases[2], n_inst, has_relu=False).reshape(n_inst, -1, H, W)
        feature110 = self.stable[2](feature11)
        feature111 = self.mask_heads_forward_split(feature11.reshape(1, -1, H, W), weights[2], biases[2], n_inst, has_relu=False).reshape(n_inst, -1, H, W)

        mask_logits_clusters = []
        for gt_ind in clusters_ids:
            gt_ind = int(gt_ind)
            mask_logits_clusters.append(torch.mean(feature111[gt_inds==gt_ind], dim=0))
        mask_logits_clusters = torch.stack(mask_logits_clusters, dim=0)
        mask_logits_clusters = mask_logits_clusters.reshape(-1, 1, H, W)
        mask_logits_clusters = aligned_bilinear(mask_logits_clusters, int(mask_feat_stride / self.mask_out_stride))
        # clusters
        unique_img_inds = torch.unique(clusters_imgids)
        mask_logits_clusters_imgs = []
        mask_gt_clusters_imgs = []
        for img_ind in unique_img_inds:
            img_ind = int(img_ind)
            mask_logits_clusters_per_img = mask_logits_clusters[clusters_imgids==img_ind]
            mask_logits_clusters_per_img = F.softmax(mask_logits_clusters_per_img.squeeze(1),dim=0).unsqueeze(1)

            ignore_map = ignore_maps[img_ind].detach()
            finds_y, finds_x = torch.nonzero(ignore_map, as_tuple=True)

            mask_logits_clusters_per_img = mask_logits_clusters_per_img.clone()
            mask_logits_clusters_per_img[...,finds_y,finds_x] = 0

            mask_logits_clusters_imgs.append(mask_logits_clusters_per_img)
            mask_gt_clusters_imgs.append(clusters_gt_masks[clusters_imgids==img_ind])
        mask_logits_clusters_imgs = torch.cat(mask_logits_clusters_imgs, dim=0)
        mask_gt_clusters_imgs = torch.cat(mask_gt_clusters_imgs, dim=0)

        select_features = {}
        for cid in self.general_choose:
            select_feature = locals()["feature{}".format(cid)]
            select_feature = aligned_bilinear(select_feature, int(mask_feat_stride / self.mask_out_stride)) 
            select_features[cid] = select_feature.sigmoid()

        return select_features, mask_logits_clusters_imgs, mask_gt_clusters_imgs.unsqueeze(1)

    def __call__(self, mask_feats, mask_feat_stride, pred_instances, gt_instances=None):
        if self.training:
            gt_inds = pred_instances.gt_inds
            gt_bitmasks_s = torch.cat([per_im.gt_bitmasks for per_im in gt_instances])
            gt_bitmasks = gt_bitmasks_s[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)
            
            bitmasks_full = []
            for gt_instance in gt_instances:
                bitmasks_full.append(gt_instance.gt_bitmasks.sum(dim=0))
            bitmasks_full = torch.stack(bitmasks_full)
            ignore_map = 1-bitmasks_full

            losses = {}
            if len(pred_instances) == 0:
                loss_mask = mask_feats.sum() * 0 + pred_instances.mask_head_params.sum() * 0
                for key, value in self.key_weight.items():
                    losses["loss_mask_bank_{}".format(key)] = loss_mask
                losses["loss_mask_cluster"] = loss_mask
            else:
                select_scores, mask_logits_clusters, mask_gts_clusters = self.mask_heads_forward_with_coords(mask_feats, mask_feat_stride, pred_instances, gt_bitmasks_s, ignore_map)
                for key, value in select_scores.items():
                    losses["loss_mask_bank_{}".format(key)] = dice_coefficient(value, gt_bitmasks).mean() * self.key_weight[key]
                
                mask_clusters_losses = dice_coefficient(mask_logits_clusters, mask_gts_clusters)
                mask_clusters_losses = mask_clusters_losses.mean()
                losses["loss_mask_cluster"] = mask_clusters_losses * self.cluster_weight
            return losses
        else:
            if len(pred_instances) > 0:
                mask_scores = self.mask_heads_forward_with_coords_test(mask_feats, mask_feat_stride, pred_instances)
                pred_instances.pred_global_masks = mask_scores.float()

            return pred_instances
