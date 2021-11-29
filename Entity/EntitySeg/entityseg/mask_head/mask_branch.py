from typing import Dict
import math

import torch
from torch import nn
import pdb
from fvcore.nn import sigmoid_focal_loss_jit
from detectron2.layers import ShapeSpec

from ..det_head.layers import conv_with_kaiming_uniform
from ..det_head.utils.comm import aligned_bilinear

INF = 100000000

def build_mask_branch(cfg, input_shape):
    return MaskBranch(cfg, input_shape)

class MaskBranch(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self.in_features = cfg.MODEL.CONDINST.MASK_BRANCH.IN_FEATURES
        self.sem_loss_on = cfg.MODEL.CONDINST.MASK_BRANCH.SEMANTIC_LOSS_ON
        self.num_outputs = cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS
        norm = cfg.MODEL.CONDINST.MASK_BRANCH.NORM
        num_convs = cfg.MODEL.CONDINST.MASK_BRANCH.NUM_CONVS
        channels = cfg.MODEL.CONDINST.MASK_BRANCH.CHANNELS
        self.out_stride = input_shape[self.in_features[0]].stride

        feature_channels = {k: v.channels for k, v in input_shape.items()}

        conv_block = conv_with_kaiming_uniform(norm, activation=True)

        self.refine = nn.ModuleList()
        for in_feature in self.in_features:
            self.refine.append(conv_block(
                feature_channels[in_feature],
                channels, 3, 1
            ))

        tower = []
        for i in range(num_convs):
            tower.append(conv_block(
                channels, channels, 3, 1
            ))
        tower.append(nn.Conv2d(
            channels, max(self.num_outputs, 1), 1
        ))
        self.add_module('tower', nn.Sequential(*tower))

    def forward(self, features, gt_instances=None):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.refine[i](features[f])
            else:
                x_p = self.refine[i](features[f])

                target_h, target_w = x.size()[2:]
                h, w = x_p.size()[2:]
                assert target_h % h == 0
                assert target_w % w == 0
                factor_h, factor_w = target_h // h, target_w // w
                assert factor_h == factor_w
                x_p = aligned_bilinear(x_p, factor_h)
                x = x + x_p

        mask_feats = self.tower(x)

        if self.num_outputs == 0:
            mask_feats = mask_feats[:, :self.num_outputs]

        return mask_feats
