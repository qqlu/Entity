#!/usr/bin/python3
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.layers import Conv2d, get_norm
from .deformable_conv_with_off import ModulatedDeformConvWithOff
from ..det_head.layers import conv_with_kaiming_uniform
import math
import pdb
from fvcore.nn import sigmoid_focal_loss_jit

class SingleHead(nn.Module):
    """
    Build single head with convolutions and coord conv.
    """
    def __init__(self, in_channel, conv_dims, num_convs, deform=False, coord=False, norm='', name=''):
        super().__init__()
        self.coord = coord
        self.conv_norm_relus = []
        if deform:
            conv_module = ModulatedDeformConvWithOff
        else:
            conv_module = Conv2d
        for k in range(num_convs):
            conv = conv_module(
                    in_channel if k==0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=get_norm(norm, conv_dims),
                    activation=F.relu,
                )
            self.add_module("{}_head_{}".format(name, k + 1), conv)
            self.conv_norm_relus.append(conv)

    def forward(self, x):
        if self.coord:
            x = self.coord_conv(x)
        for layer in self.conv_norm_relus:
            x = layer(x)
        return x
    
    def coord_conv(self, feat):
        with torch.no_grad():
            x_pos = torch.linspace(-1, 1, feat.shape[-2], device=feat.device)
            y_pos = torch.linspace(-1, 1, feat.shape[-1], device=feat.device)
            grid_x, grid_y = torch.meshgrid(x_pos, y_pos)
            grid_x = grid_x.unsqueeze(0).unsqueeze(1).expand(feat.shape[0], -1, -1, -1)
            grid_y = grid_y.unsqueeze(0).unsqueeze(1).expand(feat.shape[0], -1, -1, -1)
        feat = torch.cat([feat, grid_x, grid_y], dim=1)
        return feat

class KernelHead(nn.Module):
    """
    The head used in PanopticFCN to generate kernel weights for both Things and Stuff.
    """
    def __init__(self, cfg, num_gen_params):
        super().__init__()
        in_channel      = cfg.MODEL.FPN.OUT_CHANNELS
        conv_dims       = cfg.MODEL.KERNEL_HEAD.CONVS_DIM
        num_convs       = cfg.MODEL.KERNEL_HEAD.NUM_CONVS
        deform          = cfg.MODEL.KERNEL_HEAD.DEFORM
        coord           = cfg.MODEL.KERNEL_HEAD.COORD
        norm            = cfg.MODEL.KERNEL_HEAD.NORM

        self.num_gen_params = num_gen_params

        self.kernel_head = SingleHead(in_channel+2 if coord else in_channel, 
                                      conv_dims,
                                      num_convs,
                                      deform=deform,
                                      coord=coord,
                                      norm=norm,
                                      name='kernel_head')
        self.out_conv = Conv2d(conv_dims, self.num_gen_params, kernel_size=3, padding=1)
        nn.init.normal_(self.out_conv.weight, mean=0, std=0.01)
        if self.out_conv.bias is not None:
            nn.init.constant_(self.out_conv.bias, 0)
       
    def forward(self, feat):
        x = self.kernel_head(feat)
        x = self.out_conv(x)
        return x


class FeatureEncoder(nn.Module):
    """
    The head used in PanopticFCN for high-resolution feature generation.
    """
    def __init__(self, cfg):
        super().__init__()
        in_channel      = cfg.MODEL.SEMANTIC_FPN.CONVS_DIM
        conv_dims       = cfg.MODEL.FEATURE_ENCODER.CONVS_DIM
        num_convs       = cfg.MODEL.FEATURE_ENCODER.NUM_CONVS
        deform          = cfg.MODEL.FEATURE_ENCODER.DEFORM
        coord           = cfg.MODEL.FEATURE_ENCODER.COORD
        norm            = cfg.MODEL.FEATURE_ENCODER.NORM
        
        self.encode_head = SingleHead(in_channel+2 if coord else in_channel, 
                                      conv_dims, 
                                      num_convs, 
                                      deform=deform,
                                      coord=coord,
                                      norm=norm, 
                                      name='encode_head')

    def forward(self, feat):
        feat = self.encode_head(feat)
        return feat

class FeatureEncoderEdge(nn.Module):
    """
    The head used in PanopticFCN for high-resolution feature generation.
    """
    def __init__(self, cfg):
        super().__init__()
        in_channel      = cfg.MODEL.SEMANTIC_FPN.CONVS_DIM
        conv_dims       = cfg.MODEL.FEATURE_ENCODER.CONVS_DIM
        num_convs       = cfg.MODEL.FEATURE_ENCODER.NUM_CONVS
        deform          = cfg.MODEL.FEATURE_ENCODER.DEFORM
        coord           = cfg.MODEL.FEATURE_ENCODER.COORD
        norm            = cfg.MODEL.FEATURE_ENCODER.NORM
        
        self.encode_head = SingleHead(in_channel+2 if coord else in_channel, 
                                      conv_dims, 
                                      num_convs, 
                                      deform=deform,
                                      coord=coord,
                                      norm=norm, 
                                      name='encode_head')

        self.in_features = cfg.MODEL.CONDINST.MASK_BRANCH.IN_FEATURES
        self.out_stride  = 8

        norm = cfg.MODEL.CONDINST.MASK_BRANCH.NORM
        conv_block = conv_with_kaiming_uniform(norm, activation=True)

        self.sem_loss_on = cfg.MODEL.CONDINST.MASK_BRANCH.SEMANTIC_LOSS_ON
        if self.sem_loss_on:
            self.focal_loss_alpha = cfg.MODEL.FCOS.LOSS_ALPHA
            self.focal_loss_gamma = cfg.MODEL.FCOS.LOSS_GAMMA

            # in_channels = feature_channels[self.in_features[0]]
            self.seg_head = nn.Sequential(
                conv_block(conv_dims, conv_dims, kernel_size=3, stride=1),
                conv_block(conv_dims, conv_dims, kernel_size=3, stride=1)
            )

            self.logits = nn.Conv2d(conv_dims, 1, kernel_size=1, stride=1)

            prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.logits.bias, bias_value)

    def forward(self, feat, gt_instances=None):
        feat = self.encode_head(feat)

        losses = {}
        # auxiliary thing semantic loss
        if self.training and self.sem_loss_on:
            logits_pred = self.logits(self.seg_head(feat))
            
            boundary_targets = []
            for per_im_gt in gt_instances:
                boundary_targets.append(per_im_gt.gt_boundary_full.sum(dim=0))

            # # semantic_targets = torch.stack(semantic_targets, dim=0)
            boundary_targets = torch.stack(boundary_targets, dim=0)

            # resize target to reduce memory
            boundary_targets = boundary_targets[:, None, self.out_stride // 2::self.out_stride,self.out_stride // 2::self.out_stride]
            num_pos = (boundary_targets > 0).sum().float().clamp(min=1.0)

            loss_edge = sigmoid_focal_loss_jit(logits_pred, boundary_targets, alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma, reduction="sum") / num_pos
            losses['loss_edge_p3'] = loss_edge

        return feat, losses

def build_feature_encoder(cfg, input_shape=None):
    return FeatureEncoder(cfg)

def build_feature_encoder_edge(cfg, input_shape=None):
    return FeatureEncoderEdge(cfg)

def build_kernel_head(cfg, num_gen_params):
    return KernelHead(cfg, num_gen_params)
