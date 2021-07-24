import math
from typing import List, Dict
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.structures import ImageList
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.backbone import build_backbone
from detectron2.layers import ShapeSpec
from detectron2.modeling.postprocessing import detector_postprocess

from .layers import DFConv2d, IOULoss
# from .outputs_has_ignore import FCOSOutputs
from .outputs import FCOSOutputs
from .tower import FCOSHead

import pdb
import cv2

INF = 100000000

class FCOS(nn.Module):
    def __init__(self, cfg, backbone_shape):
        super().__init__()

        self.device               = torch.device(cfg.MODEL.DEVICE)
        self.in_features          = cfg.MODEL.FCOS.IN_FEATURES
        self.fpn_strides          = cfg.MODEL.FCOS.FPN_STRIDES
        self.yield_proposal       = cfg.MODEL.FCOS.YIELD_PROPOSAL

        feature_shapes = [backbone_shape[f] for f in self.in_features]
        self.fcos_head = FCOSHead(cfg, feature_shapes)
        self.in_channels_to_top_module = self.fcos_head.in_channels_to_top_module
        self.fcos_outputs = FCOSOutputs(cfg)
        self.to(self.device)

    def forward_head(self, features, top_module=None):
        features = [features[f] for f in self.in_features]
        pred_class_logits, pred_deltas, pred_centerness, bbox_towers, top_feats = self.fcos_head(features, top_module, self.yield_proposal)
        return pred_class_logits, pred_deltas, pred_centerness, bbox_towers, top_feats

    def forward(self, images, backbone_features, gt_instances, top_module=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        features = [backbone_features[f] for f in self.in_features]
        locations = self.compute_locations(features)
        logits_pred, reg_pred, ctrness_pred, bbox_towers, top_feats = self.fcos_head(features, top_module)
        
        results = {}
        if self.yield_proposal:
            results["features"] = {
                f: b for f, b in zip(self.in_features, bbox_towers)
            }

        if self.training:
            results, losses = self.fcos_outputs.losses(
                logits_pred, reg_pred, ctrness_pred,
                locations, gt_instances, top_feats
            )
            
            if self.yield_proposal:
                with torch.no_grad():
                    results["proposals"] = self.fcos_outputs.predict_proposals(
                        logits_pred, reg_pred, ctrness_pred,
                        locations, images.image_sizes, top_feats
                    )
            return results, losses
        else:
            results = self.fcos_outputs.predict_proposals(
                logits_pred, reg_pred, ctrness_pred,
                locations, images.image_sizes, top_feats
            )

            return results, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

def build_det_head(cfg, backbone_shape):
    return FCOS(cfg, backbone_shape)
