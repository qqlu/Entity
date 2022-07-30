# -*- coding: utf-8 -*-
import logging
import torch
import pdb
from torch import nn
import torch.nn.functional as F

from detectron2.structures import ImageList
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess, sem_seg_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import detector_postprocess, sem_seg_postprocess

from .det_head.detection import build_det_head
from .det_head.utils.comm import aligned_bilinear

from detectron2.structures import Instances, Boxes
import random
import pdb
import copy
logger = logging.getLogger(__name__)

__all__ = ["EntityFPNDET"]

@META_ARCH_REGISTRY.register()
class EntityFPNDET(nn.Module):
    """
    Implement the paper :paper:`PanopticFPN`.
    """

    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.debug  = cfg.MODEL.CONDINST.DEBUG
        self.only_class_agnostic = cfg.MODEL.CONDINST.CLASS_AGNOSTIC

        self.backbone  = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        self.det_head  = build_det_head(cfg, backbone_shape)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.pixel_mean = pixel_mean
        self.pixel_std  = pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.

                For now, each item in the list is a dict that contains:

                * "image": Tensor, image in (C, H, W) format.
                * "instances": Instances
                * "sem_seg": semantic segmentation ground truth.
                * Other information that's included in the original dicts, such as:
                  "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                each dict is the results for one image. The dict contains the following keys:

                * "instances": see :meth:`GeneralizedRCNN.forward` for its format.
                * "sem_seg": see :meth:`SemanticSegmentor.forward` for its format.
                * "panoptic_seg": available when `PANOPTIC_FPN.COMBINE.ENABLED`.
                  See the return value of
                  :func:`combine_semantic_and_instance_outputs` for its format.
        """
        if self.debug:
            print("batched_inputs:{}".format(batched_inputs))

        images      = self.preprocess_image(batched_inputs)
        features    = self.backbone(images.tensor)

        if "instances" in batched_inputs[0] and self.training:
            B = len(batched_inputs)
            for i in range(B):
                if self.only_class_agnostic:
                    batched_inputs[i]["instances"].gt_classes[:] = 0
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        proposals, proposal_losses = self.det_head(images, features, gt_instances)

        if self.training:
            losses = {}
            losses.update(proposal_losses)
            return losses
        else:
            processed_results = []
            for im_id, (input_per_image, image_size) in enumerate(zip(batched_inputs, images.image_sizes)):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])

                r = detector_postprocess(proposals[im_id], height, width)
                processed_results = [{"instances": r}]
            return processed_results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        # pdb.set_trace()
        # images = [self.normalizer(x) for x in images]
        images = [(x - self.pixel_mean)/self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images