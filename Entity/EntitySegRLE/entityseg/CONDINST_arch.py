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
from detectron2.structures.masks import PolygonMasks, polygons_to_bitmask

from .det_head.detection import build_det_head
from .det_head.utils.comm import aligned_bilinear

from .mask_head.dynamic_mask_head import build_dynamic_mask_head
from .mask_head.mask_branch import build_mask_branch

from .panopticfcn_tools.panopticfcn_head import build_kernel_head

from detectron2.structures import Instances, Boxes
import random
import pdb
import copy
logger = logging.getLogger(__name__)

__all__ = ["CONDINST"]

@META_ARCH_REGISTRY.register()
class CONDINST(nn.Module):
    """
    Implement the paper :paper:`CondINST`.
    """

    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.debug  = cfg.MODEL.CONDINST.DEBUG
        self.only_class_agnostic = cfg.MODEL.CONDINST.CLASS_AGNOSTIC

        self.backbone  = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        self.det_head  = build_det_head(cfg, backbone_shape)

        ## mask
        self.mask_head = build_dynamic_mask_head(cfg)
        self.mask_branch = build_mask_branch(cfg, self.backbone.output_shape())
        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE

        in_channels = self.det_head.in_channels_to_top_module

        self.controller = build_kernel_head(cfg, self.mask_head.num_gen_params)

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
            im_h, im_w = images.tensor.size(-2), images.tensor.size(-1)
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            for per_im_gt_inst in gt_instances:
                if not per_im_gt_inst.has("gt_masks"):
                    continue
                start = int(self.mask_out_stride // 2)
                if True:
                    polygons = per_im_gt_inst.get("gt_masks").polygons
                    per_im_bitmasks = []
                    per_im_bitmasks_full = []
                    for per_polygons in polygons:
                        bitmask = polygons_to_bitmask(per_polygons, im_h, im_w)
                        bitmask = torch.from_numpy(bitmask).to(self.device).float()
                        start = int(self.mask_out_stride // 2)
                        bitmask_full = bitmask.clone()
                        bitmask = bitmask[start::self.mask_out_stride, start::self.mask_out_stride]

                        assert bitmask.size(0) * self.mask_out_stride == im_h
                        assert bitmask.size(1) * self.mask_out_stride == im_w

                        per_im_bitmasks.append(bitmask)
                        per_im_bitmasks_full.append(bitmask_full)

                    per_im_gt_inst.gt_bitmasks = torch.stack(per_im_bitmasks, dim=0)
                    per_im_gt_inst.gt_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0)
        else:
            gt_instances = None

        mask_feats = self.mask_branch(features, gt_instances)
        proposals, proposal_losses = self.det_head(images, features, gt_instances, self.controller)

        if self.training:
            loss_masks = self._forward_mask_heads_train(proposals, mask_feats, gt_instances)
            losses = {}
            losses.update(proposal_losses)
            losses.update(loss_masks)
            return losses
        else:
            pred_instances_w_masks = self._forward_mask_heads_test(proposals, mask_feats)
            padded_im_h, padded_im_w = images.tensor.size()[-2:]
            processed_results = []
            for im_id, (input_per_image, image_size) in enumerate(zip(batched_inputs, images.image_sizes)):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])

                instances_per_im = pred_instances_w_masks[pred_instances_w_masks.im_inds == im_id]
                instances_per_im = self.postprocess(
                    instances_per_im, height, width,
                    padded_im_h, padded_im_w
                )

                processed_results.append({
                    "instances": instances_per_im
                })

            return processed_results

    def _forward_mask_heads_train(self, proposals, mask_feats, gt_instances):
        # prepare the inputs for mask heads
        pred_instances = proposals["instances"]

        # if 0 <= self.max_proposals < len(pred_instances):
        #     inds = torch.randperm(len(pred_instances), device=mask_feats.device).long()
        #     logger.info("clipping proposals from {} to {}".format(
        #         len(pred_instances), self.max_proposals
        #     ))
        #     pred_instances = pred_instances[inds[:self.max_proposals]]

        pred_instances.mask_head_params = pred_instances.top_feats

        loss_masks = self.mask_head(
            mask_feats, self.mask_branch.out_stride,
            pred_instances, gt_instances
        )
        return loss_masks

    def _forward_mask_heads_test(self, proposals, mask_feats):
        # prepare the inputs for mask heads
        for im_id, per_im in enumerate(proposals):
            per_im.im_inds = per_im.locations.new_ones(len(per_im), dtype=torch.long) * im_id
        pred_instances = Instances.cat(proposals)
        pred_instances.mask_head_params = pred_instances.top_feat

        pred_instances_w_masks = self.mask_head(mask_feats, self.mask_branch.out_stride, pred_instances)

        return pred_instances_w_masks

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for i,x in enumerate(batched_inputs)]
        # pdb.set_trace()
        # images = [self.normalizer(x) for x in images]
        images = [(x - self.pixel_mean)/self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def postprocess(self, results, output_height, output_width, padded_im_h, padded_im_w, mask_threshold=0.5):
        """
        Resize the output instances.
        The input images are often resized when entering an object detector.
        As a result, we often need the outputs of the detector in a different
        resolution from its inputs.
        This function will resize the raw outputs of an R-CNN detector
        to produce outputs according to the desired output resolution.
        Args:
            results (Instances): the raw outputs from the detector.
                `results.image_size` contains the input image resolution the detector sees.
                This object might be modified in-place.
            output_height, output_width: the desired output resolution.
        Returns:
            Instances: the resized output from the model, based on the output resolution
        """
        scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
        resized_im_h, resized_im_w = results.image_size
        results = Instances((output_height, output_width), **results.get_fields())

        if results.has("pred_boxes"):
            output_boxes = results.pred_boxes
        elif results.has("proposal_boxes"):
            output_boxes = results.proposal_boxes

        output_boxes.scale(scale_x, scale_y)
        output_boxes.clip(results.image_size)

        results = results[output_boxes.nonempty()]

        if results.has("pred_global_masks"):
            mask_h, mask_w = results.pred_global_masks.size()[-2:]
            factor_h = padded_im_h // mask_h
            factor_w = padded_im_w // mask_w
            assert factor_h == factor_w
            factor = factor_h
            pred_global_masks = aligned_bilinear(
                results.pred_global_masks, factor
            )
            pred_global_masks = pred_global_masks[:, :, :resized_im_h, :resized_im_w]
            pred_global_masks = F.interpolate(
                pred_global_masks,
                size=(output_height, output_width),
                mode="bilinear", align_corners=False
            )
            pred_global_masks = pred_global_masks[:, 0, :, :]
            results.pred_masks = (pred_global_masks > mask_threshold).float()

        return results