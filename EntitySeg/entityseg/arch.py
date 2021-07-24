# -*- coding: utf-8 -*-
import logging
import torch
from torch import nn
import torch.nn.functional as F

from detectron2.structures import ImageList
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess, sem_seg_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

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

__all__ = ["ItemFPN"]
@META_ARCH_REGISTRY.register()
class EntityFPN(nn.Module):
    """
    Implement the paper :paper:`PanopticFPN`.
    """

    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone  = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        self.det_head  = build_det_head(cfg, backbone_shape)

        ## mask
        self.mask_head = build_dynamic_mask_head(cfg)
        self.mask_branch = build_mask_branch(cfg, self.backbone.output_shape())
        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE
        self.max_proposals   = cfg.MODEL.CONDINST.MAX_PROPOSALS
        self.only_class_agnostic = cfg.MODEL.CONDINST.CLASS_AGNOSTIC

        in_channels = self.det_head.in_channels_to_top_module

        self.controller = build_kernel_head(cfg, self.mask_head.num_gen_params)
        self.train_max_proposals_per_image = cfg.MODEL.CONDINST.TRAIN_MAX_PROPOSALS_PER_IMAGE

        self.use_mask_rescore_infer = cfg.MODEL.CONDINST.MASK_BRANCH.USE_MASK_RESCORE

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

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
        
        # for x in batched_inputs:
        #     print(x["file_name"])
        images      = self.preprocess_image(batched_inputs)
        features    = self.backbone(images.tensor)

        if "instances" in batched_inputs[0] and self.training:
            B = len(batched_inputs)
            for i in range(B):
                if self.only_class_agnostic:
                    batched_inputs[i]["instances"].gt_classes[:] = 0
                
                instance_map = batched_inputs[i]["instance_map"]
                num_instances = int(torch.max(instance_map)+1)
                instanceid = batched_inputs[i]["instances"].instanceid
                gt_bitmasks_pad = F.one_hot(instance_map.long(), num_instances)[...,instanceid].permute((2,0,1))
                
                pad_h, pad_w = images.tensor.size(-2), images.tensor.size(-1)
                no_pad_h, no_pad_w = gt_bitmasks_pad.shape[1:]

                padding_size = [0, pad_w - no_pad_w, 0, pad_h-no_pad_h]
                gt_bitmasks_pad = F.pad(gt_bitmasks_pad, padding_size, value=0)

                start = int(self.mask_out_stride // 2)
                bitmask_full = gt_bitmasks_pad.clone()
                bitmask  = gt_bitmasks_pad[:,start::self.mask_out_stride, start::self.mask_out_stride]

                N = bitmask.shape[0]
                batched_inputs[i]["instances"].gt_bitmasks = bitmask.int()
                batched_inputs[i]["instances"].gt_bitmasks_full = bitmask_full.int()
                
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        mask_feats = self.mask_branch(features, gt_instances)
        proposals, proposal_losses = self.det_head(images, features, gt_instances, self.controller)

        if self.training:
            max_num_proposals = self.train_max_proposals_per_image * len(batched_inputs)
            actual_num_proposals = len(proposals["instances"])
            if actual_num_proposals >= max_num_proposals:
                select = random.sample(list(range(actual_num_proposals)), max_num_proposals)
                proposals["instances"] = proposals["instances"][select]

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

        if 0 <= self.max_proposals < len(pred_instances):
            inds = torch.randperm(len(pred_instances), device=mask_feats.device).long()
            logger.info("clipping proposals from {} to {}".format(
                len(pred_instances), self.max_proposals
            ))
            pred_instances = pred_instances[inds[:self.max_proposals]]

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
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
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
            results.pred_masks_score = pred_global_masks

        # from high score to low score
        origin_masks = results.pred_masks
        num_instances, H, W = origin_masks.shape
        filter_masks = []

        # initialize background
        mask_0 = torch.zeros((H, W)).cuda() + 0.001
        filter_masks.insert(0, mask_0)
        score = 0.002
        for index in range(num_instances):
            mask = origin_masks[num_instances-index-1]
            mask[mask==1] = score
            filter_masks.insert(0, mask)
            score = score + 0.001
        
        filter_masks = torch.stack(filter_masks, dim=0)
        _, instance_ids = torch.max(filter_masks, dim=0)
        unique_instance_ids = torch.unique(instance_ids)

        ori_scores = results.scores.clone()
        has_mask_valid = []
        for instance_id in unique_instance_ids:
            if instance_id == num_instances:
                continue
            mask = (instance_ids==instance_id).float()
            finds_y, finds_x = torch.nonzero(mask==1, as_tuple=True)
            if len(finds_y) == 0:
                continue
            x1 = torch.min(finds_x)
            x2 = torch.max(finds_x)
            y1 = torch.min(finds_y)
            y2 = torch.max(finds_y)
            
            if x2-x1==0 or y2-y1==0:
                continue
            has_mask_valid.append(int(instance_id))
            
            ## mask rescoring would obtain higher performance
            if self.use_mask_rescore_infer:
                mask_score = results.pred_masks_score[instance_id]
                seg_scores = (mask_score * mask).sum() / mask.sum()
                results.scores[instance_id] = results.scores[instance_id] * seg_scores

            results.pred_masks[instance_id] = mask
            results.pred_boxes.tensor[instance_id][0] = x1
            results.pred_boxes.tensor[instance_id][1] = y1
            results.pred_boxes.tensor[instance_id][2] = x2
            results.pred_boxes.tensor[instance_id][3] = y2
        
        results.ori_scores = ori_scores
        results = results[has_mask_valid]
        return results