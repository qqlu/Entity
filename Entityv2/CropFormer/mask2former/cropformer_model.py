# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
import pdb
import numpy as np
import cv2
import os

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from .modeling.criterion_view import ViewSetCriterion
from .modeling.matcher_view import ViewHungarianMatcher
import pdb
import copy

@META_ARCH_REGISTRY.register()
class CropFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """
    @configurable
    def __init__(
        self,
        *,
        cfg,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion_2d: nn.Module,
        criterion_3d: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion_2d = criterion_2d
        self.criterion_3d = criterion_3d
        ## colors
        self.colors = [info["color"] for info in COCO_CATEGORIES]

        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        ## colors
        self.colors = [info["color"] for info in COCO_CATEGORIES]

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher_2d = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        matcher_3d = ViewHungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion_2d = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher_2d,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        criterion_3d = ViewSetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher_3d,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "cfg": cfg,
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion_2d": criterion_2d,
            "criterion_3d": criterion_3d,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        ## make new images
        batched_inputs_new = []
        for batched_input in batched_inputs:
            ori_infos = {"height": batched_input["height"],
                        "width": batched_input["width"], 
                        "image": batched_input["image"],
                        # "file_name": batched_input["file_name"],
                        }
            if "instances" in batched_input.keys():
                ori_instances = batched_input["instances"]
                ori_instances.original_indices = torch.arange(0, len(ori_instances)).long()
                ori_infos["instances"] = ori_instances
            batched_inputs_new.append(ori_infos)
            ## cropped patches
            # pdb.set_trace()
            crop_region = batched_input["crop_region"]
            crop_images = batched_input["image_crop"]
            crop_o_width  = int(crop_region[0][2]-crop_region[0][0])
            crop_o_height = int(crop_region[0][3]-crop_region[0][1])

            if "instances_crop" in batched_input.keys():
                crop_instances = batched_input["instances_crop"]
            else:
                crop_instances = None

            for crop_index, crop_image in enumerate(crop_images):
                crop_infos = {"height": crop_o_height, "width": crop_o_width, "image": crop_image}
                if not crop_instances == None:
                    crop_instance = crop_instances[crop_index]
                    crop_instance.original_indices = torch.arange(0, len(crop_instance)).long()
                    crop_infos["instances"] = crop_instance
                batched_inputs_new.append(crop_infos)

        images = [x["image"].to(self.device) for x in batched_inputs_new]
        ## +1 means 
        num_views = self.cfg.ENTITY.CROP_SAMPLE_NUM_TRAIN+1 if self.training else self.cfg.ENTITY.CROP_SAMPLE_NUM_TEST+1
        for i in range(len(images)):
            if i%num_views==0:
                continue
            _, c_h, c_w = images[i].shape
            if "instances" in batched_inputs_new[i].keys():
                batched_inputs_new[i]["instances"]._image_size = (c_h, c_w)
        
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        outputs_2d, outputs_3d = self.sem_seg_head(features)

        if self.training:
            if self.cfg.ENTITY.ENABLE:
                for i in range(len(batched_inputs_new)):
                    batched_inputs_new[i]["instances"].gt_classes[:] = 0
            
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs_new]
                targets_2d = self.prepare_targets_2d(copy.deepcopy(gt_instances), copy.deepcopy(images))
                targets_3d = self.prepare_targets_3d(copy.deepcopy(gt_instances), copy.deepcopy(images), num_views)
            else:
                targets = None

            # bipartite matching-based loss
            losses = {}
            losses_2d = self.criterion_2d(outputs_2d, targets_2d)
            losses_3d = self.criterion_3d(outputs_3d, targets_3d)

            for k in list(losses_2d.keys()):
                if k in self.criterion_2d.weight_dict:
                    losses[k+"_2d"] = losses_2d[k] * self.criterion_2d.weight_dict[k] * 0.5
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses_2d.pop(k)
            
            for k in list(losses_3d.keys()):
                if k in self.criterion_3d.weight_dict:
                    losses[k+"_3d"] = losses_3d[k] * self.criterion_3d.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses_3d.pop(k)
            return losses
        else:
            mask_cls_results_3d  = outputs_3d["pred_logits"][0] ## 100,2
            mask_pred_results_3d = outputs_3d["pred_masks"][0]  ## 100,5,200, 304

            mask_cls_results_2d  = outputs_2d["pred_logits"]
            mask_pred_results_2d = outputs_2d["pred_masks"]
            # upsample masks
            
            mask_pred_results_3d = retry_if_cuda_oom(F.interpolate)(
                mask_pred_results_3d,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            mask_pred_results_2d = F.interpolate(
                mask_pred_results_2d,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs_2d, outputs_3d
            
            crop_regions = batched_input["crop_region"][:num_views-1]
            processed_results = retry_if_cuda_oom(self.inference_whole_views)(
                                mask_cls_results_3d,
                                mask_pred_results_3d,
                                mask_cls_results_2d,
                                mask_pred_results_2d,
                                batched_inputs_new,
                                images.image_sizes, 
                                crop_regions)

            # processed_results = retry_if_cuda_oom(self.instance_inference_nonoverlap)(
            #                             mask_cls_results_2d[0], 
            #                             mask_pred_results_2d[0],
            #                             batched_inputs_new[0], 
            #                             images.image_sizes[0])

            return [{"instances": processed_results}]

    def prepare_targets_2d(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            gt_masks = targets_per_image.gt_masks.tensor           
            gt_valid = targets_per_image.gt_boxes_valid
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            valid_index = torch.nonzero(gt_valid).flatten()
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes[valid_index],
                    "masks": padded_masks[valid_index],
                }
            )
        return new_targets
    
    def prepare_targets_3d(self, targets_ori, images, num_views):
        T = num_views
        B = int(len(targets_ori) / T)
        h_pad, w_pad = images.tensor.shape[-2:]
        
        ## reshape to new targets
        new_targets = []
        for count, target in enumerate(targets_ori):
            b_index, t_index = int(count // T), int(count % T)
            if t_index == 0:
                new_targets.append([target])
            else:
                new_targets[b_index].append(target)

        gt_instances = []
        for count, targets in enumerate(new_targets):
            _num_instance = len(targets[0])
            mask_shape = [_num_instance, T, h_pad, w_pad]
            gt_masks_per_view = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)

            for v_i, targets_per_view in enumerate(targets):
                assert torch.all(targets[0].original_indices == targets_per_view.original_indices)
            
            gt_ids_per_view   = []
            gt_ids_per_valid  = []
            gt_ids_categories = []
            ## view first, then entities
            for v_i, targets_per_view in enumerate(targets):
                targets_per_view = targets_per_view.to(self.device)
                h, w = targets_per_view.image_size
                for i_i, (instance_mask, instance_valid) in enumerate(zip(targets_per_view.gt_masks.tensor, targets_per_view.gt_boxes_valid)):
                    if instance_valid == 1:
                        gt_masks_per_view[i_i, v_i, :h, :w] = instance_mask
                gt_ids_per_valid.append(targets_per_view.gt_boxes_valid[None,:])
                gt_ids_per_view.append(targets_per_view.original_indices[None,:])
                gt_ids_categories.append(targets_per_view.gt_classes[None, :])
            ## (num_instances, num_views)
            gt_ids_per_valid = torch.cat(gt_ids_per_valid, dim=0).permute((1,0))
            gt_ids_per_view = torch.cat(gt_ids_per_view, dim=0).permute((1,0))
            gt_ids_categories = torch.cat(gt_ids_categories, dim=0).permute((1,0))
            
            gt_ids_per_view[gt_ids_per_valid == 0] = -1
            valid_idx = (gt_ids_per_view != 1).any(dim=-1)
            ## categoreis 
            gt_classes_per_group = gt_ids_categories[:,0]   ## N
            gt_ids_per_group = gt_ids_per_view   ## N, num_views
            gt_masks_per_group = gt_masks_per_view.float() ## N, num_views, H, W

            ## 
            gt_instances.append({"labels": gt_classes_per_group, 
                                "ids": gt_ids_per_group,
                                "masks": gt_masks_per_group})

        return gt_instances

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )
            return panoptic_seg, segments_info
    
    def instance_inference_nonoverlap(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.sem_seg_head.num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        ###### ranks
        pred_masks = (mask_pred>0).float()
        pred_masks_logits = mask_pred.sigmoid()
        pred_scores = scores_per_image

        _, m_H, m_W = pred_masks.shape
        mask_id = torch.zeros((m_H, m_W), dtype=torch.int).to(pred_masks.device)
        sorted_scores, ranks = torch.sort(pred_scores)
        ranks = ranks + 1
        for index in ranks:
            mask_id[(pred_masks[index-1]==1)] = int(index)
        # re-generate mask
        new_scores = []
        new_masks  = []
        new_masks_logits = []
        entity_nums = len(ranks)
        for ii in range(entity_nums):
            index = int(ranks[entity_nums-ii-1])
            score = sorted_scores[entity_nums-ii-1]
            new_scores.append(score)
            new_masks.append((mask_id==index).float())
            new_masks_logits.append(pred_masks_logits[index-1])
        
        new_scores = torch.stack(new_scores)
        new_masks  = torch.stack(new_masks)
        new_masks_logits = torch.stack(new_masks_logits)

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = new_masks
        result.pred_boxes = Boxes(torch.zeros(new_masks.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)

        # calculate average mask prob
        mask_scores_per_image = (new_masks_logits.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = new_scores * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]
        
        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]
        
        topk_indices = topk_indices // self.sem_seg_head.num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        # pdb.set_trace()
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result
    
    def inference_whole_views(self, pred_cls, pred_masks, pred_cls_2d, pred_masks_2d, batched_inputs, image_sizes, crop_regions):
        ## pred_masks: [100, 5, 800, 1216]
        ## pred_masks_2d: [5, 100, 800, 1216]
        scores = F.softmax(pred_cls, dim=-1)[:,:-1]   # 100,1
        scores_2d = F.softmax(pred_cls_2d, dim=-1)[:, :, :-1]  # 5, 100, 1
        
        # scores = (scores+scores_2d[0])/2
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        ### keep all the indices
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        labels_per_image = labels[topk_indices]
        # topk_indices = topk_indices // self.sem_seg_head.num_classes
        topk_indices = torch.div(topk_indices, self.sem_seg_head.num_classes, rounding_mode="trunc")
        pred_masks = pred_masks[topk_indices]
        pred_masks = pred_masks.permute((1,0,2,3))

        new_pred_masks = []
        for view_index, (pred_masks_per_view, batched_input_per_view, image_size_per_view) in enumerate(zip(pred_masks, batched_inputs, image_sizes)):
            O_H = batched_input_per_view["height"]
            O_W = batched_input_per_view["width"]

            SO_H, SO_W = image_size_per_view

            pred_masks_per_view = pred_masks_per_view[..., : SO_H, :SO_W]
            pred_masks_per_view = F.interpolate(pred_masks_per_view[None], size=(O_H, O_W), mode="bilinear", align_corners=False)

            new_pred_masks.append(pred_masks_per_view[0].sigmoid())
        
        ## fuse the masks
        full_image_masks  = new_pred_masks[0]
        
        ## fuse crop image
        fused_image_masks = torch.zeros_like(full_image_masks).float()
        fused_image_masks_valid = torch.zeros_like(full_image_masks).float() + 1e-16
        for crop_region_per_view, pred_masks_per_view in zip(crop_regions, new_pred_masks[1:]):
            x0, y0, x1, y1 = crop_region_per_view
            fused_image_masks[..., y0:y1, x0:x1] += pred_masks_per_view
            fused_image_masks_valid[..., y0:y1, x0:x1] += 1
        
        # add original masks
        fused_image_masks += full_image_masks
        fused_image_masks_valid += 1

        ## average
        fuse_image_masks = fused_image_masks / fused_image_masks_valid

        ###### change to the single image, begin to non_overlap_supression
        ##  ranks
        pred_masks_logits = fuse_image_masks
        pred_masks = (fuse_image_masks>0.5).float()
        pred_scores = scores_per_image

        _, m_H, m_W = pred_masks.shape
        ## for visualization
        mask_id = torch.zeros((m_H, m_W), dtype=torch.int).to(pred_masks.device)
        
        # mask_id_colors = np.zeros((m_H, m_W, 3), dtype=np.uint8)
        # pred_masks_np = pred_masks.cpu().numpy()

        sorted_scores, ranks = torch.sort(pred_scores)
        ranks = ranks + 1
        for index in ranks:
            mask_id[(pred_masks[index-1]==1)] = int(index)
            # mask_id_colors[(pred_masks_np[index-1]==1)] = self.colors[index]
        # base_path = "/group/20018/gavinqi/vis_entityv2_release_debug"
        # pdb.set_trace()
        # file_name = batched_inputs[0]["file_name"]
        # split_index, img_name = file_name.split("/")[-2:]
        # save_name = img_name.split(".")[0]+".png"
        # if not os.path.exists(os.path.join(base_path, save_name)):
        #     cv2.imwrite(os.path.join(base_path, save_name), mask_id_colors)
        # re-generate mask
        new_scores = []
        new_masks  = []
        new_masks_logits = []
        entity_nums = len(ranks)
        for ii in range(entity_nums):
            index = int(ranks[entity_nums-ii-1])
            score = sorted_scores[entity_nums-ii-1]
            new_scores.append(score)
            new_masks.append((mask_id==index).float())
            new_masks_logits.append(pred_masks_logits[index-1])
        
        new_scores = torch.stack(new_scores)
        new_masks  = torch.stack(new_masks)
        new_masks_logits = torch.stack(new_masks_logits)
        # make result
        image_size = (batched_inputs[0]["height"], batched_inputs[0]["width"])
        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = new_masks
        result.pred_boxes = Boxes(torch.zeros(new_masks.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)

        # calculate average mask prob
        mask_scores_per_image = (new_masks_logits.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = new_scores * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result