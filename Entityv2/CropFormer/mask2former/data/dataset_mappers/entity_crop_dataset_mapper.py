# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging

import numpy as np
import pycocotools.mask as mask_util
import torch
import random
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances, polygons_to_bitmask

from .crop_augmentations import BatchResizeShortestEdge, EntityCrop, EntityCropTransform, transform_instance_annotations_crop, annotations_to_instances_crop

import copy
import pdb

__all__ = ["EntityCropDatasetMapper"]

def empty_instances_indexes(
    instances, by_box=True, by_mask=True, box_threshold=1e-5, return_mask=False
):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty
        return_mask (bool): whether to return boolean mask of filtered instances

    Returns:
        Instances: the filtered instances.
        tensor[bool], optional: boolean mask of filtered instances
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    # TODO: can also filter visible keypoints

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x
    return m


class EntityCropDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for instance segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        cfg,
        image_format,
        instance_mask_format,
        size_divisibility,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.cfg = cfg

        self.img_format = image_format
        self.instance_mask_format = instance_mask_format
        self.size_divisibility = size_divisibility

    def generate_augs(self):
        ## make sure the shortest side and flip for both full image and crops
        if self.is_train:
            shortest_side = np.random.choice(self.cfg.INPUT.MIN_SIZE_TRAIN)
            flip_flag = (random.random() >= 0.5)
        else:
            shortest_side = np.random.choice([self.cfg.INPUT.MIN_SIZE_TEST])
            flip_flag = False
        
        augs = [
            T.RandomFlip(
                1.0 if flip_flag else 0.0, 
                horizontal=self.cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=self.cfg.INPUT.RANDOM_FLIP == "vertical",
            ),

            T.ResizeShortestEdge(
                (shortest_side,),
                self.cfg.INPUT.MAX_SIZE_TRAIN if self.is_train else self.cfg.INPUT.MAX_SIZE_TEST,
                self.cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            ),
            
        ]
        if self.cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=self.cfg.INPUT.FORMAT))

        # Build original image augmentation
        crop_augs = []
        crop_augs.append(T.RandomFlip(
                1.0 if flip_flag else 0.0, 
                horizontal=self.cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=self.cfg.INPUT.RANDOM_FLIP == "vertical",
                ))
        entity_crops = EntityCrop(self.cfg.ENTITY.CROP_AREA_RATIO, 
                                    self.cfg.ENTITY.CROP_STRIDE_RATIO,
                                    self.cfg.ENTITY.CROP_SAMPLE_NUM_TRAIN if self.is_train else self.cfg.ENTITY.CROP_SAMPLE_NUM_TEST, 
                                    self.is_train)
        crop_augs.append(entity_crops)
        
        entity_resize = BatchResizeShortestEdge((shortest_side,), self.cfg.INPUT.MAX_SIZE_TRAIN, self.cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING)
        crop_augs.append(entity_resize)
        crop_augs = T.AugmentationList(crop_augs)
        return augs, crop_augs

    @classmethod
    def from_config(cls, cfg, is_train=True):
        ret = {
            "is_train": is_train,
            "cfg": cfg,
            "image_format": cfg.INPUT.FORMAT,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # assert self.is_train, "MaskFormerPanopticDatasetMapper should only be used for training!"
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        tfm_gens, crop_augs = self.generate_augs()
        ### original images
        aug_input_ori = T.AugInput(copy.deepcopy(image))
        aug_input_ori, transforms_ori = T.apply_transform_gens(tfm_gens, aug_input_ori)
        image_ori = aug_input_ori.image
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image_ori.transpose(2, 0, 1)))

        #### crop images
        aug_input_crop = T.AugInput(copy.deepcopy(image))
        transforms_crop = crop_augs(aug_input_crop)
        image_crop = aug_input_crop.image
        assert len(image_crop.shape)==4, "the image shape must be [N, H, W, C]"
        dataset_dict["image_crop"] = torch.as_tensor(np.ascontiguousarray(image_crop.transpose(0, 3, 1, 2)))

        #### transform instnace masks of original images
        annos_ori = [utils.transform_instance_annotations(obj, transforms_ori, image_ori.shape[:2]) for obj in copy.deepcopy(dataset_dict["annotations"]) if obj.get("iscrowd", 0) == 0]
        image_shape = image_ori.shape[:2]
        if self.is_train:
            instances_ori = utils.annotations_to_instances(annos_ori, image_shape, mask_format=self.instance_mask_format)
            gt_valid = empty_instances_indexes(instances_ori).int()
            instances_ori.gt_boxes_valid = gt_valid
            dataset_dict["instances"] = instances_ori

        #### transform instnace masks of crop images
        assert len(image_crop.shape)==4, "the image shape must be [N, H, W, C]"
        image_shape = image_crop.shape[1:3]  # h, w
        if self.is_train:
            annos_crop = [transform_instance_annotations_crop(obj, transforms_crop, image_shape) for obj in copy.deepcopy(dataset_dict["annotations"]) if obj.get("iscrowd", 0) == 0]
            patch_instances, crop_ori_size, crop_indexes = annotations_to_instances_crop(annos_crop, image_shape, mask_format=self.instance_mask_format, return_indexes=True)
            for patch_instance in patch_instances:
                gt_valid = empty_instances_indexes(patch_instance).int()
                patch_instance.gt_boxes_valid = gt_valid
            dataset_dict["instances_crop"] = patch_instances
            dataset_dict["crop_region"] = crop_ori_size
            dataset_dict["crop_indexes"] = crop_indexes
        else:
            for transform_type in transforms_crop:
                if isinstance(transform_type, EntityCropTransform):
                    crop_axises = transform_type.crop_axises
                    crop_indexes = transform_type.crop_indexes
            dataset_dict["crop_region"] = torch.tensor(crop_axises)
            dataset_dict["crop_indexes"] = crop_indexes
        

        dataset_dict.pop("annotations")

        return dataset_dict
