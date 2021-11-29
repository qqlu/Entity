# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
import torch.nn.functional as F
from typing import List, Optional, Union
import torch
import os
import pdb
import cv2

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from .Auginput import ItemAugInput

from panopticapi.utils import rgb2id, id2rgb

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapper"]


def filter_empty_instances(instances, by_box=True, by_mask=True, box_threshold=1e-5):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty

    Returns:
        Instances: the filtered instances.
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
    return instances[m], m


class DatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = augmentations
        self.image_format           = image_format
        # fmt: on
        logger = logging.getLogger(__name__)
        logger.info("Augmentations used in training: " + str(augmentations))

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = utils.build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
        }
        return ret 

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        
        name  = dataset_dict["file_name"].split("/")[-1].split(".")[0]

        if self.is_train:
            panoptic_annotation_path = os.path.join("datasets/coco/entity_train2017", name+".npz")
        else:
            panoptic_annotation_path = os.path.join("datasets/coco/entity_val2017", name+".npz")


        panoptic_semantic_map = np.load(panoptic_annotation_path)
        # x1,y1,x2,y2,category,thing_or_stuff,instance_id
        bounding_boxes = panoptic_semantic_map["bounding_box"].astype(np.float)
        
        info_map       = panoptic_semantic_map["map"]
        instance_map   = info_map[0]
        semantic_map   = info_map[1]
        # instance_map   = torch.tensor(instance_map).long()
        num_instances  = len(dataset_dict["annotations"])
        
        seg_info       = {"instance_map": instance_map, 
                          "semantic_map": semantic_map}

        aug_input  = ItemAugInput(image, seg_info=seg_info)
        transforms = aug_input.apply_augmentations(self.augmentations)

        image        = aug_input.image
        instance_map = aug_input.seg_info["instance_map"].copy()
        semantic_map = aug_input.seg_info["semantic_map"]

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        new_anns = dataset_dict.pop("annotations")
        new_anns = [obj for obj in new_anns if obj.get("iscrowd", 0) == 0]
        # assert len(new_anns) == bounding_boxes.shape[0], print("{}:{}".format(len(new_anns), bounding_boxes.shape[0]))
        isthing_list = []
        instance_id_list = []
        for i in range(len(new_anns)):
            x1, y1, x2, y2, category, thing, instance_id = bounding_boxes[i]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            w  = x2 - x1
            h  = y2 - y1
            new_anns[i]["bbox"] = [x1, y1, w, h]
            new_anns[i]["category_id"] = int(category)
            isthing_list.append(int(thing))
            instance_id_list.append(int(instance_id))

        isthing_list = torch.tensor(isthing_list, dtype=torch.int64)
        instance_id_list = torch.tensor(instance_id_list, dtype=torch.int64)

        # annos = [utils.transform_instance_annotations(obj, transforms, image_shape) for obj in new_anns if obj.get("iscrowd", 0) == 0]
        annos = [utils.transform_instance_annotations(obj, transforms, image_shape) for obj in new_anns if obj.get("iscrowd", 0) == 0]
        instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
        instances.instanceid = instance_id_list

        instances, select = filter_empty_instances(instances)

        dataset_dict["instances"] = instances
        dataset_dict["instance_map"] = torch.as_tensor(np.ascontiguousarray(instance_map))
        return dataset_dict
