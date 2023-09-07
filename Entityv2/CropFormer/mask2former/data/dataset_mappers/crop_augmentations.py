# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Implement many useful :class:`Augmentation`.
"""
import numpy as np
import sys
from typing import Tuple
from PIL import Image
import random

from fvcore.transforms.transform import NoOpTransform, Transform

from detectron2.data.transforms.augmentation import Augmentation
import pdb
import math

import logging
import numpy as np
import pycocotools.mask as mask_util
import torch
from PIL import Image
from collections import defaultdict
import copy
from detectron2.data import transforms as T
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)
from detectron2.utils.file_io import PathManager

__all__ = [
    "BatchResizeShortestEdge",
    "EntityCrop",
]

class BatchResizeTransform(Transform):
    """
    Resize the image to a target size.
    """

    def __init__(self, h, w, new_h, new_w, interp=None):
        """
        Args:
            h, w (int): original image size
            new_h, new_w (int): new image size
            interp: PIL interpolation methods, defaults to bilinear.
        """
        # TODO decide on PIL vs opencv
        super().__init__()
        if interp is None:
            interp = Image.BILINEAR
        self._set_attributes(locals())

    def apply_image(self, imgs, interp=None):
        dim_num = len(imgs.shape)
        assert dim_num == 4
        interp_method = interp if interp is not None else self.interp
        resized_imgs = []
        for img in imgs:
            if len(img.shape) > 2 and img.shape[2] == 1:
                pil_image = Image.fromarray(img[:, :, 0], mode="L")
            else:
                pil_image = Image.fromarray(img)
            pil_image = pil_image.resize((self.new_w, self.new_h), interp_method)
            ret = np.asarray(pil_image)
            if len(img.shape) > 2 and img.shape[2] == 1:
                ret = np.expand_dims(ret, -1)
            resized_imgs.append(ret)
        resized_imgs = np.stack(resized_imgs)
        return resized_imgs

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords
    
    def apply_box(self, boxes):
        boxes = boxes[0]
        new_boxes = super(BatchResizeTransform, self).apply_box(boxes[:,:4])
        boxes[...,:4] = new_boxes
        return boxes[None]

    def apply_segmentation(self, segmentation):
        if len(segmentation.shape)==3:
            segmentation = segmentation[..., None]
            segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
            segmentation = segmentation[..., 0]
        else:
            segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation

class EntityCropTransform(Transform):
    """
    Consectively crop the images
    """
    def __init__(self, crop_axises, crop_indexes):
        super().__init__()
        self._set_attributes(locals())
    
    def apply_image(self, img):
        """
        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255]
        returns:
            ndarray: cropped images
        """
        dim_num = len(img.shape)
        imgs = []
        
        for crop_axis in self.crop_axises:
            x0, y0, x1, y1 = crop_axis
            if dim_num <= 3:
                crop_img = img[y0:y1, x0:x1]
            else:
                crop_img = img[..., y0:y1, x0:x1, :]
            imgs.append(crop_img)

        if dim_num <= 3:
            imgs = np.stack(imgs, axis=0)
        else:
            imgs = np.concatenate(imgs, axis=0)
        return imgs
    
    def apply_coords(self, coords: np.ndarray, x0, y0):
        coords[:, 0] -= x0
        coords[:, 1] -= y0
        return coords
    
    def apply_box(self, box: np.ndarray) -> np.ndarray:
        """
        box: Nx4, [x0, y0, x1, y1]
        """
        idxs = np.array([(0, 1), (2, 1), (0, 3), (2, 3)]).flatten()
        coords = np.asarray(box).reshape(-1, 4)[:, idxs].reshape(-1, 2)
        split_boxes = []
        crop_ws, crop_hs = [], []
        for crop_axis in self.crop_axises:
            startw, starth, endw, endh = crop_axis
            coords_new = self.apply_coords(copy.deepcopy(coords), startw, starth).reshape((-1, 4, 2))
            minxy = coords_new.min(axis=1)
            maxxy = coords_new.max(axis=1)
            trans_boxes = np.concatenate((minxy, maxxy), axis=1)
            
            crop_ws.append(endw-startw)
            crop_hs.append(endh-starth)
            split_boxes.append(trans_boxes)
        split_boxes = np.stack(split_boxes, axis=1)
        ### clip to the image boundary
        ## assert each crop size is equal
        for crop_index, (crop_w, crop_h) in enumerate(zip(crop_ws, crop_hs)):
            assert crop_w == crop_ws[0], "crop width is not equal, crop_{}: {}, crop_0: {}".format(crop_index, crop_w, crop_ws[0])
            assert crop_h == crop_hs[0], "crop height is not equal, crop_{}: {}, crop_0: {}".format(crop_index, crop_h, crop_hs[0])
        crop_w = crop_ws[0]
        crop_h = crop_hs[0]
        # pdb.set_trace()
        split_boxes[...,0::2] = np.clip(split_boxes[...,0::2], 0, crop_w)
        split_boxes[...,1::2] = np.clip(split_boxes[...,1::2], 0, crop_h)
        valid_inds = (split_boxes[...,2]>split_boxes[...,0]) & (split_boxes[...,3]>split_boxes[...,1])
        split_infos = np.concatenate((split_boxes, valid_inds[...,None]), axis=-1)
        return split_infos

class BatchResizeShortestEdge(Augmentation):
    """
    Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    def __init__(
        self, short_edge_length, max_size=sys.maxsize, sample_style="range", interp=Image.BILINEAR
    ):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
        """
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style

        self.is_range = sample_style == "range"
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        if self.is_range:
            assert len(short_edge_length) == 2, (
                "short_edge_length must be two values using 'range' sample style."
                f" Got {short_edge_length}!"
            )
        self._init(locals())

    def get_transform(self, image):
        dim_num = len(image.shape)
        assert dim_num == 4, "the tensor should be in [B, H, W, C]"
        h, w = image.shape[1:3]
        if self.is_range:
            size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
        else:
            size = np.random.choice(self.short_edge_length)
        if size == 0:
            return NoOpTransform()

        scale = size * 1.0 / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return BatchResizeTransform(h, w, newh, neww, self.interp)

class EntityCrop(Augmentation):
    def __init__(self, crop_ratio, stride_ratio, sample_num, is_train):
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        h, w = image.shape[:2]
        crop_axises, crop_indexes = self.get_crop_axises((h, w))
        transform = EntityCropTransform(crop_axises, crop_indexes)
        return transform
    
    def get_crop_axises(self, image_size):
        h, w = image_size
        crop_w = int(self.crop_ratio*w)
        crop_h = int(self.crop_ratio*h)
        # if self.is_train:
        stride_w = int(self.stride_ratio*w)
        stride_h = int(self.stride_ratio*h)
        # pdb.set_trace()

        crop_axises  = []
        for starth in range(0, h, stride_h):
            for startw in range(0, w, stride_w):
                endh = min(starth+crop_h, h)
                endw = min(startw+crop_w, w)
                starth = int(endh-crop_h)
                startw = int(endw-crop_w)
                crop_axises.append([startw, starth, endw, endh])
        if self.is_train:
            crop_indexes = random.sample([i for i in range(len(crop_axises))], self.sample_num)
            crop_axises = [crop_axises[i] for i in crop_indexes]
        else:
            crop_indexes = [i for i in range(self.sample_num)]
        # left_upper   = [0, 0, crop_w, crop_h]
        # right_upper  = [w-crop_w, 0, w, crop_h]
        # left_bottom  = [0, h-crop_h, crop_w, h]
        # right_bottom = [w-crop_w, h-crop_h, w, h]
        
        # crop_axises = [left_upper, right_upper, left_bottom, right_bottom]
        # crop_indexes = [0,1,2,3]
        assert len(crop_axises)==len(crop_indexes)
        return crop_axises, crop_indexes

def transform_instance_annotations_crop(
    annotation, transforms, image_size, *, keypoint_hflip_indices=None
):
    """
    Apply transforms to box, segmentation and keypoints annotations of a single instance.

    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
            It will be modified in-place.
        transforms (TransformList or list[Transform]):
        image_size (tuple): the height, width of the transformed image
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.

    Returns:
        dict:
            the same input dict with fields "bbox", "segmentation", "keypoints"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)
    # bbox is 1d (per-instance bounding box)
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    
    # clip transformed bbox to image size
    bboxes_info = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
    annotation["bbox"] = np.minimum(bbox, list(image_size + image_size)[::-1])
    annotation["bbox"] = bboxes_info[...,:4]
    annotation["bbox_mode"] = BoxMode.XYXY_ABS
    annotation["bbox_valid"] = bboxes_info[...,4]
    for transform_type in transforms:
        if isinstance(transform_type, EntityCropTransform):
            annotation["crop_axises"] = transform_type.crop_axises
            annotation["crop_indexes"] = transform_type.crop_indexes

    if "segmentation" in annotation:
        segm = annotation["segmentation"]
        assert isinstance(segm, dict), "requiring segmentation encoding -> RLE"
        # RLE
        mask = mask_util.decode(segm)
        mask = transforms.apply_segmentation(mask)
        annotation["segmentation"] = mask
    return annotation

def annotations_to_instances_crop(annos, image_size, mask_format="polygon", return_indexes=False):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes",
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    ###
    all_boxes = []
    all_boxes_valid = []
    all_classes = []
    all_segmentations = []
    all_iscrowds = []
    # pdb.set_trace()
    annos_num = len(annos)
    patches_num = len(annos[0]["bbox"])
    for ann_index, obj in enumerate(annos):
        for split_index in range(len(obj["bbox"])):
            all_boxes.append(BoxMode.convert(obj["bbox"][split_index], obj["bbox_mode"], BoxMode.XYXY_ABS))
            all_boxes_valid.append(obj["bbox_valid"][split_index])
            all_classes.append(obj["category_id"])
            all_segmentations.append(obj["segmentation"][split_index])
            all_iscrowds.append(obj["iscrowd"])
            # print("ann_index:{}, split_index:{}".format(ann_index, split_index))
    
    new_targets = []
    crop_axises = annos[0]["crop_axises"]
    # pdb.set_trace()
    crop_size = (crop_axises[0][3], crop_axises[0][2])
    crop_axises = torch.tensor(crop_axises)
    
    for split_index in range(patches_num):
        new_targets.append(Instances(crop_size))
        # pdb.set_trace()
        ## boxes
        new_targets[-1].gt_boxes = Boxes(all_boxes[split_index::patches_num])
        new_targets[-1].gt_boxes_valid = torch.tensor(all_boxes_valid[split_index::patches_num], dtype=torch.int64)
        ## categories
        new_targets[-1].gt_classes = torch.tensor(all_classes[split_index::patches_num], dtype=torch.int64)

        ## masks
        if "segmentation" in annos[0]:
            new_targets[-1].gt_masks = BitMasks(torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in all_segmentations[split_index::patches_num]]))
        
    # pdb.set_trace()
    if return_indexes:
        return new_targets, crop_axises, annos[0]["crop_indexes"]
    else:
        return new_targets, crop_axises

class EntityCascadedCrop(Augmentation):
    def __init__(self, crop_ratio, stride_ratio, sample_num, cascade_num, is_train):
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        h, w = image.shape[:2]
        crop_axises, crop_indexes = self.get_crop_axises((h, w))
        transform = EntityCropTransform(crop_axises, crop_indexes)
        return transform
    
    def get_crop_axises(self, image_size):
        h, w = image_size
        # for i in range(self.cascade_num):
        #     crop_w = int((self.crop_ratio**(i+1))*w)
        #     crop_h = int((self.crop_ratio**(i+1))*h)
        #     stride_w = int((self.stride_ratio**(i+1))*w)
        #     stride_h = int((self.stride_ratio**(i+1))*h)
        #     crop_axises = []
        #     if i==0:

        #     for starth in range(0, )


        crop_axises  = []
        for starth in range(0, h, stride_h):
            for startw in range(0, w, stride_w):
                endh = min(starth+crop_h, h)
                endw = min(startw+crop_w, w)
                starth = int(endh-crop_h)
                startw = int(endw-crop_w)
                crop_axises.append([startw, starth, endw, endh])
        if self.is_train:
            crop_indexes = random.sample([i for i in range(len(crop_axises))], self.sample_num)
            crop_axises = [crop_axises[i] for i in crop_indexes]
        else:
            crop_indexes = [i for i in range(self.sample_num)]
        # left_upper   = [0, 0, crop_w, crop_h]
        # right_upper  = [w-crop_w, 0, w, crop_h]
        # left_bottom  = [0, h-crop_h, crop_w, h]
        # right_bottom = [w-crop_w, h-crop_h, w, h]
        
        # crop_axises = [left_upper, right_upper, left_bottom, right_bottom]
        # crop_indexes = [0,1,2,3]
        assert len(crop_axises)==len(crop_indexes)
        return crop_axises, crop_indexes