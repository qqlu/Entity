import inspect
import numpy as np
import pprint
from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple, Union
from fvcore.transforms.transform import Transform, TransformList
from detectron2.data.transforms import AugInput
from detectron2.data.transforms.augmentation import _check_img_dtype

class ItemAugInput(AugInput):
    """
    A standard implementation of :class:`AugInput` for the majority of use cases.
    This class provides the following standard attributes that are common to use by
    Augmentation (augmentation policies). These are chosen because most
    :class:`Augmentation` won't need anything more to define a augmentation policy.
    After applying augmentations to these special attributes, the returned transforms
    can then be used to transform other data structures that users have.

    Attributes:
        image (ndarray): image in HW or HWC format. The meaning of C is up to users
        boxes (ndarray or None): Nx4 boxes in XYXY_ABS mode
        sem_seg (ndarray or None): HxW semantic segmentation mask

    Examples:
    ::
        input = StandardAugInput(image, boxes=boxes)
        tfms = input.apply_augmentations(list_of_augmentations)
        transformed_image = input.image
        transformed_boxes = input.boxes
        transformed_other_data = tfms.apply_other(other_data)

    An extended project that works with new data types may require augmentation
    policies that need more inputs. An algorithm may need to transform inputs
    in a way different from the standard approach defined in this class. In those
    situations, users can implement new subclasses of :class:`AugInput` with differnt
    attributes and the :meth:`transform` method.
    """
    def __init__(
        self,
        image: np.ndarray,
        *,
        boxes = None,
        seg_info= None,
    ):
        """
        Args:
            image: (H,W) or (H,W,C) ndarray of type uint8 in range [0, 255], or
                floating point in range [0, 1] or [0, 255].
            boxes: (N,4) ndarray of float32. It represents the instance bounding boxes
                of N instances. Each is in XYXY format in unit of absolute coordinates.
            sem_seg: (H,W) ndarray of type uint8. Each element is an integer label of pixel.
        """
        _check_img_dtype(image)
        self.image = image
        self.boxes = boxes
        self.seg_info = seg_info

    def transform(self, tfm: Transform) -> None:
        """
        In-place transform all attributes of this class.
        """
        self.image = tfm.apply_image(self.image)
        if self.boxes is not None:
            self.boxes = tfm.apply_box(self.boxes)
        if self.seg_info is not None:
            assert type(self.seg_info) == dict, "seg_info is dictionary"
            for key, value in self.seg_info.items():
                self.seg_info[key] = tfm.apply_segmentation(value)