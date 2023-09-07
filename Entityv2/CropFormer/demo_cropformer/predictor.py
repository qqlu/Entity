# Copyright (c) Facebook, Inc. and its affiliates.
# Copied from: https://github.com/facebookresearch/detectron2/blob/master/demo/predictor.py
import atexit
import bisect
import multiprocessing as mp
from collections import deque

import pdb
import cv2
import copy
import torch
import numpy as np

import detectron2.data.transforms as T
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer

from mask2former.data.dataset_mappers.crop_augmentations import BatchResizeShortestEdge, EntityCrop, EntityCropTransform


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = CropFormerPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        predictions = self.predictor(image)
        return predictions

class CropFormerPredictor(DefaultPredictor):
    """
    """

    def __init__(self, cfg):
        super().__init__(cfg)
    
    def generate_img_augs(self):
        shortest_side = np.random.choice([self.cfg.INPUT.MIN_SIZE_TEST])

        augs = [
            T.ResizeShortestEdge(
                (shortest_side,),
                self.cfg.INPUT.MAX_SIZE_TEST,
                self.cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            ),
            
        ]

        # Build original image augmentation
        crop_augs = []
        entity_crops = EntityCrop(self.cfg.ENTITY.CROP_AREA_RATIO, 
                                    self.cfg.ENTITY.CROP_STRIDE_RATIO,
                                    self.cfg.ENTITY.CROP_SAMPLE_NUM_TEST, 
                                    False)
        crop_augs.append(entity_crops)
        
        entity_resize = BatchResizeShortestEdge((shortest_side,), self.cfg.INPUT.MAX_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING)
        crop_augs.append(entity_resize)

        # augs      = T.AugmentationList(augs)
        crop_augs = T.AugmentationList(crop_augs)
        return augs, crop_augs

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            
            # build cropformer augmentations
            augs, crop_augs = self.generate_img_augs()

            height, width = original_image.shape[:2]
            aug_input_ori = T.AugInput(copy.deepcopy(original_image))

            aug_input_ori, _ = T.apply_transform_gens(augs, aug_input_ori)
            image_ori = aug_input_ori.image
            image_ori = torch.as_tensor(image_ori.astype("float32").transpose(2, 0, 1))

            aug_input_crop = T.AugInput(copy.deepcopy(original_image))
            transforms_crop = crop_augs(aug_input_crop)
            image_crop = aug_input_crop.image
            assert len(image_crop.shape)==4, "the image shape must be [N, H, W, C]"
            image_crop = torch.as_tensor(image_crop.astype("float32").transpose(0, 3, 1, 2))
            
            for transform_type in transforms_crop:
                if isinstance(transform_type, EntityCropTransform):
                    crop_axises = transform_type.crop_axises
                    crop_indexes = transform_type.crop_indexes

            inputs = {"image": image_ori, 
                      "height": height, 
                      "width": width,
                      "image_crop": image_crop,
                      "crop_region": crop_axises,
                      "crop_indexes": crop_indexes
                      }
            # pdb.set_trace()
            predictions = self.model([inputs])[0]
            return predictions

class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = CropFormerPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
