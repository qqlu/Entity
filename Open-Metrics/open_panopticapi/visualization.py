#!/usr/bin/env python2
'''
Visualization demo for panoptic COCO sample_data

The code shows an example of color generation for panoptic data (with
"generate_new_colors" set to True). For each segment distinct color is used in
a way that it close to the color of corresponding semantic class.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os, sys
import numpy as np
import json
import argparse

import PIL.Image as Image
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

from panopticapi.utils import IdGenerator, rgb2id

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.utils.logger import setup_logger
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.datasets.builtin_meta import ADE20K_PAN_SEG_CATEGORIES

from OPSNet.mask2former import add_maskformer2_config

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == "__main__":
    #mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    # whether from the PNG are used or new colors are generated
    generate_new_colors = True

    json_file = '/group/20018/gavinqi/zhao/OPSNet/output/inference/ade_predictions.json'
    segmentations_folder = '/group/20018/gavinqi/zhao/OPSNet/output/'
    img_folder = '/group/20018/gavinqi/zhao/datasets/ADEChallengeData2016/images/validation'
    panoptic_coco_categories = './panoptic_coco_categories.json'

    with open(json_file, 'r') as f:
        coco_d = json.load(f)

    categegories = {category['id']: category for category in ADE20K_PAN_SEG_CATEGORIES}

    # find input img that correspond to the annotation
    img = None
    print('+'*100)
    for pred_ann in coco_d['annotations'][:10]: # [{'id': 'ADE_val_00001896', 'file_name': 'ADE_val_00001896.jpg', 'width': 512, 'height': 774},]
        image_id = pred_ann['image_id']
        img = np.array(Image.open(os.path.join(img_folder, image_id+'.jpg')))
        segmentation = np.array(
            Image.open(os.path.join(segmentations_folder, image_id+'.png')),
            dtype=np.uint8
        )
        segmentation_id = rgb2id(segmentation)

        # find segments boundaries
        boundaries = find_boundaries(segmentation_id, mode='thick')

        if generate_new_colors:
            segmentation[:, :, :] = 0
            color_generator = IdGenerator(categegories)
            segments_info = pred_ann['segments_info']
            print('*'*100)
            print(segments_info)
            for segment_info in segments_info:
                color = color_generator.get_color(segment_info['category_id'])
                mask = segmentation_id == segment_info['id']
                segmentation[mask] = color

        # depict boundaries
        segmentation[boundaries] = [0, 0, 0]
        metadata = MetadataCatalog.get(
                    cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
                )
        instance_mode_local = ColorMode.IMAGE,
        visualizer = Visualizer(image, metadata, instance_mode=instance_mode_local)
        vis_output = visualizer.draw_panoptic_seg_predictions(
            segmentation_id, segments_info
        )