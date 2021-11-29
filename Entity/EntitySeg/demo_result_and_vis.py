# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import numpy as np
import copy

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.engine import default_setup

from entityseg import *

from predictor import VisualizationDemo
import pdb

# constants
WINDOW_NAME = "Image Segmentation"

def make_colors():
    from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
    colors = []
    for cate in COCO_CATEGORIES:
        colors.append(cate["color"])
    return colors

def mask_to_boundary(mask, dilation_ratio=0.0008):
	"""
	Convert binary mask to boundary mask.
	:param mask (numpy array, uint8): binary mask
	:param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
	:return: boundary mask (numpy array)
	"""
	h, w = mask.shape
	img_diag = np.sqrt(h ** 2 + w ** 2)
	dilation = int(round(dilation_ratio * img_diag))
	if dilation < 1:
	    dilation = 1
	# Pad image so mask truncated by the image border is also considered as boundary.
	new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
	kernel = np.ones((3, 3), dtype=np.uint8)
	new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
	mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
	# G_d intersects G in the paper.
	return mask - mask_erode


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_entity_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    default_setup(cfg, args)
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
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
        default=0.2,
        help="Minimum score for instance predictions to be shown",
    )

    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
        "See config references at "
        "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    colors = make_colors()

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            data = demo.run_on_image_wo_vis(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(data[0])),
                    time.time() - start_time,
                )
            )

            if os.path.isdir(args.output):
                assert os.path.isdir(args.output), args.output
                out_filename = os.path.join(args.output, os.path.basename(path))
            else:
                assert len(args.input) == 1, "Please specify a directory with args.output"
                out_filename = args.output
            ## save inference result, [0] original score by detection head, [1] mask rescoring score, [2] mask_id
            ori_scores = data[0]
            scores = data[1]
            mask_id = data[2]
            np.savez(out_filename.split(".")[0]+".npz", ori_scores=ori_scores, scores=scores, mask_id=mask_id)

            ## save visualization
            img_for_paste = copy.deepcopy(img)
            color_mask     = copy.deepcopy(img)
            masks_edge     = np.zeros(img.shape[:2], dtype=np.uint8)
            alpha  = 0.4
            count  = 0
            for index, score in enumerate(scores):
                if score <= args.confidence_threshold:
                    break
                color_mask[mask_id==count] = colors[count]
                boundary = mask_to_boundary((mask_id==count).astype(np.uint8))
                masks_edge[boundary>0] = 1
                count += 1
            img_wm = cv2.addWeighted(img_for_paste, alpha, color_mask, 1-alpha, 0)
            img_wm[masks_edge==1] = 0
            fvis = np.concatenate((img, img_wm))
            cv2.imwrite(out_filename.split(".")[0]+".jpg",fvis)



                





