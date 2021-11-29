#!/usr/bin/env python
'''
The script uses a simple procedure to combine semantic segmentation and instance
segmentation predictions. The procedure is described in section 7 of the
panoptic segmentation paper https://arxiv.org/pdf/1801.00868.pdf.

On top of the procedure described in the paper. This script remove from
prediction small segments of stuff semantic classes. This addition allows to
decrease number of false positives.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os, sys
import argparse
import numpy as np
from collections import defaultdict
import json
import time
import multiprocessing
import copy

from panopticapi.utils import IdGenerator, id2rgb, save_json

import PIL.Image     as Image

try:
    from pycocotools import mask as COCOmask
except Exception:
    raise Exception("Please install pycocotools module from https://github.com/cocodataset/cocoapi")


def combine_to_panoptic_single_core(proc_id, img_ids, img_id2img, inst_by_image,
                                    sem_by_image, segmentations_folder, overlap_thr,
                                    stuff_area_limit, categories):
    panoptic_json = []
    id_generator = IdGenerator(categories)

    for idx, img_id in enumerate(img_ids):
        img = img_id2img[img_id]

        if idx % 100 == 0:
            print('Core: {}, {} from {} images processed.'.format(proc_id, idx,
                                                                  len(img_ids)))

        pan_segm_id = np.zeros((img['height'],
                                img['width']), dtype=np.uint32)
        used = None
        annotation = {}
        try:
            annotation['image_id'] = int(img_id)
        except Exception:
            annotation['image_id'] = img_id

        annotation['file_name'] = img['file_name'].replace('.jpg', '.png')

        segments_info = []
        for ann in inst_by_image[img_id]:
            area = COCOmask.area(ann['segmentation'])
            if area == 0:
                continue
            if used is None:
                intersect = 0
                used = copy.deepcopy(ann['segmentation'])
            else:
                intersect = COCOmask.area(
                    COCOmask.merge([used, ann['segmentation']], intersect=True)
                )
            if intersect / area > overlap_thr:
                continue
            used = COCOmask.merge([used, ann['segmentation']], intersect=False)

            mask = COCOmask.decode(ann['segmentation']) == 1
            if intersect != 0:
                mask = np.logical_and(pan_segm_id == 0, mask)
            segment_id = id_generator.get_id(ann['category_id'])
            panoptic_ann = {}
            panoptic_ann['id'] = segment_id
            panoptic_ann['category_id'] = ann['category_id']
            pan_segm_id[mask] = segment_id
            segments_info.append(panoptic_ann)

        for ann in sem_by_image[img_id]:
            mask = COCOmask.decode(ann['segmentation']) == 1
            mask_left = np.logical_and(pan_segm_id == 0, mask)
            if mask_left.sum() < stuff_area_limit:
                continue
            segment_id = id_generator.get_id(ann['category_id'])
            panoptic_ann = {}
            panoptic_ann['id'] = segment_id
            panoptic_ann['category_id'] = ann['category_id']
            pan_segm_id[mask_left] = segment_id
            segments_info.append(panoptic_ann)

        annotation['segments_info'] = segments_info
        panoptic_json.append(annotation)

        Image.fromarray(id2rgb(pan_segm_id)).save(
            os.path.join(segmentations_folder, annotation['file_name'])
        )

    return panoptic_json


def combine_to_panoptic_multi_core(img_id2img, inst_by_image,
                                   sem_by_image, segmentations_folder, overlap_thr,
                                   stuff_area_limit, categories):
    cpu_num = multiprocessing.cpu_count()
    img_ids_split = np.array_split(list(img_id2img), cpu_num)
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(img_ids_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, img_ids in enumerate(img_ids_split):
        p = workers.apply_async(combine_to_panoptic_single_core,
                                (proc_id, img_ids, img_id2img, inst_by_image,
                                 sem_by_image, segmentations_folder, overlap_thr,
                                 stuff_area_limit, categories))
        processes.append(p)
    panoptic_json = []
    for p in processes:
        panoptic_json.extend(p.get())
    return panoptic_json


def combine_predictions(semseg_json_file, instseg_json_file, images_json_file,
                        categories_json_file, segmentations_folder,
                        panoptic_json_file, confidence_thr, overlap_thr,
                        stuff_area_limit):
    start_time = time.time()

    with open(semseg_json_file, 'r') as f:
        sem_results = json.load(f)
    with open(instseg_json_file, 'r') as f:
        inst_results = json.load(f)
    with open(images_json_file, 'r') as f:
        images_d = json.load(f)
    img_id2img = {img['id']: img for img in images_d['images']}

    with open(categories_json_file, 'r') as f:
        categories_list = json.load(f)
    categories = {el['id']: el for el in categories_list}

    if segmentations_folder is None:
        segmentations_folder = panoptic_json_file.rsplit('.', 1)[0]
    if not os.path.isdir(segmentations_folder):
        print("Creating folder {} for panoptic segmentation PNGs".format(segmentations_folder))
        os.mkdir(segmentations_folder)

    print("Combining:")
    print("Semantic segmentation:")
    print("\tJSON file: {}".format(semseg_json_file))
    print("and")
    print("Instance segmentations:")
    print("\tJSON file: {}".format(instseg_json_file))
    print("into")
    print("Panoptic segmentations:")
    print("\tSegmentation folder: {}".format(segmentations_folder))
    print("\tJSON file: {}".format(panoptic_json_file))
    print("List of images to combine is takes from {}".format(images_json_file))
    print('\n')

    inst_by_image = defaultdict(list)
    for inst in inst_results:
        if inst['score'] < confidence_thr:
            continue
        inst_by_image[inst['image_id']].append(inst)
    for img_id in inst_by_image.keys():
        inst_by_image[img_id] = sorted(inst_by_image[img_id], key=lambda el: -el['score'])

    sem_by_image = defaultdict(list)
    for sem in sem_results:
        if categories[sem['category_id']]['isthing'] == 1:
            continue
        sem_by_image[sem['image_id']].append(sem)

    panoptic_json = combine_to_panoptic_multi_core(
        img_id2img,
        inst_by_image,
        sem_by_image,
        segmentations_folder,
        overlap_thr,
        stuff_area_limit,
        categories
    )

    with open(images_json_file, 'r') as f:
        coco_d = json.load(f)
    coco_d['annotations'] = panoptic_json
    coco_d['categories'] = list(categories.values())
    save_json(coco_d, panoptic_json_file)

    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script uses a simple procedure to combine semantic \
        segmentation and instance segmentation predictions. See this \
        file's head for more information."
    )
    parser.add_argument('--semseg_json_file', type=str,
                        help="JSON file with semantic segmentation predictions")
    parser.add_argument('--instseg_json_file', type=str,
                        help="JSON file with instance segmentation predictions")
    parser.add_argument('--images_json_file', type=str,
                        help="JSON file with correponding image set information")
    parser.add_argument('--categories_json_file', type=str,
                        help="JSON file with Panoptic COCO categories information",
                        default='./panoptic_coco_categories.json')
    parser.add_argument('--panoptic_json_file', type=str,
                        help="JSON file with resulting COCO panoptic format prediction")
    parser.add_argument(
        '--segmentations_folder', type=str, default=None, help="Folder with \
         panoptic COCO format segmentations. Default: X if panoptic_json_file is \
         X.json"
    )
    parser.add_argument('--confidence_thr', type=float, default=0.5,
                        help="Predicted segments with smaller confidences than the threshold are filtered out")
    parser.add_argument('--overlap_thr', type=float, default=0.5,
                        help="Segments that have higher that the threshold ratio of \
                        their area being overlapped by segments with higher confidence are filtered out")
    parser.add_argument('--stuff_area_limit', type=float, default=64*64,
                        help="Stuff segments with area smaller that the limit are filtered out")
    args = parser.parse_args()
    combine_predictions(args.semseg_json_file,
                        args.instseg_json_file,
                        args.images_json_file,
                        args.categories_json_file,
                        args.segmentations_folder,
                        args.panoptic_json_file,
                        args.confidence_thr,
                        args.overlap_thr,
                        args.stuff_area_limit)
