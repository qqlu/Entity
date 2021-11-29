from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os, sys
import numpy as np
import pdb
import mmcv
import copy
import cv2
from collections import OrderedDict
from pycocotools.coco import COCO
import pycocotools.mask as mask_utils

import PIL.Image as Image
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
from panopticapi.utils import IdGenerator, rgb2id
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

thread_num   = int(sys.argv[1])
thread_idx   = int(sys.argv[2])
type_        = sys.argv[3]

OFFSET = 256 * 256 * 256

GT_base_path             = "/data/ceph/gavinqi/data/coco"
GT_panoptic_png_path     = os.path.join(GT_base_path, "panoptic_{}".format(type_))
GT_panoptic_json_path    = os.path.join(GT_base_path, "annotations/panoptic_{}.json".format(type_))
GT_instance_json_path    = os.path.join(GT_base_path, "annotations/instances_{}.json".format(type_))
save_base_path           = os.path.join(GT_base_path, "entity_{}".format(type_))

if not os.path.exists(save_base_path):
    os.makedirs(save_base_path)

coco_g          = mmcv.load(GT_panoptic_json_path)
categories_list = COCO_CATEGORIES
catid_map       = {category['id']: [cid, category["isthing"]] for cid, category in enumerate(categories_list)}
idcat_map = {}
for key, value in catid_map.items():
    idcat_map[value[0]] = [key,value[1]]

name2panopticindex = OrderedDict()
id2name            = OrderedDict()

for i_index, image_info in enumerate(coco_g["images"]):
    file_name = image_info["file_name"].split(".")[0]
    name2panopticindex[file_name] = {"i_index": i_index}
    id2name[image_info["id"]] = file_name

for a_index, ann in enumerate(coco_g["annotations"]):
    file_name = id2name[ann["image_id"]]
    name2panopticindex[file_name]["a_index"] = a_index
print("build name to panoptic index finished")

# imgs and instance_anns
instances_api      = COCO(GT_instance_json_path)
img_ids            = instances_api.getImgIds()
imgs               = instances_api.loadImgs(img_ids)
instance_anns      = [instances_api.imgToAnns[img_id] for img_id in img_ids]
assert len(name2panopticindex.keys()) == len(imgs)
imgs_instancesanns = list(zip(imgs, instance_anns))
print("build imgs and instance_anns finished")

for img_index, (img_dict, ann_dict_list) in enumerate(imgs_instancesanns):
    if img_index % thread_num != thread_idx:
        continue
    
    file_name        = img_dict["file_name"].split(".")[0]
    image_h, image_w = img_dict["height"], img_dict["width"]

    ## panoptic mask from panoptic annotation
    panoptic_i_index, panoptic_a_index = name2panopticindex[file_name]["i_index"], name2panopticindex[file_name]["a_index"]
    panoptic_img_infos = coco_g["images"][panoptic_i_index]
    panoptic_ann_infos = coco_g["annotations"][panoptic_a_index]
    assert panoptic_img_infos["file_name"].split(".")[0] == file_name, "Something wrong with panoptic_img_infos"
    assert panoptic_ann_infos["file_name"].split(".")[0] == file_name, "Something wrong with panoptic_ann_infos"

    panoptic              = np.array(Image.open(os.path.join(GT_panoptic_png_path, file_name+".png")), dtype=np.uint8)
    panoptic_id           = rgb2id(panoptic)
    panoptic_entity_id    = np.zeros(panoptic_id.shape, dtype=np.uint8)
    panoptic_class_id     = np.zeros(panoptic_id.shape, dtype=np.uint8) + 255
    unique_panoptic_id    = np.unique(panoptic_id)

    for ii, segment_info in enumerate(panoptic_ann_infos["segments_info"]):
        if segment_info["iscrowd"] == 1:
            continue
        old_entity_id     = segment_info["id"]
        new_entity_id     = ii + 1
        category          = segment_info["category_id"]
        panoptic_entity_id[panoptic_id==old_entity_id] = new_entity_id
        panoptic_class_id[panoptic_id==old_entity_id]  = catid_map[category][0]
    
    unique_ids        = np.unique(panoptic_entity_id)
    count = 1

    bounding_box = []
    for entity_id in unique_ids:
        if entity_id == 0:
            continue
        mask     = (panoptic_entity_id==entity_id).astype(np.uint8)
        category = int(np.unique(panoptic_class_id[panoptic_entity_id==entity_id]))

        finds_y, finds_x = np.where(mask==1)
        y1 = int(np.min(finds_y))
        y2 = int(np.max(finds_y))
        x1 = int(np.min(finds_x))
        x2 = int(np.max(finds_x))
        thing_or_stuff = int(idcat_map[category][1])
        bounding_box.append([x1,y1,x2,y2,category,thing_or_stuff,entity_id])
    
    bounding_box = np.array(bounding_box)

    panoptic_info = np.stack((panoptic_entity_id, panoptic_class_id), axis=0)
    np.savez(os.path.join(save_base_path, file_name),map=panoptic_info, bounding_box=bounding_box)

    print("{}, {}, {}".format(thread_idx, img_index, file_name))
    
