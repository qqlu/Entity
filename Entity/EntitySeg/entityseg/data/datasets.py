# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import pdb

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES, _get_builtin_metadata
from detectron2.data.datasets import load_coco_json, register_coco_instances
import torchvision

_root = os.getenv("DETECTRON2_DATASETS", "datasets")

SPLITS = {}
SPLITS["coco_2017_train_entity"]  = ("coco/train2017", "coco/annotations/entity_train2017.json")
SPLITS["coco_2017_val_entity"]    = ("coco/val2017", "coco/annotations/entity_val2017.json")

def _get_coco_trans_meta():
	oc2nc_map = {category['id']: [cid, category["isthing"], category["name"], category["color"]] for cid, category in enumerate(COCO_CATEGORIES)}
	NEW_COCO_CATEGORIES = []
	for key, value in oc2nc_map.items():
		new_info = {"id": value[0], "isthing": value[1], "name": value[2], "color": value[3]}
		NEW_COCO_CATEGORIES.append(new_info)
	
	thing_ids     = [k["id"] for k in NEW_COCO_CATEGORIES]
	thing_colors  = [k["color"] for k in NEW_COCO_CATEGORIES]
	thing_classes = [k["name"] for k in NEW_COCO_CATEGORIES]
	thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
	ret = {
		"thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
	return ret


for key, (image_root, json_file) in SPLITS.items():
	register_coco_instances(key,
		_get_coco_trans_meta(),
		os.path.join(_root, json_file) if "://" not in json_file else json_file,
		os.path.join(_root, image_root)
	)
