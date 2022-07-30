# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import pdb

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES, _get_builtin_metadata
from detectron2.data.datasets import load_coco_json, register_coco_instances
import torchvision

_root = os.getenv("DETECTRON2_DATASETS", "datasets")

SPLITS = {}
SPLITS["coco_2017_train_entity"]          = ("coco/train2017", "coco/annotations/entity_train2017.json")
SPLITS["coco_2017_val_entity"]            = ("coco/val2017", "coco/annotations/entity_val2017.json")
SPLITS["coco_2017_unlabeled_enity"]       = ("coco/unlabel2017", "coco/annotations/entity_unlabel2017.json")
SPLITS["coco_2017_val_entity_rle"]        = ("coco/val2017", "coco/annotations/entity_val2017.json")
SPLITS["coco_2017_train_entity_rle"]      = ("coco/train2017", "coco/annotations/entity_train2017_RLE.json")

for i in [0,1,3]:
    SPLITS["places_{}_10_train_entity".format(i)] = ("places/data_large", "places/annotations/annotations_unlabeled_{}_10.json".format(i))

for i in [2,4,5,6,7,8,9]:
    SPLITS["places_{}_10_train_entity".format(i)] = ("places/data_large", "places/annotations/annotations_unlabeled_{}_10_new.json".format(i))

for i in [0,1,2,3,4,5,6,7,8,9]:
    SPLITS["openimage_{}_10_train_entity".format(i)] = ("openimage/data", "openimage/annotations/annotations_unlabeled_{}_10.json".format(i))

def _get_coco_trans_meta():
	oc2nc_map = {category['id']: [cid, category["isthing"], category["name"], category["color"]] for cid, category in enumerate(COCO_CATEGORIES)}
	NEW_COCO_CATEGORIES = []
	for key, value in oc2nc_map.items():
		new_info = {"id": value[0], "isthing": value[1], "name": value[2], "color": value[3]}
		NEW_COCO_CATEGORIES.append(new_info)
	
	thing_ids     = [k["id"]    for k in NEW_COCO_CATEGORIES]
	thing_colors  = [k["color"] for k in NEW_COCO_CATEGORIES]
	thing_classes = [k["name"]  for k in NEW_COCO_CATEGORIES]
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

ADE20K_CATEGORIES = [
{"color":[220,20,60], "isthing": 1, "id": 1, "name": "wall"},
{"color":[119,11,32], "isthing": 1, "id": 2, "name": "building"},
{"color":[0,0,142], "isthing": 1, "id": 3, "name": "sky"},
{"color":[0,0,230], "isthing": 1, "id": 4, "name": "floor"},
{"color":[106,0,228], "isthing": 1, "id": 5, "name": "tree"},
{"color":[0,60,100], "isthing": 1, "id": 6, "name": "ceiling"},
{"color":[0,80,100], "isthing": 1, "id": 7, "name": "road"},
{"color":[0,0,70], "isthing": 1, "id": 8, "name": "bed "},
{"color":[0,0,192], "isthing": 1, "id": 9, "name": "windowpane"},
{"color":[250,170,30], "isthing": 1, "id": 10, "name": "grass"},
{"color":[100,170,30], "isthing": 1, "id": 11, "name": "cabinet"},
{"color":[220,220,0], "isthing": 1, "id": 12, "name": "sidewalk"},
{"color":[175,116,175], "isthing": 1, "id": 13, "name": "person"},
{"color":[250,0,30], "isthing": 1, "id": 14, "name": "earth"},
{"color":[165,42,42], "isthing": 1, "id": 15, "name": "door"},
{"color":[255,77,255], "isthing": 1, "id": 16, "name": "table"},
{"color":[0,226,252], "isthing": 1, "id": 17, "name": "mountain"},
{"color":[182,182,255], "isthing": 1, "id": 18, "name": "plant"},
{"color":[0,82,0], "isthing": 1, "id": 19, "name": "curtain"},
{"color":[120,166,157], "isthing": 1, "id": 20, "name": "chair"},
{"color":[110,76,0], "isthing": 1, "id": 21, "name": "car"},
{"color":[174,57,255], "isthing": 1, "id": 22 , "name": "water"},
{"color":[199,100,0], "isthing": 1, "id": 23, "name": "painting"},
{"color":[72,0,118], "isthing": 1, "id": 24 , "name": "sofa"},
{"color":[255,179,240], "isthing": 1, "id": 25 , "name": "shelf"},
{"color":[0,125,92], "isthing": 1, "id": 26 , "name": "house"},
{"color":[209,0,151], "isthing": 1, "id": 27 , "name": "sea"},
{"color":[188,208,182], "isthing": 1, "id": 28, "name": "mirror"},
{"color":[0,220,176], "isthing": 1, "id": 29, "name": "rug"},
{"color":[255,99,164], "isthing": 1, "id": 30, "name": "field"},
{"color":[92,0,73], "isthing": 1, "id": 31, "name": "armchair"},
{"color":[133,129,255], "isthing": 1, "id": 32, "name": "seat"},
{"color":[78,180,255], "isthing": 1, "id": 33, "name": "fence"},
{"color":[0,228,0], "isthing": 1, "id": 34, "name": "desk"},
{"color":[174,255,243], "isthing": 1, "id": 35, "name": "rock"},
{"color":[45,89,255], "isthing": 1, "id": 36, "name": "wardrobe"},
{"color":[134,134,103], "isthing": 1, "id": 37, "name": "lamp"},
{"color":[145,148,174], "isthing": 1, "id": 38, "name": "bathtub"},
{"color":[255,208,186], "isthing": 1, "id": 39, "name": "railing"},
{"color":[197,226,255], "isthing": 1, "id": 40, "name": "cushion"},
{"color":[171,134,1], "isthing": 1, "id": 41, "name": "base"},
{"color":[109,63,54], "isthing": 1, "id": 42, "name": "box"},
{"color":[207,138,255], "isthing": 1, "id": 43, "name": "column"},
{"color":[151,0,95], "isthing": 1, "id": 44, "name": "signboard"},
{"color":[9,80,61], "isthing": 1, "id": 45, "name": "chest of drawers"},
{"color":[84,105,51], "isthing": 1, "id": 46, "name": "counter"},
{"color":[74,65,105], "isthing": 1, "id": 47, "name": "sand"},
{"color":[166,196,102], "isthing": 1, "id": 48, "name": "sink"},
{"color":[208,195,210], "isthing": 1, "id": 49, "name": "skyscraper"},
{"color":[255,109,65], "isthing": 1, "id": 50, "name": "fireplace"},
{"color":[0,143,149], "isthing": 1, "id": 51, "name": "refrigerator"},
{"color":[179,0,194], "isthing": 1, "id": 52, "name": "grandstand"},
{"color":[209,99,106], "isthing": 1, "id": 53, "name": "path"},
{"color":[5,121,0], "isthing": 1, "id": 54, "name": "stairs"},
{"color":[227,255,205], "isthing": 1, "id": 55, "name": "runway"},
{"color":[147,186,208], "isthing": 1, "id": 56, "name": "case"},
{"color":[153,69,1], "isthing": 1, "id": 57, "name": "pool table"},
{"color":[3,95,161], "isthing": 1, "id": 58, "name": "pillow"},
{"color":[163,255,0], "isthing": 1, "id": 59, "name": "screen door"},
{"color":[119,0,170], "isthing": 1, "id": 60, "name": "stairway"},
{"color":[0,182,199], "isthing": 1, "id": 61, "name": "river"},
{"color":[0,165,120], "isthing": 1, "id": 62, "name": "bridge"},
{"color":[183,130,88], "isthing": 1, "id": 63, "name": "bookcase"},
{"color":[95,32,0], "isthing": 1, "id": 64, "name": "blind"},
{"color":[130,114,135], "isthing": 1, "id": 65, "name": "coffee table"},
{"color":[110,129,133], "isthing": 1, "id": 66, "name": "toilet"},
{"color":[166,74,118], "isthing": 1, "id": 67, "name": "flower"},
{"color":[219,142,185], "isthing": 1, "id": 68, "name": "book"},
{"color":[79,210,114], "isthing": 1, "id": 69, "name": "hill"},
{"color":[178,90,62], "isthing": 1, "id": 70, "name": "bench"},
{"color":[65,70,15], "isthing": 1, "id": 71, "name": "countertop"},
{"color":[127,167,115], "isthing": 1, "id": 72, "name": "stove"},
{"color":[59,105,106], "isthing": 1, "id": 73, "name": "palm"},
{"color":[142,108,45], "isthing": 1, "id": 74, "name": "kitchen island"},
{"color":[196,172,0], "isthing": 1, "id": 75, "name": "computer"},
{"color":[95,54,80], "isthing": 1, "id": 76, "name": "swivel chair"},
{"color":[128,76,255], "isthing": 1, "id": 77, "name": "boat"},
{"color":[201,57,1], "isthing": 1, "id": 78, "name": "bar"},
{"color":[246,0,122], "isthing": 1, "id": 79, "name": "arcade machine"},
{"color":[191,162,208], "isthing": 1, "id": 80, "name": "hovel"},
{"color":[255,255,128], "isthing": 1, "id": 81, "name": "bus"},
{"color":[147,211,203], "isthing": 1, "id": 82, "name": "towel"},
{"color":[150,100,100], "isthing": 1, "id": 83, "name": "light"},
{"color":[168,171,172], "isthing": 1, "id": 84, "name": "truck"},
{"color":[146,112,198], "isthing": 1, "id": 85, "name": "tower"},
{"color":[210,170,100], "isthing": 1, "id": 86, "name": "chandelier"},
{"color":[92,136,89], "isthing": 1, "id": 87, "name": "awning"},
{"color":[218,88,184], "isthing": 1, "id": 88, "name": "streetlight"},
{"color":[241,129,0], "isthing": 1, "id": 89, "name": "booth"},
{"color":[217,17,255], "isthing": 1, "id": 90, "name": "television receiver"},
{"color":[124,74,181], "isthing": 1, "id": 91, "name": "airplane"},
{"color":[70,70,70], "isthing": 1, "id": 92, "name": "dirt track"},
{"color":[255,228,255], "isthing": 1, "id": 93, "name": "apparel"},
{"color":[154,208,0], "isthing": 1, "id": 94, "name": "pole"},
{"color":[193,0,92], "isthing": 1, "id": 95, "name": "land"},
{"color":[76,91,113], "isthing": 1, "id": 96, "name": "bannister"},
{"color":[255,180,195], "isthing": 1, "id": 97, "name": "escalator"},
{"color":[106,154,176], "isthing": 1, "id": 98, "name": "ottoman"},
{"color":[230,150,140], "isthing": 1, "id": 99, "name": "bottle"},
{"color":[60,143,255], "isthing": 1, "id": 100, "name": "buffet"},
{"color":[128,64,128], "isthing": 1, "id": 101, "name": "poster"},
{"color":[92,82,55], "isthing": 1, "id": 102, "name": "stage"},
{"color":[254,212,124], "isthing": 1, "id": 103, "name": "van"},
{"color":[73,77,174], "isthing": 1, "id": 104, "name": "ship"},
{"color":[255,160,98], "isthing": 1, "id": 105, "name": "fountain"},
{"color":[255,255,255], "isthing": 1, "id": 106, "name": "conveyer belt"},
{"color":[104,84,109], "isthing": 1, "id": 107, "name": "canopy"},
{"color":[169,164,131], "isthing": 1, "id": 108, "name": "washer"},
{"color":[225,199,255], "isthing": 1, "id": 109, "name": "plaything"},
{"color":[137,54,74], "isthing": 1, "id": 110, "name": "swimming pool"},
{"color":[135,158,223], "isthing": 1, "id": 111, "name": "stool"},
{"color":[7,246,231], "isthing": 1, "id": 112, "name": "barrel"},
{"color":[107,255,200], "isthing": 1, "id": 113, "name": "basket"},
{"color":[58,41,149], "isthing": 1, "id": 114, "name": "waterfall"},
{"color":[183,121,142], "isthing": 1, "id": 115, "name": "tent"},
{"color":[255,73,97], "isthing": 1, "id": 116, "name": "bag"},
{"color":[107,142,35], "isthing": 1, "id": 117, "name": "minibike"},
{"color":[190,153,153], "isthing": 1, "id": 118, "name": "cradle"},
{"color":[146,139,141], "isthing": 1, "id": 119, "name": "oven"},
{"color":[70,130,180], "isthing": 1, "id": 120, "name": "ball"},
{"color":[134,199,156], "isthing": 1, "id": 121, "name": "food"},
{"color":[209,226,140], "isthing": 1, "id": 122, "name": "step"},
{"color":[96,36,108], "isthing": 1, "id": 123, "name": "tank"},
{"color":[96,96,96], "isthing": 1, "id": 124, "name": "trade name"},
{"color":[64,170,64], "isthing": 1, "id": 125, "name": "microwave"},
{"color":[152,251,152], "isthing": 1, "id": 126, "name": "pot"},
{"color":[208,229,228], "isthing": 1, "id": 127, "name": "animal"},
{"color":[206,186,171], "isthing": 1, "id": 128, "name": "bicycle"},
{"color":[152,161,64], "isthing": 1, "id": 129, "name": "lake"},
{"color":[116,112,0], "isthing": 1, "id": 130, "name": "dishwasher"},
{"color":[0,114,143], "isthing": 1, "id": 131, "name": "screen"},
{"color":[102,102,156], "isthing": 1, "id": 132, "name": "blanket"},
{"color":[220,20,60], "isthing": 1, "id": 133, "name": "sculpture"},
{"color":[119,11,32], "isthing": 1, "id": 134, "name": "hood"},
{"color":[0,0,142], "isthing": 1, "id": 135, "name": "sconce"},
{"color":[0,0,230], "isthing": 1, "id": 136, "name": "vase"},
{"color":[106,0,228], "isthing": 1, "id": 137, "name": "traffic light"},
{"color":[0,60,100], "isthing": 1, "id": 138, "name": "tray"},
{"color":[0,80,100], "isthing": 1, "id": 139, "name": "ashcan"},
{"color":[0,0,70], "isthing": 1, "id": 140, "name": "fan"},
{"color":[0,0,192], "isthing": 1, "id": 141, "name": "pier"},
{"color":[250,170,30], "isthing": 1, "id": 142, "name": "crt screen"},
{"color":[100,170,30], "isthing": 1, "id": 143, "name": "plate"},
{"color":[220,220,0], "isthing": 1, "id": 144, "name": "monitor"},
{"color":[175,116,175], "isthing": 1, "id": 145, "name": "bulletin board"},
{"color":[250,0,30], "isthing": 1, "id": 146, "name": "shower"},
{"color":[165,42,42], "isthing": 1, "id": 147, "name": "radiator"},
{"color":[255,77,255], "isthing": 1, "id": 148, "name": "glass"},
{"color":[0,226,252], "isthing": 1, "id": 149, "name": "clock"},
{"color":[182,182,255], "isthing": 1, "id": 150, "name": "flag"},
]


def _get_ade20k_trans_meta():
    oc2nc_map = {category['id']: [cid, category["isthing"], category["name"], category["color"]] for cid, category in enumerate(ADE20K_CATEGORIES)}
    # transformed info
    NEW_ADE20K_CATEGORIES = []
    for key, value in oc2nc_map.items():
        new_info = {"id": value[0], "isthing": value[1], "name": value[2], "color": value[3]}
        NEW_ADE20K_CATEGORIES.append(new_info)

    thing_ids     = [k["id"] for k in NEW_ADE20K_CATEGORIES]
    thing_colors  = [k["color"] for k in NEW_ADE20K_CATEGORIES]
    thing_classes = [k["name"] for k in NEW_ADE20K_CATEGORIES]
    
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}

    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }

    return ret

SPLITS_ADE = {}
SPLITS_ADE["ade20k_train_entity"]  = ("ade20k/train", "ade20k/annotations/ade20k_instances_training.json")
SPLITS_ADE["ade20k_val_entity"]    = ("ade20k/val", "ade20k/annotations/ade20k_instances_validation.json")

for key, (image_root, json_file) in SPLITS_ADE.items():
	register_coco_instances(key,
		_get_ade20k_trans_meta(),
		os.path.join(_root, json_file) if "://" not in json_file else json_file,
		os.path.join(_root, image_root)
	)

CITYSCAPES_CATEGORIES = [
{"color":[220,20,60], "isthing": 0, "id": 1, "name": "road"},
{"color":[119,11,32], "isthing": 0, "id": 2, "name": "sidewalk"},
{"color":[0,0,142], "isthing": 0, "id": 3, "name": "building"},
{"color":[0,0,230], "isthing": 0, "id": 4, "name": "wall"},
{"color":[106,0,228], "isthing": 0, "id": 5, "name": "fence"},
{"color":[0,60,100], "isthing": 0, "id": 6, "name": "pole"},
{"color":[0,80,100], "isthing": 0, "id": 7, "name": "traffic light"},
{"color":[0,0,70], "isthing": 0, "id": 8, "name": "traffic sign"},
{"color":[0,0,192], "isthing": 0, "id": 9, "name": "vegetation"},
{"color":[250,170,30], "isthing": 0, "id": 10, "name": "terrain"},
{"color":[100,170,30], "isthing": 0, "id": 11, "name": "sky"},
{"color":[220,220,0], "isthing": 1, "id": 12, "name": "person"},
{"color":[175,116,175], "isthing": 1, "id": 13, "name": "rider"},
{"color":[250,0,30], "isthing": 1, "id": 14, "name": "car"},
{"color":[165,42,42], "isthing": 1, "id": 15, "name": "truck"},
{"color":[255,77,255], "isthing": 1, "id": 16, "name": "bus"},
{"color":[0,226,252], "isthing": 1, "id": 17, "name": "train"},
{"color":[182,182,255], "isthing": 1, "id": 18, "name": "motorcycle"},
{"color":[0,82,0], "isthing": 1, "id": 19, "name": "bicycle"},
]


def _get_cityscapes_trans_meta():
    oc2nc_map = {category['id']: [cid, category["isthing"], category["name"], category["color"]] for cid, category in enumerate(CITYSCAPES_CATEGORIES)}
    # transformed info
    NEW_CITY_CATEGORIES = []
    for key, value in oc2nc_map.items():
        new_info = {"id": value[0], "isthing": value[1], "name": value[2], "color": value[3]}
        NEW_CITY_CATEGORIES.append(new_info)

    thing_ids     = [k["id"] for k in NEW_CITY_CATEGORIES]
    thing_colors  = [k["color"] for k in NEW_CITY_CATEGORIES]
    thing_classes = [k["name"] for k in NEW_CITY_CATEGORIES]
    
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}

    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }

    return ret

SPLITS_CITY = {}
SPLITS_CITY["city_train_entity"]  = ("city/images/train", "city/annotations/cityscapes_panoptic_new_train.json")
SPLITS_CITY["city_val_entity"]    = ("city/images/val", "city/annotations/cityscapes_panoptic_new_val.json")

for key, (image_root, json_file) in SPLITS_CITY.items():
	register_coco_instances(key,
		_get_cityscapes_trans_meta(),
		os.path.join(_root, json_file) if "://" not in json_file else json_file,
		os.path.join(_root, image_root)
	)
