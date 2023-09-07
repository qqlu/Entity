# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
import mmcv

ENTITYV2_SEMSEG150_CATEGORIES = [
{'name': 'sky', 'id': 0},
{'name': 'concretewall', 'id': 1},
{'name': 'tree', 'id': 2},
{'name': 'grass', 'id': 3},
{'name': 'house', 'id': 4},
{'name': 'mountain', 'id': 5},
{'name': 'sea', 'id': 6},
{'name': 'floor', 'id': 7},
{'name': 'shrub', 'id': 8},
{'name': 'floor_structure_ot', 'id': 9},
{'name': 'woman', 'id': 10},
{'name': 'building_houses_ot', 'id': 11},
{'name': 'lake', 'id': 12},
{'name': 'high-rise', 'id': 13},
{'name': 'ceiling', 'id': 14},
{'name': 'non-buildinghouse_ot', 'id': 15},
{'name': 'man', 'id': 16},
{'name': 'noncommon_furniture', 'id': 17},
{'name': 'table_ot', 'id': 18},
{'name': 'snow', 'id': 19},
{'name': 'road', 'id': 20},
{'name': 'rock', 'id': 21},
{'name': 'river', 'id': 22},
{'name': 'window', 'id': 23},
{'name': 'sand', 'id': 24},
{'name': 'sunk_fence', 'id': 25},
{'name': 'building_ot', 'id': 26},
{'name': 'park_ground', 'id': 27},
{'name': 'soil', 'id': 28},
{'name': 'wall_ot', 'id': 29},
{'name': 'chair', 'id': 30},
{'name': 'curtain', 'id': 31},
{'name': 'buildingstructure_ot', 'id': 32},
{'name': 'rug', 'id': 33},
{'name': 'fluid_ot', 'id': 34},
{'name': 'potted_plants', 'id': 35},
{'name': 'painting', 'id': 36},
{'name': 'porch', 'id': 37},
{'name': 'person_ot', 'id': 38},
{'name': 'bedroom bed', 'id': 39},
{'name': 'court', 'id': 40},
{'name': 'car', 'id': 41},
{'name': 'crops', 'id': 42},
{'name': 'skyscraper', 'id': 43},
{'name': 'leaf', 'id': 44},
{'name': 'cabinets_ot', 'id': 45},
{'name': 'swimmingpool', 'id': 46},
{'name': 'sculpture', 'id': 47},
{'name': 'bridge', 'id': 48},
{'name': 'sidewalk', 'id': 49},
{'name': 'stone', 'id': 50},
{'name': 'book', 'id': 51},
{'name': 'castle', 'id': 52},
{'name': 'kitchen cabinets', 'id': 53},
{'name': 'entertainment_appliances_ot', 'id': 54},
{'name': 'mat', 'id': 55},
{'name': 'utility_ot', 'id': 56},
{'name': 'gravel', 'id': 57},
{'name': 'flower', 'id': 58},
{'name': 'cushion', 'id': 59},
{'name': 'pond', 'id': 60},
{'name': 'facility_ot', 'id': 61},
{'name': 'glasswall', 'id': 62},
{'name': 'nonindividual_plants_ot', 'id': 63},
{'name': 'land_transportation_ot', 'id': 64},
{'name': 'artifical_ground_ot', 'id': 65},
{'name': 'step', 'id': 66},
{'name': 'toy_ot', 'id': 67},
{'name': 'couch', 'id': 68},
{'name': 'box', 'id': 69},
{'name': 'mirror', 'id': 70},
{'name': 'dog', 'id': 71},
{'name': 'rigid_container_ot', 'id': 72},
{'name': 'refrigerator', 'id': 73},
{'name': 'wiredfence', 'id': 74},
{'name': 'boy', 'id': 75},
{'name': 'fence', 'id': 76},
{'name': 'waterfall', 'id': 77},
{'name': 'girl', 'id': 78},
{'name': 'ordniary_sofa', 'id': 79},
{'name': 'signboard', 'id': 80},
{'name': 'double_sofa', 'id': 81},
{'name': 'paper', 'id': 82},
{'name': 'dirt', 'id': 83},
{'name': 'cutting_board', 'id': 84},
{'name': ' medical_equipment', 'id': 85},
{'name': 'laptop', 'id': 86},
{'name': 'horse_ot', 'id': 87},
{'name': 'kiosk', 'id': 88},
{'name': 'boat', 'id': 89},
{'name': 'lab_tool_ot', 'id': 90},
{'name': 'train', 'id': 91},
{'name': 'trunk', 'id': 92},
{'name': 'airplane', 'id': 93},
{'name': 'television', 'id': 94},
{'name': 'wardrobe', 'id': 95},
{'name': 'dessert_snacks_ot', 'id': 96},
{'name': 'blanket', 'id': 97},
{'name': 'birds_ot', 'id': 98},
{'name': 'bottle', 'id': 99},
{'name': 'cat', 'id': 100},
{'name': 'drink_ot', 'id': 101},
{'name': 'tower', 'id': 102},
{'name': 'plate', 'id': 103},
{'name': 'outdoor_supplies_ot', 'id': 104},
{'name': 'pillow', 'id': 105},
{'name': 'temple', 'id': 106},
{'name': 'decoration', 'id': 107},
{'name': 'solid_ot', 'id': 108},
{'name': 'bathroom cabinets', 'id': 109},
{'name': 'parkinglot', 'id': 110},
{'name': 'blanket', 'id': 111},
{'name': 'stool', 'id': 112},
{'name': 'doll_ot', 'id': 113},
{'name': 'dining_table', 'id': 114},
{'name': 'pier', 'id': 115},
{'name': 'bathhub', 'id': 116},
{'name': 'playground', 'id': 117},
{'name': 'vine', 'id': 118},
{'name': 'tent', 'id': 119},
{'name': 'billiard_table', 'id': 120},
{'name': 'washing_machine', 'id': 121},
{'name': 'alley', 'id': 122},
{'name': 'barrier_ot', 'id': 123},
{'name': 'banister', 'id': 124},
{'name': 'bath_tool_ot', 'id': 125},
{'name': 'mammal_ot', 'id': 126},
{'name': 'stair', 'id': 127},
{'name': 'branch', 'id': 128},
{'name': 'chandilier', 'id': 129},
{'name': 'meat_ot', 'id': 130},
{'name': 'blackboard', 'id': 131},
{'name': 'table_cloth', 'id': 132},
{'name': 'backpack', 'id': 133},
{'name': 'ceiling_lamp', 'id': 134},
{'name': 'bench', 'id': 135},
{'name': 'screen', 'id': 136},
{'name': 'billboard', 'id': 137},
{'name': 'towel', 'id': 138},
{'name': 'cow_ot', 'id': 139},
{'name': 'fitness_equipment_ot', 'id': 140},
{'name': 'sofa_ot', 'id': 141},
{'name': 'bus', 'id': 142},
{'name': 'path', 'id': 143},
{'name': 'musical_instrument', 'id': 144},
{'name': 'cup_ot', 'id': 145},
{'name': 'mobilephone', 'id': 146},
{'name': 'photo', 'id': 147},
{'name': 'pants', 'id': 148},
{'name': 'pumkin', 'id': 149},
]

def load_sem_seg_w_txt(txt_path, image_root, gt_root):
    """
    """
    infos = mmcv.list_from_file(txt_path)
    input_files = []
    gt_files = []
    for info in infos:
        img_name, gt_name = info.strip().split(" ")
        input_files.append(os.path.join(image_root, img_name))
        gt_files.append(os.path.join(gt_root, gt_name))

    dataset_dicts = []
    for (img_path, gt_path) in zip(input_files, gt_files):
        record = {}
        record["file_name"] = img_path
        record["sem_seg_file_name"] = gt_path
        dataset_dicts.append(record)

    return dataset_dicts

def _get_entityv2_semseg_150_meta():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing.
    semseg_ids = [k["id"] for k in ENTITYV2_SEMSEG150_CATEGORIES]
    assert len(semseg_ids) == 150, len(semseg_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(semseg_ids)}
    stuff_classes = [k["name"] for k in ENTITYV2_SEMSEG150_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret


def register_entityv2_sem150(root):
    meta = _get_entityv2_semseg_150_meta()
    for name, txt_name, image_dirname, sem_seg_dirname in [
        ("train", "entityseg/annotations/semantic_segmentation/train.txt", "entityseg/images/entity_01_11580", "entityseg/annotations/semantic_segmentation/semantic_maps_train"),
        ("test", "entityseg/annotations/semantic_segmentation/val.txt", "entityseg/images/entity_01_11580", "entityseg/annotations/semantic_segmentation/semantic_maps_val"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        txt_path = os.path.join(root, txt_name)

        meta = _get_entityv2_semseg_150_meta()

        name = f"entityv2_sem150_{name}"
        DatasetCatalog.register(
            name, lambda t=txt_path, x=image_dir, y=gt_dir: load_sem_seg_w_txt(t,x,y)
        )
        MetadataCatalog.get(name).set(
            txt_path=txt_path,
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_entityv2_sem150(_root)
