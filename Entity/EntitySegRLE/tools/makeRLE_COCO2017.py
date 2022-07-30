import os
import copy
import mmcv
import numpy as np
import pdb
import pycocotools.mask as mask_utils
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

prefix = "train2017"
base_path = "/data/ceph/gavinqi/data/coco"

entity_base_path              = os.path.join(base_path, "entity_{}".format(prefix))
annotation_path               = os.path.join(base_path, "annotations/instances_{}.json".format(prefix))
save_entity_path              = os.path.join(base_path, "annotations/entity_{}_RLE.json".format(prefix))

## build catid to continous
categories_list               = COCO_CATEGORIES
# pdb.set_trace()
catid_map                     = {category['id']: [cid, category["isthing"], category["name"], category["name"]] for cid, category in enumerate(categories_list)}
idcat_map = {}
for key, value in catid_map.items():
    idcat_map[value[0]] = [key,value[1]]

instance_annotations       = mmcv.load(annotation_path)
entity_annotations         = copy.deepcopy(instance_annotations)
# instance_annotations_thing = copy.deepcopy(instance_annotations)
# instance_annotations_stuff = copy.deepcopy(instance_annotations)

# update category
print("Updating categories...")
entity_annotations["categories"] = []
# instance_annotations_thing["categories"] = []
# instance_annotations_stuff["categories"] = []
for origin_catid, new_catid_info in catid_map.items():
    new_catid = int(new_catid_info[0])
    is_thing  = int(new_catid_info[1])
    name      = new_catid_info[2]
    nsuper    = new_catid_info[3]
    entity_annotations["categories"].append({"supercategory":nsuper, "id": new_catid, "name":name, "is_thing": is_thing})
    # new_dict  = dict()
    # new_dict["supercategory"]=nsuper
    # new_dict["id"] = new_catid
    # new_dict["name"] = name
    # new_dict["is_thing"] = is_thing
    # entity_annotations["categories"] = new_dict
    
    # entity_annotations["categories"].append({"supercategory": nsuper, "id": new_catid, "name": name, "is_thing": is_thing})
print("Update category finished")
# pdb.set_trace()

entity_annotations["annotations"] = []
# instance_annotations_thing["annotations"] = []
# instance_annotations_stuff["annotations"] = []
npz_names = os.listdir(entity_base_path)
entities_id = 0
# thing_id  = 0
# stuff_id  = 0

for index, npz_name in enumerate(npz_names):
    entity_info = np.load(os.path.join(entity_base_path, npz_name))
    image_id = int(npz_name.split(".")[0])
    bounding_boxes = entity_info["bounding_box"]
    entity_id_map = entity_info["map"]
    entity_id_map = entity_id_map[0]
    if len(bounding_boxes)==0:
        continue
    # 0-x1, 1-y1, 2-x2, 3-y2, 4-category, 5-thing_or_stuff, 6-entity_id
    # thing_mask  = bounding_boxes[:,5] > 0
    # stuff_mask  = bounding_boxes[:,5] == 0

    for box in bounding_boxes:
        x1, y1, x2, y2, category_id, thing_or_stuff, entity_id = box
        area = (y2-y1) * (x2-x1)
        
        mask = (entity_id_map==entity_id)
        mask = np.array(mask, order="F", dtype="uint8")
        rle  = mask_utils.encode(mask)
        rle["counts"] = rle["counts"].decode("utf-8")

        anno = {"iscrowd": 0, 
                "area": area, 
                "image_id": image_id, 
                "bbox": [x1, y1, x2-x1, y2-y1], 
                "category_id": category_id, 
                "id": entities_id,
                "segmentation": rle}
        
        entity_annotations["annotations"].append(anno)
        entities_id = entities_id + 1
    
    print("{},{}".format(index, npz_name))

mmcv.dump(entity_annotations, save_entity_path)
# mmcv.dump(instance_annotations_stuff, save_stuff_path)

# thing_info = instance_annotations_thing
# stuff_info = instance_annotations_stuff

# thst       = thing_info
# thst["categories"].extend(stuff_info["categories"])
# nums = len(thst["annotations"]) + 1
# for index, anno in enumerate(stuff_info["annotations"]):
# 	anno["id"] = index + nums
# 	thst["annotations"].append(anno)
# mmcv.dump(thst, save_entity_path)