## Data Description
Within the EntitySeg dataset downloaded from the official repo [EntitySeg-Dataset](https://github.com/adobe-research/EntitySeg-Dataset/releases/tag/v1.0), we offer both the high-resolution (original) images as well as their downsized, low-resolution counterparts. The term `lr` denotes these low-resolution versions. Comprehensive details about the dataset can be found in the files: `entityv2_010203_entity_train.json` and `entityv2_010203_entity_val.json`. Any additional annotations have been derived or processed based on these two primary annotation files. 

For the annotation formats, we strictly follow the dataloader format of detectron2 in semantic, instance and panoptic segmentation.
```
├detectron2
├── ...
├── projects
│   ├──CropFormer
│   │  ├── ...  
├── datasets
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   └── val2017
│   └── entityseg
│       ├──images
│       │   ├──entity_01_11580
│       │   ├──entity_02_11598
│       │   └──entity_03_10049
│       │──images_lr
│       │   ├──entity_01_11580
│       │   ├──entity_02_11598
│       │   └──entity_03_10049
│       ├── annotations
│       │   ├──entity_segmentation
│       │   │  ├──entityv2_01_entity_train.json
│       │   │  ├──entityv2_02_entity_train.json
│       │   │  ├──entityv2_03_entity_train.json
│       │   │  ├──entityv2_010203_entity_train.json
│       │   │  ├──entityv2_010203_entity_train_lr.json
│       │   │  ├──entityv2_010203_entity_val.json
│       │   │  └──entityv2_010203_entity_val_lr.json
│       │   ├──instance_segmentation
│       │   │  ├──entityv2_01_instances_train.json
│       │   │  └──entityv2_01_instances_val.json
│       │   ├──semantic_segmentation
│       │   │  ├──semantic_maps_train
│       │   │  ├──semantic_maps_val
│       │   │  ├──train.txt
│       │   │  └──val.txt
│       │   ├──panoptic_segmentation
│       │   │  ├──panoptic_maps_train
│       │   │  ├──panoptic_maps_val
│       │   │  ├──entityv2_01_panoptic_train.json
├───────└───└──└──entityv2_01_panoptic_val.json
```

#### Low-Resolution Entity Annotations
For our low-resolution versions, we resize the original images and annotations such that the shortest side 800 pixels and the longest side 1333 pixels. While resizing, we employ bilinear interpolation for the images and nearest-neighbor interpolation for the annotations.

We resized the images primarily because evaluating high-resolution images demands significant memory, which may not be feasible for everyone. Instead, you can evaluate the segmentation model on the low-resolution images.

#### Instance Segmentation Annotations
We selected the top 150 categories for instance segmentation based on entity frequency.

#### Semantic Segmentation Annotations
We selected the top 150 categories for semantic segmentation based on pixel frequency.

#### Panoptic Segmentation Annotations
We selected the top 350 categories for panoptic segmentation based on entity frequency.

#### Code about Category Information
```
import mmcv
mmcv.load("entityv2_01_train.json")["categories"]
```  