# High-Quality Entity Segmentation <font size=6>[Project Website](http://luqi.info/entityv2.github.io/)</font>
Lu Qi, Jason Kuen, Tiancheng Shen, Jiuxiang Gu, Weidong Guo, Jiaya Jia, Zhe Lin, Ming-Hsuan Yang

This project provides an implementation for the paper "[High-Quality Entity Segmentation](https://arxiv.org/abs/2211.05776)" based on [Detectron2](https://github.com/facebookresearch/detectron2). Entity Segmentation is a segmentation task with the aim to segment everything in an image into semantically-meaningful regions without considering any category labels.

In this repository, we provide the link of EntitySeg Dataset and the code of our proposed method CropFormer for fine-grained entity segmentation.

<div align="center">
  <img src="figures/teaser_mosaic_low.png" width="90%"/>
</div><br/>

## News
* 2023.07.01: The paper is accepted as ICCV2023 oral.
* 2022.01.12: Initialize the Github.

## Installation
This project is based on [Detectron2](https://github.com/facebookresearch/detectron2) and [Mask2Former](https://github.com/facebookresearch/Mask2Former), which can be constructed as follows.
* Install Detectron2 following [the instructions](https://detectron2.readthedocs.io/tutorials/install.html). We are noting that our code is implemented in detectron2 commit version e39b8d0 and pytorch 1.11.
* Copy this project to `/path/to/detectron2/projects/CropFormer` and complile it following the [the instructions](https://github.com/facebookresearch/Mask2Former/blob/main/INSTALL.md).
* Install the modified cocoapi for evaluating entity segmentation performance. Please install it in our `/path/to/detectron2/projects/CropFormer/entity_api`.
##### You could refer to the bash code for install Detectron2, Mask2Former and COCOAPI as follows:
```
cd /XXX/
sudo python -m pip install -e detectron2
cd detectron2
cp -r /path/to/CropFormer projects/CropFormer
cd projects/CropFormer/entity_api/PythonAPI"
sudo make
cd ../../../..
cd projects/CropFormer/mask2former/modeling/pixel_decoder/ops
sudo sh make.sh
```

#### Pretrained weights of [Swin Transformers](https://github.com/microsoft/Swin-Transformer)

Use the tools/convert-pretrained-swin-model-to-d2.py to convert the pretrained weights of Swin Transformers to the detectron2 format.

#### Code and Data Path
The google drive link to download EntitySeg Dataset.
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
│       ├── annotations
│       │   ├──entity_segmentation
│       │   │  ├──entityv2_01_entity_train.json
│       │   │  ├──entityv2_01_entity_val.json
│       │   │  ├──entityv2_02_entity_train.json
│       │   │  ├──entityv2_02_entity_val.json
│       │   │  ├──entityv2_03_entity_train.json
│       │   │  ├──entityv2_03_entity_val.json
│       │   │  ├──entityv2_010203_entity_train.json
│       │   │  └──entityv2_010203_entity_val.json
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
├── ...
```


## Training
To train model with 8 GPUs, run:
```bash
cd /path/to/detectron2
python3 projects/CropFormer/train_net_entity.py --config-file <projects/EntitySeg/configs/config.yaml> --num-gpus 8
```

For example, to launch entity segmentation training (1x schedule) with ResNet-50 backbone on 8 GPUs and save the model in the path "/data/entity_model".
one should execute:
```bash
cd /path/to/detectron2
python3 projects/CropFormer/train_net_entity.py --config-file projects/CropFormer/configs/entityv2/cropformer_swin_tiny_1x.yaml --num-gpus 8 OUTPUT_DIR /data/entity_model
```

## Evaluation
To evaluate a pre-trained model with 8 GPUs, run:
```bash
cd /path/to/detectron2
python3 projects/CropFormer/train_net_entity.py --config-file <config.yaml> --num-gpus 8 --eval-only MODEL.WEIGHTS model_checkpoint
```

## Visualization
To visualize some image result of a pre-trained model, run:
```bash
cd /path/to/detectron2
python3 projects/CropFormer/demo_from_dirs.py --config-file <config.yaml> --input <input_path> --output <output_path> MODEL.WEIGHTS model_checkpoint MODEL.CONDINST.MASK_BRANCH.USE_MASK_RESCORE "True"
```
For example,
```bash
cd /path/to/detectron2
cp -r /path/to/CropFormer projects/CropFormer
python3 projects/CropFormer/demo_cropformer/demo_from_dirs.py --config-file projects/CropFormer/configs/coco_person/cropformer_swin_large_3x_noise_000_100_200.yaml --input /group/20018/gavinqi/data/ft_local/100m_crop_sample/*.jpg --output /group/20027/gavinqi/100m_vis/ --opts MODEL.WEIGHTS /group/20027/gavinqi/model/coco_person_noise_000_100_200/model_final.pth
```

## Visualization of Person with Atmospheric Noise
```bash
python3 projects/Mask2Former/demo_cropformer/demo_from_dirs.py --config-file projects/Mask2Former/configs/coco_person/cropformer_swin_large_3x_noise_000_100_200.yaml --input /group/20018/gavinqi/data/ft_local/100m_crop_sample/*.jpg --output /group/20027/gavinqi/100m_vis/ --opts MODEL.WEIGHTS /group/20027/gavinqi/model/coco_person_noise_000_100_200/model_final.pth
```

## Model Zoo
We provide the results of several pretrained models on EntitySeg *val* set. For all the training, we use the COCO-Pretrain models. The AP&e

### Entity Segmentation
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="center">Method</th>
<th valign="center">Backbone</th>
<th valign="center">Sched</th>
<th valign="center">AP^e</th>
<th valign="center">download</th>
<th valign="center">config file</th>

<tr><td align="center">Mask2Former</td>
<td align="center">Swin-T</td>
<td align="center">3x</td>
<td align="center">  </td>
<td align="center"> <a href="">model</a> </td>
<td align="left"> configs/entityv2/entity_segmentation/mask2former_swin_tiny_3x.yaml </td>

<tr><td align="center">CropFormer</td>
<td align="center">Swin-T</td>
<td align="center">3x</td>
<td align="center"> 42.8 </td>
<td align="center"> <a href="">model</a> </td>
<td align="left"> configs/entityv2/entity_segmentation/cropformer_swin_tiny_3x.yaml </td>

<tr><td align="center">Mask2Former</td>
<td align="center"> Swin-L </td>
<td align="center"> 3x </td>
<td align="center"> 46.2 </td>
<td align="center"> <a href="">model</a> </td>
<td align="left"> configs/entityv2/entity_segmentation/mask2former_swin_large_3x.yaml </td>

<tr><td align="center">CropFormer</td>
<td align="center">Swin-L</td>
<td align="center">3x</td>
<td align="center"> 48.2 </td>
<td align="center"> <a href="">model</a></td>
<td align="left"> configs/entityv2/entity_segmentation/cropformer_swin_large_3x.yaml </td>

<tr><td align="center">Mask2Former</td>
<td align="center"> Hornet-L </td>
<td align="center"> 3x </td>
<td align="center"> 49.2 </td>
<td align="center"> <a href="">model</a> </td>
<td align="left"> configs/entityv2/entity_segmentation/mask2former_swin_large_3x.yaml </td>
</tbody></table>

### Instance Segmentation

### Semantic Segmentation

### Panoptic Segmentation

## <a name="Citing Ours"></a>Citing Ours

Consider to cite **Fine-Grained Entity Segmentation** if it helps your research.

```
@article{qi2022fine,
  title={Fine-Grained Entity Segmentation},
  author={Qi, Lu and Kuen, Jason and Guo, Weidong and Shen, Tiancheng and Gu, Jiuxiang and Li, Wenbo and Jia, Jiaya and Lin, Zhe and Yang, Ming-Hsuan},
  journal={arXiv preprint arXiv:2211.05776},
  year={2022}
}
```