## Installation
* Install Detectron2 following [the instructions](https://detectron2.readthedocs.io/tutorials/install.html). We are noting that our code is implemented in detectron2 commit version e39b8d0 and pytorch 1.11.
* Copy this project to `/path/to/detectron2/projects/CropFormer` and complile it following the similar [instructions](https://github.com/facebookresearch/Mask2Former/blob/main/INSTALL.md) of Mask2Former. .
* Install the modified cocoapi for evaluating entity segmentation performance. Please install it in our `/path/to/detectron2/projects/CropFormer/entity_api`.
##### You could refer to the pseudo code for install Detectron2, CropFormer and EntityAPI as follows:
```
cd /XXX
sudo python -m pip install -e detectron2
cd detectron2
cp -r /path/to/CropFormer projects/CropFormer
cd projects/CropFormer/entity_api/PythonAPI
sudo make
cd ../../../..
cd projects/CropFormer/mask2former/modeling/pixel_decoder/ops
sudo sh make.sh
```

## Warning before training and eveluation. 
For class-agnostic entity segmentation, please launch the `projects/CropFormer/train_net_entity.py`. 

For class-aware semantic, instance and panoptic segmentation, please launch the `projects/CropFormer/train_net.py`.

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