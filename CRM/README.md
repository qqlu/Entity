# High Quality Segmentation for Ultra High-resolution Images 
<!-- <font size=6>[Project Website](http://luqi.info/Entity_Web/)</font> -->
Tiancheng Shen, Yuechen Zhang, Lu Qi, Jason Kuen, Xingyu Xie, Jianlong Wu, Zhe Lin, Jiaya Jia

This project provides an implementation for the paper "[High Quality Segmentation for Ultra High-resolution Images](https://arxiv.org/abs/2111.14482)". 

<div align="center">
  <img src="figures/teaser.png" width="100%"/>
</div><br/>


<!-- based on [Detectron2](https://github.com/facebookresearch/detectron2).  -->
To segment 4K or 6K ultra high-resolution images needs extra computation consideration in image segmentation. Common strategies, such as down-sampling, patch cropping, and cascade model, cannot address well the balance issue between accuracy and computation cost. Motivated by the fact that humans distinguish among objects continuously from coarse to precise levels, we propose the Continuous Refinement Model~(CRM) for the ultra high-resolution segmentation refinement task. CRM continuously aligns the feature map with the refinement target and aggregates features to reconstruct these image details. Besides, our CRM shows its significant generalization ability to fill the resolution gap between low-resolution training images and ultra high-resolution testing ones. We present quantitative performance evaluation and visualization to show that our proposed method is fast and effective on image segmentation refinement.

<div align="center">
  <img src="figures/results.png" width="100%"/>
</div><br/>


## Installation
This project is based on [CascadePSP](https://github.com/hkchengrex/CascadePSP).
<!-- This project has 2 code styles. The first is based on [CascadePSP](https://github.com/hkchengrex/CascadePSP), The second is based on [Detectron2](https://github.com/facebookresearch/detectron2). -->
 <!-- which can be constructed as follows. -->
<!-- * Install Detectron2 following [the instructions](https://detectron2.readthedocs.io/tutorials/install.html). We are noting that our code is implemented in detectron2 commit version 28174e932c534f841195f02184dc67b941c65a67 and pytorch 1.8.
* Setup the coco dataset including instance and panoptic annotations following [the structure](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md). The code of entity evaluation metric is saved in the file of modified_cocoapi. You can directly replace your compiled coco.py with modified_cocoapi/PythonAPI/pycocotools/coco.py. 
* Copy this project to `/path/to/detectron2/projects/EntitySeg`
* Set the "find_unused_parameters=True" in distributed training of your own detectron2. You could modify it in detectron2/engine/defaults.py. -->

<!-- ## Data pre-processing
(1) Generate the entity information of each image by the instance and panoptic annotation. Please change the path of coco annotation files in the following code.
```bash
cd /path/to/detectron2/projects/EntitySeg/make_data
bash make_entity_mask.sh
```
(2) Change the generated entity information to the json files.
```bash
cd /path/to/detectron2/projects/EntitySeg/make_data
python3 entity_to_json.py
``` -->
## Dependencies

CRM can be trained and tested on PyTorch 1.7.1 or higher version. Other dependencies are needed to be installed by:
```
pip install progressbar2
pip install opencv-python
```

## Download the dataset 

Use the script in  CRM/scripts/ to download the training dataset. The training dataset merges the following datasets: MSRA-10K, DUT-OMRON, ECSSD, and FSS-1000.
 
```
cd ./scripts/
python download_training_dataset.py
```

For the evaluation dataset BIG. Please download it follow the CascadePSP's instruction.

## Training
To train model with 2 GPUs, run:
```bash
cd CRM/
python train.py Exp_ID -i 45000 -b 12 --steps 22500 37500 --lr 2.25e-4 --ce_weight 1.0 --l1_weight 0.5 --l2_weight 0.5 --grad_weight 2.0
```

## Evaluation and Visualization
To evaluate a pre-trained model on BIG dataset, run:
```
python test.py \
    --dir /PathTO/BIG_PSPNet_SS \
    --model /PathTO/weights/Exp_ID/model_45705 \
    --output /PathTO/output/Exp_ID \
    --clear

python eval_post0.125.py --dir /PathTO/output/Exp_ID

python eval_post0.25.py --dir /PathTO/output/Exp_ID

python eval_post0.5.py --dir /PathTO/output/Exp_ID

python eval_post1.0.py --dir /PathTO/output/Exp_ID

```

<!-- ## Visualization
To visualize some image result of a pre-trained model, run:
```bash
cd /path/to/detectron2
python3 projects/EntitySeg/demo_result_and_vis.py --config-file <config.yaml> --input <input_path> --output <output_path> MODEL.WEIGHTS model_checkpoint MODEL.CONDINST.MASK_BRANCH.USE_MASK_RESCORE "True"
```
For example,
```bash
python3 projects/EntitySeg/demo_result_and_vis.py --config-file projects/EntitySeg/configs/entity_swin_lw7_1x.yaml --input /data/input/*.jpg --output /data/output MODEL.WEIGHTS /data/pretrained_model/R_50.pth MODEL.CONDINST.MASK_BRANCH.USE_MASK_RESCORE "True"
``` -->
## Pretrained weights of CRM

Checkpoint  | Downloads | File Size | 
---------------------- | -----------------|  --------- |
This is the model that we used to generate all of our results in the paper. | [OneDrive](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155154510_link_cuhk_edu_hk/EWdbUAf33ZlNvB2d1dkBv-QBwSnRA0ong9wFqig54I5Iyw?e=pcjp8y) | 93MB |
<!-- This is the newly trained model with restructured code and updated hyperparameters in this repo. It has slightly better performance. | [Google Drive](https://drive.google.com/open?id=1FMmUYtWsZB4fReoQmtqqn-NOZrC8CfWK) <br> [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jchungaa_connect_ust_hk/EW7CBmiBK9RJlmORaEpXRg4B4gZ0GtU3L6K64oFdD-GKWw?e=q0Tg5p) | 259MB | -->

## Segmentation Results

For convenience, we provide segmentation results from other models for evaluation and Visualization. 

| Dataset | Coarse Mask Source | Output Link |
|--------------|:-------------------------:|:-------------------------------:|
| BIG (Test)   | DeeplabV3+  | [Link](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155154510_link_cuhk_edu_hk/Ef_hUoB9l3dFvm6oonaT478BCOrNhEzP7uggqxTFbQROBA?e=K5WduB) |
|              | RefineNet   | [Link](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155154510_link_cuhk_edu_hk/EfpxzKaGxh5IiJbln-Q2JkcB2IHJLxBl3SgHwu0lFmttNA?e=CTBECY) |
|              | PSPNet      | [Link](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155154510_link_cuhk_edu_hk/EbTj8f0BybtEq6Xa_Xh_86cBpwJh9S29GO_PfdoUDAoU5A?e=yfrDvp) |
|              | FCN-8s      | [Link](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155154510_link_cuhk_edu_hk/Ecw5Oj8JcsROt5HMgL03-BwBTM0I8ysTFIGZ-XzoVKZ1Uw?e=n8yJeR) |

## <a name="Citing Ours"></a>Citing Ours

Consider to cite **High Quality Segmentation for Ultra High-resolution Images** if it helps your research.

```
@article{shen2021high,
  title={High Quality Segmentation for Ultra High-resolution Images},
  author={Tiancheng Shen, Yuechen Zhang, Lu Qi, Jason Kuen, Xingyu Xie, Jianlong Wu, Zhe Lin, Jiaya Jia},
  journal={CVPR},
  year={2022}
}
```
