# Open-World Entity Segmentation <font size=6>[Project Website](http://luqi.info/Entity_Web/)</font>
Lu Qi\*, Jason Kuen\*, Yi Wang, Jiuxiang Gu, Hengshuang Zhao, Zhe Lin, Philip Torr, Jiaya Jia

<div align="center">
  <img src="figures/motivation.png" width="80%"/>
</div><br/>

This project provides an implementation for the paper "[Open-World Entity Segmentation](https://arxiv.org/abs/2107.14228)" based on [Detectron2](https://github.com/facebookresearch/detectron2). Entity Segmentation is a segmentation task with the aim to segment everything in an image into semantically-meaningful regions without considering any category labels. Our entity segmentation models can perform exceptionally well in a cross-dataset setting where we use only COCO as the training dataset but we test the model on images from other datasets at inference time. Please refer to project website for more details and visualizations.

<div align="center">
  <img src="figures/Generalization_imagenet.png" width="600"/>
</div><br/>


## Installation
This project is based on [Detectron2](https://github.com/facebookresearch/detectron2), which can be constructed as follows.
* Install Detectron2 following [the instructions](https://detectron2.readthedocs.io/tutorials/install.html). We are noting that our code is implemented in detectron2 commit version 28174e932c534f841195f02184dc67b941c65a67 and pytorch 1.8.
* Setup the coco dataset including instance and panoptic annotations following [the structure](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md). The code of entity evaluation metric is saved in the file of modified_cocoapi. You can directly replace your compiled coco.py with modified_cocoapi/PythonAPI/pycocotools/coco.py. 
* Copy this project to `/path/to/detectron2/projects/EntitySeg`
* Set the "find_unused_parameters=True" in distributed training of your own detectron2. You could modify it in detectron2/engine/defaults.py.

## Data pre-processing
(1) Generate the entity information of each image by the instance and panoptic annotation. Please change the path of coco annotation files in the following code.
```bash
cd /path/to/detectron2/projects/EntitySeg/make_data
bash make_entity_mask.sh
```
(2) Change the generated entity information to the json files.
```bash
cd /path/to/detectron2/projects/EntitySeg/make_data
python3 entity_to_json.py
```


## Training
To train model with 8 GPUs, run:
```bash
cd /path/to/detectron2
python3 projects/EntitySeg/train_net.py --config-file <projects/EntitySeg/configs/config.yaml> --num-gpus 8
```

For example, to launch entity segmentation training (1x schedule) with ResNet-50 backbone on 8 GPUs and save the model in the path "/data/entity_model".
one should execute:
```bash
cd /path/to/detectron2
python3 projects/EntitySeg/train_net.py --config-file projects/EntitySeg/configs/entity_default.yaml --num-gpus 8 OUTPUT_DIR /data/entity_model
```

## Evaluation
To evaluate a pre-trained model with 8 GPUs, run:
```bash
cd /path/to/detectron2
python3 projects/EntitySeg/train_net.py --config-file <config.yaml> --num-gpus 8 --eval-only MODEL.WEIGHTS model_checkpoint
```

## Visualization
To visualize some image result of a pre-trained model, run:
```bash
cd /path/to/detectron2
python3 projects/EntitySeg/demo_result_and_vis.py --config-file <config.yaml> --input <input_path> --output <output_path> MODEL.WEIGHTS model_checkpoint MODEL.CONDINST.MASK_BRANCH.USE_MASK_RESCORE "True"
```
For example,
```bash
python3 projects/EntitySeg/demo_result_and_vis.py --config-file projects/EntitySeg/configs/entity_swin_lw7_1x.yaml --input /data/input/*.jpg --output /data/output MODEL.WEIGHTS /data/pretrained_model/R_50.pth MODEL.CONDINST.MASK_BRANCH.USE_MASK_RESCORE "True"
```
## Pretrained weights of [Swin Transformers](https://github.com/microsoft/Swin-Transformer)

Use the tools/convert_swin_to_d2.py to convert the pretrained weights of Swin Transformers to the detectron2 format. For example,
```bash
pip install timm
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
python tools/convert_swin_to_d2.py swin_tiny_patch4_window7_224.pth swin_tiny_patch4_window7_224_trans.pth
```

## Pretrained weights of [Segformer Backbone](https://github.com/NVlabs/SegFormer)

Use the tools/convert_mit_to_d2.py to convert the pretrained weights of [SegFormer Backbone](https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia) to the detectron2 format. For example,
```bash
pip install timm
python tools/convert_mit_to_d2.py mit_b0.pth mit_b0_trans.pth
```

## Results
We provide the results of several pretrained models on COCO *val* set. It is easy to extend it to other backbones. We first describe the results of using CNN backbone.

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="center">Method</th>
<th valign="center">Backbone</th>
<th valign="center">Sched</th>
<th valign="center">Entity AP</th>
<th valign="bottom">download</th>
<tr><td align="center">Baseline</td>
<td align="center">R50</td>
<td align="center">1x</td>
<td align="center"> 28.3 </td>
<td align="center"> <a href="https://drive.google.com/file/d/17MsgUfjVSOs4_R8FO6mzMwtg0vH4HC57/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/1UQ50Fj8e-5-LHiFEfKuOgz5SrQ7ocahD/view?usp=sharing">metrics</a> </td>
<!-- <td align="center"> To be released </td> -->

<tr><td align="center">Ours</td>
<td align="center">R50</td>
<td align="center">1x</td>
<td align="center"> 29.8 </td>
<td align="center"> <a href="https://drive.google.com/file/d/1_p_gP5_NTTqVlSXJFqdh3h8rW2KwoV5Q/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/1E1jKu29u9dwLBRA7GFDmquQUhz8ZNU8A/view?usp=sharing">metrics</a> </td>

<tr><td align="center">Ours</td>
<td align="center">R50</td>
<td align="center">3x</td>
<td align="center"> 31.8 </td>
<td align="center"> <a href="https://drive.google.com/file/d/1AygMH7vq3ufBwalqgycuKvagjWcWue70/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/1FStVA04AUk-cs2kC07vVfG1YSyZEHlxG/view?usp=sharing">metrics</a> </td>
<!-- <td align="center"> To be released </td> -->
</tr>
<tr><td align="center">Ours</td>
<td align="center">R101</td>
<td align="center">1x</td>
<td align="center"> 31.0 </td>
<td align="center"> <a href="https://drive.google.com/file/d/13oxyTQvYKKim1SEdlS-a9ME-yVTaQhmG/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/17nuCXu9cqoJfqOsW-xFkTeDXbYYSNIzA/view?usp=sharing">metrics</a> </td>

<tr><td align="center">Ours</td>
<td align="center">R101</td>
<td align="center">3x</td>
<td align="center">33.2</td>
<td align="center"> <a href="https://drive.google.com/file/d/1a58lNf8n6aJYY0_Lq-R002AHJpWqnEtv/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/1aFYBQgK7ji6KMOfffFlyWA7gL_1bCgde/view?usp=sharing">metrics</a> </td>

<tr><td align="center">Ours</td>
<td align="center">R101-DCNv2</td>
<td align="center">3x</td>
<td align="center"> 35.5 </td>
<td align="center"> <a href="https://drive.google.com/file/d/1bpjZk8svC-WPvsexInXfwgIdj7rLg2gM/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/1PcYLxtqHTvEsU7bx4T9Hxx-HcJ_72pnF/view?usp=sharing">metrics</a> </td>
</tbody></table>

The results of using transformer backbone as follows.The *Mask Rescore* indicates that we use mask rescoring in inference by setting `MODEL.CONDINST.MASK_BRANCH.USE_MASK_RESCORE` to `True`.

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="center">Method</th>
<th valign="center">Backbone</th>
<th valign="center">Sched</th>
<th valign="center">Entity AP</th>
<th valign="center">Mask Rescore</th>
<th valign="bottom">download</th>

<tr><td align="center">Ours</td>
<td align="center">Swin-T</td>
<td align="center">1x</td>
<td align="center"> 33.0 </td>
<td align="center"> 34.6 </td>
<td align="center"> <a href="https://drive.google.com/file/d/1uMxGjCx7pA_GocdVA-3rmvcAZQw2nvIC/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/1zSqPrm9qs8pP02_bClpEnCj_dfERzkVW/view?usp=sharing">metrics</a> </td>

<tr><td align="center">Ours</td>
<td align="center">Swin-L-W7</td>
<td align="center">1x</td>
<td align="center"> 37.8 </td>
<td align="center"> 39.3 </td>
<td align="center"> <a href="https://drive.google.com/file/d/1uAJgkFsBr_f3wGzNby_mKZA2JkkQARHh/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/1nThaanHv_O21LQGaEGQuiPQab_k5NyTS/view?usp=sharing">metrics</a> </td>

<tr><td align="center">Ours</td>
<td align="center">Swin-L-W7</td>
<td align="center">3x</td>
<td align="center"> 38.6 </td>
<td align="center"> 40.0 </td>
<td align="center"> <a href="https://drive.google.com/file/d/1xPWkf0WiF14h7wM7nuapOAEIqZsStGdm/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/19ffW1Oz-Cyf46y8k0Tz8rILLANjLqZES/view?usp=sharing">metrics</a> </td>

<tr><td align="center">Ours</td>
<td align="center">Swin-L-W12</td>
<td align="center">3x</td>
<td align="center"> 38.7 </td>
<td align="center"> 40.1 </td>
<td align="center"> <a href="https://drive.google.com/file/d/1Z7o1w3NM1MsXsJJyLX9DyCAw88yrzHDz/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/1TUwUWKlFH4QnnmBhnheGDconz5MDQqav/view?usp=sharing">metrics</a> </td>

<tr><td align="center">Ours</td>
<td align="center">MiT-b0</td>
<td align="center">1x</td>
<td align="center"> 28.8 </td>
<td align="center"> 30.4 </td>
<td align="center"> <a href="https://drive.google.com/file/d/1VxQAYsvJeNASHdfE8_XSOkQ9Kwwok4Ah/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/1IUMyBbLlk5xxqVZn2NeK0FRu4cTMeDTd/view?usp=sharing">metrics</a> </td>

<tr><td align="center">Ours</td>
<td align="center">MiT-b2</td>
<td align="center">1x</td>
<td align="center"> 35.1 </td>
<td align="center"> 36.6 </td>
<td align="center"> <a href="https://drive.google.com/file/d/1alwiIhouGSA-3W9PlN90ArwCDlg-t_ih/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/18IxTXBaW6k_oL6icCHgyfD4C-1F9TSHE/view?usp=sharing">metrics</a> </td>

<tr><td align="center">Ours</td>
<td align="center">MiT-b3</td>
<td align="center">1x</td>
<td align="center"> 36.9 </td>
<td align="center"> 38.5 </td>
<td align="center"> <a href="https://drive.google.com/file/d/17PdEcCIOt-Uzjp2xYL1c3ejzqmHWA8e5/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/1OKzOjTyzr3ce_SJjt1IrnZZ0oh8ebPOy/view?usp=sharing">metrics</a> </td>

<tr><td align="center">Ours</td>
<td align="center">MiT-b5</td>
<td align="center">1x</td>
<td align="center"> 37.2 </td>
<td align="center"> 38.7 </td>
<td align="center"> <a href="https://drive.google.com/file/d/1KKjcu8A7p7fvGBPaK7P3DAit5N0B5tD-/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/1y7AmbxGI6O8AGh2AG2ROUy54YovQk1TX/view?usp=sharing">metrics</a> </td>

<tr><td align="center">Ours</td>
<td align="center">MiT-b5</td>
<td align="center">3x</td>
<td align="center"> 37.4 </td>
<td align="center"> 38.7 </td>
<td align="center"> <a href="https://drive.google.com/file/d/1gLSXDFbLSqo3HjG5nzr5DQf2Rb1LIIVx/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/1rBheseWajXPoVWR5tTACLsxYdqfKHYTG/view?usp=sharing">metrics</a> </td>

</tbody></table>

## Cross Dataset Results for R-50 (12 Epoch)
TBD
<table><tbody>
<th valign="center">Dataset</th>
<th valign="center">TEST-COCO</th>
<th valign="center">TEST-ADE20K</th>
<th valign="center">TEST-CITY</th>
<th valign="center">download</th>

<tr><td align="center">TRAIN-COCO</td>
<td align="center">TBD</td>
<td align="center">TBD</td>
<td align="center">TBD </td>
<td align="center"> <a href="">model</a>&nbsp;|&nbsp;<a href="">metrics</a> </td>

<tr><td align="center">TRAIN-ADE20K</td>
<td align="center">TBD</td>
<td align="center">TBD</td>
<td align="center">TBD </td>
<td align="center"> <a href="">model</a>&nbsp;|&nbsp;<a href="">metrics</a> </td>

<tr><td align="center">TRAIN-CITY</td>
<td align="center">TBD</td>
<td align="center">TBD</td>
<td align="center">TBD </td>
<td align="center"> <a href="">model</a>&nbsp;|&nbsp;<a href="">metrics</a> </td>

<tr><td align="center">TRAIN-ALL</td>
<td align="center">TBD</td>
<td align="center">TBD</td>
<td align="center">TBD </td>
<td align="center"> <a href="">model</a>&nbsp;|&nbsp;<a href="">metrics</a> </td>

<table><tbody>

## Cross Dataset Results for Swin-L7 (36 Epoch)
<table><tbody>
<th valign="center">Dataset</th>
<th valign="center">TEST-COCO</th>
<th valign="center">TEST-ADE20K</th>
<th valign="center">TEST-CITY</th>
<th valign="center">download</th>

<tr><td align="center">TRAIN-ALL</td>
<td align="center">38.9</td>
<td align="center">37.0</td>
<td align="center">33.0 </td>
<td align="center"> <a href="https://drive.google.com/file/d/1ljAeCFlSh6BG6GM1UtzBiJdKVj1_ztSE/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://drive.google.com/file/d/1MwF1oeJ7W782m_4YXgrPvGP_IYEWY7ms/view?usp=sharing">metrics</a> </td>

<table><tbody>


## <a name="Citing Ours"></a>Citing Ours

Consider to cite **Open-World Entity Segmentation** if it helps your research.

```
@inprocedings{qi2021open,
  title={Open World Entity Segmentation},
  author={Lu Qi, Jason Kuen, Yi Wang, Jiuxiang Gu, Hengshuang Zhao, Zhe Lin, Philip Torr, Jiaya Jia},
  booktitle={arxiv},
  year={2021}
}
```
