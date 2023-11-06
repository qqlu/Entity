# Open Evaluation Metrics API (Beta version)
Hao Zhou, Tiancheng Shen, Xu Yang, Hai Huang, Xiangtai Li, Lu Qi, Ming-Hsuan Yang

This package contain evalution codes of Open mIoU, Open AP, and Open PQ. We will release the similarity score matrix of each dataset soon.

## Installation
Clone Repo:
```bash
pip clone 
```
Install the API of Open AP:
```bash
cd open_instanceapi/PythonAPI
pip install -e .
```
Install the API of Open PQ:
```bash
cd open_panopticapi
pip install -e .
```
The usage of Open mIoU:
```bash
cd open_semanticseg
cp * /dir_to_detectron2/evaluation # copy files to the evaluation folder of detectron2
```

### Open AP
For the usage of Open AP, please refer to:
```bash
open_instanceapi/evaluate_coco.py
```

### Open PQ
For the usage of Open PQ, please refer to:
```bash
open_panopticapi/panopticapi/evaluation.py
```

### Open mIoU
For the usage of Open mIoU, please refer to the document of mIoU in detectron2.
