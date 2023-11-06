#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Bowen Cheng from: https://github.com/bowenc0221/boundary-iou-api/blob/master/tools/coco_instance_evaluation.py

"""
Evaluation for COCO val2017:
python ./tools/coco_instance_evaluation.py \
    --gt-json-file COCO_GT_JSON \
    --dt-json-file COCO_DT_JSON
"""
import argparse
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

annFile = "dir_to_ann_json_file"
resFile = "dir_to_result_json_file"
simiPath = "dir_to_the_similarity_matrix_of_dataset"
cocoGt = COCO(annFile)

# remove box predictions
resFile = json.load(open(resFile))
for c in resFile:
    c.pop("bbox", None)

cocoDt = cocoGt.loadRes(resFile)
cocoEval = COCOeval(cocoGt, cocoDt, "segm")
cocoEval.params.useCats=1 # class-agnositc
cocoEval.params.openEva=1 # use open-vocabulary evaluation or not
cocoEval.params.simiPath = simiPath
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
