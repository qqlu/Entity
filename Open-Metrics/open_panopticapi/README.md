# COCO 2018 Panoptic Segmentation Task API (Beta version)
This API is an experimental version of [COCO 2018 Panoptic Segmentation Task API](http://cocodataset.org/#panoptic-2018).

To install panopticapi, run:
```
pip install git+https://github.com/cocodataset/panopticapi.git
```

## Summary
**Evaluation script**

[panopticapi/evaluation.py](panopticapi/evaluation.py) calculates [PQ metrics](http://cocodataset.org/#panoptic-eval).
For more information about the script usage: `python -m panopticapi.evaluation --help`

**Format converters**

COCO panoptic segmentation is stored in a new [format](http://cocodataset.org/#format-data). Unlike COCO detection format that stores each segment independently, COCO panoptic format stores all segmentations for an image in a single PNG file. This compact representation naturally maintains non-overlapping property of the panoptic segmentation.

We provide several converters for COCO panoptic format. Full description and usage examples are available [here](https://github.com/cocodataset/panopticapi/blob/master/CONVERTERS.md).

**Semantic and instance segmentation heuristic combination**

We provide [a simple script](panopticapi/combine_semantic_and_instance_predictions.py)
that heuristically combines semantic and instance segmentation predictions into panoptic segmentation prediction.

The merging logic of the script is described in the panoptic segmentation [paper](https://arxiv.org/abs/1801.00868).
In addition, this script is able to filter out stuff predicted segments that have their area below the threshold defined by `--stuff_area_limit` parameter.

For more information about the script logic and usage: `python -m panopticapi.combine_semantic_and_instance_predictions.py --help`

**COCO panoptic segmentation challenge categories**

Json file [panoptic_coco_categories.json](panoptic_coco_categories.json) contains the list of all categories used in COCO panoptic segmentation challenge 2018.

**Visualization**

[visualization.py](visualization.py) provides an example of generating visually appealing representation of the panoptic segmentation data.

## Contact
If you have any questions regarding this API, please contact us at alexander.n.kirillov-at-gmail.com.
