# Converters for COCO panoptic segmentation format

Different COCO formats are described [here](http://cocodataset.org/#format-data).

## From COCO detection format to COCO panoptic format

COCO detection format is used to store both COCO instance segmentation and COCO stuff annotations.
The script `converters/detection2panoptic_coco_format.py` converts it to COCO panoptic format. Note that panoptic segmentation does not allow different segments to overlap, therefore, only dataset without overlaps can be converted.

Conversion example:
``` bash
python converters/detection2panoptic_coco_format.py \
  --input_json_file sample_data/panoptic_coco_detection_format.json \
  --output_json_file converted_data/panoptic_coco_panoptic_format.json
```

## From COCO panoptic format to COCO detection formats

The script `converts/panoptic2detection_coco_format.py` converts COCO panoptic format to COCO detection format. Each segmentation is stored as RLE. Note, some frameworks (for example [Detectron](https://github.com/facebookresearch/Detectron)) cannot work with segments stored as RLEs. There are, however, several ways ([1](https://github.com/facebookresearch/Detectron/issues/100), [2](https://github.com/facebookresearch/Detectron/pull/458)) to overcome this issue.

To convert all data to COCO detection format:
``` bash
python converters/panoptic2detection_coco_format.py \
  --input_json_file sample_data/panoptic_examples.json \
  --output_json_file converted_data/panoptic_coco_detection_format.json
```

To convert only segments of *things* classes to COCO detection format:
``` bash
python converters/panoptic2detection_coco_format.py \
  --input_json_file sample_data/panoptic_examples.json \
  --output_json_file converted_data/panoptic_coco_detection_format_things_only.json \
  --things_only
```

## Extract semantic segmentation from data in COCO panoptic format

The script `converters/panoptic2semantic_segmentation.py` merges all segments of the same category on an image into one segment.

It can be used to get semantic segmentation in COCO detection format:
``` bash
python converters/panoptic2semantic_segmentation.py \
  --input_json_file sample_data/panoptic_examples.json \
  --output_json_file converted_data/semantic_segmentation_coco_format.json
```

or to save semantic segmentation in a PNG format (pixel values corresponds to semantic categories):
``` bash
python converters/panoptic2semantic_segmentation.py \
  --input_json_file sample_data/panoptic_examples.json \
  --semantic_seg_folder converted_data/semantic_segmentation_pngs
```

In COCO stuff segmentation challenge 2017 all thing classes were merged into one *other* semantic category (`category_id=183`). Option `--things_other` in this script will do the same merging.

## Convert panoptic segmentation from 2 channels format to COCO panoptic format.

In the panoptic segmentation [paper](https://arxiv.org/abs/1801.00868) naive format to store panoptic segmentation is proposed. We call the format *2 channel format*. Each segment is defined by two labels:
(1) semantic category label and (2) instance ID label. Together this two labels form a unique pair that distinguishes one segment from another. These two labels are stored as first two channels of a PNG file correspondingly. Example of panoptic data saved in the 2 channel format can be found in [sample_data/panoptic_examples_2ch_format](https://github.com/cocodataset/panopticapi/blob/master/sample_data/panoptic_examples_2ch_format) folder.

The script `converters/2channels2panoptic_coco_format.py` converts panoptic segmentation prediction from 2 channels format to COCO panoptic format:

``` bash
python converters/2channels2panoptic_coco_format.py \
  --source_folder sample_data/panoptic_examples_2ch_format \
  --images_json_file sample_data/images_info_examples.json \
  --prediction_json_file converted_data/panoptic_coco_from_2ch.json
```

In this script `--images_json_file` json file is a file that contains information (in COCO [format](http://cocodataset.org/#format-data)) about all images that will be converted. An example is [images_info_examples.json](https://github.com/cocodataset/panopticapi/blob/master/sample_data/images_info_examples.json). Note, that conversion script assumes that PNGs with 2 channel formatted panoptic segmentations have the following name structure `image['file_name'].replace('.jpg', '.png')`.
