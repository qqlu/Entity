# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import itertools
import json
import numpy as np
import os
from collections import OrderedDict
import PIL.Image as Image
import torch
import pandas as pd

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager

from detectron2.evaluation import SemSegEvaluator
from .ow_sem_seg_simi_matrix import OWSemSegSimiMatrix
import copy


class OWSemSegEvaluator(SemSegEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        num_classes=None,
        ignore_label=None,
        post_process_func=None,
        simi_matrix_dir=None,
    ):
        super().__init__(
            dataset_name,
            distributed=distributed,
            output_dir=output_dir,
            num_classes=num_classes,
            ignore_label=ignore_label,
        )
        meta = MetadataCatalog.get(dataset_name)
        try:
            self._evaluation_set = meta.evaluation_set
        except AttributeError:
            self._evaluation_set = None
        self.post_process_func = (
            post_process_func
            if post_process_func is not None
            else lambda x, **kwargs: x
        )
        
        if simi_matrix_dir is not None:
            self.simi_matrix = OWSemSegSimiMatrix(simi_matrix_dir).findSimiMatrix()
            print('='*100)
            print('The dir of similarity matrix used in this experiment is:', simi_matrix_dir)
            print('='*100)
            print(simi_matrix_dir)
            print('true way')
        else:
            self.simi_matrix = None

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        file = os.path.join(self._output_dir, "per_image_iou.txt")
        fn = open(file, 'a')
        for input, output in zip(inputs, outputs):
            output = self.post_process_func(
                output["sem_seg"], image=np.array(Image.open(input["file_name"]))
            )
            output = output.argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=np.int)

            with PathManager.open(
                self.input_file_to_gt_file[input["file_name"]], "rb"
            ) as f:
                gt = np.array(Image.open(f), dtype=np.int)

            gt[gt == self._ignore_label] = self._num_classes

            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape) # has a shape of (num_class+1, num_class+1)

            # begin from this line
            cur_conf_matrix = np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)
            cur_conf_matrix = cur_conf_matrix.astype(np.float64)

            cur_ow_conf_matrix = copy.deepcopy(cur_conf_matrix)
            ow_tp = np.sum(cur_ow_conf_matrix * self.simi_matrix, axis=0)
            cur_ow_conf_matrix *= (1-self.simi_matrix)
            row, col = np.diag_indices_from(cur_ow_conf_matrix)
            cur_ow_conf_matrix[row,col] = ow_tp

            tp = cur_conf_matrix.diagonal()[:-1].astype(np.float)
            iou = np.full(self._num_classes, np.nan, dtype=np.float)
            pos_gt = np.sum(cur_conf_matrix[:-1, :-1], axis=0).astype(np.float)
            pos_pred = np.sum(cur_conf_matrix[:-1, :-1], axis=1).astype(np.float)
            acc_valid = pos_gt > 0
            iou_valid = (pos_gt + pos_pred) > 0
            union = pos_gt + pos_pred - tp
            iou[acc_valid] = tp[acc_valid] / union[acc_valid]
            miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)

            tp_ow = cur_ow_conf_matrix.diagonal()[:-1].astype(np.float)
            iou_ow = np.full(self._num_classes, np.nan, dtype=np.float)
            pos_gt_ow = np.sum(cur_ow_conf_matrix[:-1, :-1], axis=0).astype(np.float)
            pos_pred_ow = np.sum(cur_ow_conf_matrix[:-1, :-1], axis=1).astype(np.float)
            acc_valid_ow = pos_gt_ow > 0
            iou_valid_ow = (pos_gt_ow + pos_pred_ow) > 0
            union_ow = pos_gt_ow + pos_pred_ow - tp_ow
            iou_ow[acc_valid_ow] = tp_ow[acc_valid_ow] / union_ow[acc_valid_ow]
            miou_ow = np.sum(iou_ow[acc_valid_ow]) / np.sum(iou_valid_ow)

            fn.write(input["file_name"]+'\t'+"mIoU\t"+str(miou)+'\t'+'OWmIoU\t'+str(miou_ow)+'\t'+'difference\t'+str(miou_ow-miou)+'\t')
            perIoU = {}
            perOWIoU = {}
            for i, name in enumerate(self._class_names):
                if not np.isnan(iou[i]):
                    perIoU["IoU-{}".format(name)] = 100 * iou[i]
                if not np.isnan(iou_ow[i]):
                    perOWIoU["IoU-{}".format(name)] = 100 * iou_ow[i]
            fn.write(str(perIoU)+'\t'+str(perOWIoU)+'\n')

            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))
        fn.close()

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

        if self.simi_matrix is not None:
            self._conf_matrix = self._conf_matrix.astype(np.float64) 
            ow_tp = np.sum(self._conf_matrix * self.simi_matrix, axis=0)
            self._conf_matrix *= (1 - self.simi_matrix)
            row, col = np.diag_indices_from(self._conf_matrix)
            self._conf_matrix[row,col] = ow_tp

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))

        acc = np.full(self._num_classes, np.nan, dtype=np.float)
        iou = np.full(self._num_classes, np.nan, dtype=np.float)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res["IoU-{}".format(name)] = 100 * iou[i]
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(self._class_names):
            res["ACC-{}".format(name)] = 100 * acc[i]
        if self._evaluation_set is not None:
            for set_name, set_inds in self._evaluation_set.items():
                iou_list = []
                set_inds = np.array(set_inds, np.int)
                mask = np.zeros((len(iou),)).astype(np.bool)
                mask[set_inds] = 1
                miou = np.sum(iou[mask][acc_valid[mask]]) / np.sum(iou_valid[mask])
                pacc = np.sum(tp[mask]) / np.sum(pos_gt[mask])
                res["mIoU-{}".format(set_name)] = 100 * miou
                res["pAcc-{}".format(set_name)] = 100 * pacc
                iou_list.append(miou)
                miou = np.sum(iou[~mask][acc_valid[~mask]]) / np.sum(iou_valid[~mask])
                pacc = np.sum(tp[~mask]) / np.sum(pos_gt[~mask])
                res["mIoU-un{}".format(set_name)] = 100 * miou
                res["pAcc-un{}".format(set_name)] = 100 * pacc
                iou_list.append(miou)
                res["hIoU-{}".format(set_name)] = (
                    100 * len(iou_list) / sum([1 / iou for iou in iou_list])
                )
        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results
