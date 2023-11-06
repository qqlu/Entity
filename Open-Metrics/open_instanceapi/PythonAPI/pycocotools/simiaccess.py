import os
import numpy as np
import pandas as pd

class SIMIaccess:
    def __init__(self, path=None):
        assert os.path.exists(path), 'similarity matrix {} is not exists.'.format(path)
        df_sim = pd.read_csv(path, index_col=0)
        self.matrix = df_sim.values
        self.labels = list(df_sim.columns)

    def findSimi(self, dt_label, gt_label):
        assert isinstance(dt_label, np.int64), 'detection label should be in int type, but is {} type'.format(type(dt_label))
        assert isinstance(gt_label, np.int64), 'groundtruth label should be in int type, but is {} type'.format(type(gt_label))

        dt_label = str(dt_label)
        gt_label = str(gt_label)

        if dt_label == gt_label:
            return 1
        elif dt_label in self.labels and gt_label in self.labels:
            index_i = self.labels.index(dt_label)
            index_j = self.labels.index(gt_label)
            simi = self.matrix[index_i, index_j]
            return simi
        else:
            return 0
