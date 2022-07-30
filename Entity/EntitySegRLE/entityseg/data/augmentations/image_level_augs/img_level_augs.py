import random
import numpy as np
from detectron2.augmentations.image_level_augs.zoom_in import Zoom_in
from detectron2.augmentations.image_level_augs.zoom_out import Zoom_out
from detectron2.augmentations.image_level_augs.scale_jitter import scale_jitter


class Img_augs(object):
    def __init__(self, img_augs_dict):
        self.zoom_in_prob = img_augs_dict['zoom_in']['prob']
        self.zoom_out_prob = img_augs_dict['zoom_out']['prob']

        zoom_in_ratio_range = np.linspace(1.0, 1.5, 11)[1:]
        zoom_out_ratio_range = np.linspace(1.0, 0.5, 11)[1:]

        self.Zoom_in = Zoom_in(ratio=zoom_in_ratio_range[img_augs_dict['zoom_in']['level']])
        self.Zoom_out = Zoom_out(ratio=zoom_out_ratio_range[img_augs_dict['zoom_out']['level']])

    def __call__(self, tensor, target):
        
        if random.random() < self.zoom_in_prob:
            tensor_out, target_out = self.Zoom_in(tensor=tensor, target=target)
        elif random.random() < self.zoom_in_prob + self.zoom_out_prob:
            tensor_out, target_out = self.Zoom_out(tensor=tensor, target=target)
        else:
            tensor_out, target_out = tensor, target

        jitter_factor = random.sample(np.linspace(0.75, 1.25, 5).tolist(), 1)[0]
        tensor_out, target_out, _ = scale_jitter(tensor_out, target_out, jitter_factor)

        return tensor_out, target_out
