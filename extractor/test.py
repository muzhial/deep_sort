import os
import yaml
from easydict import EasyDict as edict

import numpy as np
from PIL import Image
import torch

from AICityReID.extract_features import AICityReIDExtractor


class YamlParser(edict):
    """
    This is yaml parser based on EasyDict.
    """
    def __init__(self, cfg_dict=None, config_file=None):
        if cfg_dict is None:
            cfg_dict = {}
        if config_file is not None:
            assert(os.path.isfile(config_file))
            with open(config_file, 'r') as fo:
                cfg_dict.update(yaml.safe_load(fo.read()))

        super(YamlParser, self).__init__(cfg_dict)

    def merge_from_file(self, config_file):
        with open(config_file, 'r') as fo:
            self.update(yaml.safe_load(fo.read()))

    def merge_from_dict(self, config_dict):
        self.update(config_dict)


def get_config(config_file=None):
    return YamlParser(config_file=config_file)


if __name__ == '__main__':
    data = torch.randn(2, 3, 320, 320)
    data = [
        np.random.randn(320, 240, 3),
        np.random.randn(240, 345, 3)
    ]
    data_ = [Image.fromarray(np.uint8(dat)).convert('RGB')
            for dat in data]
    config = get_config('../configs/deep_sort.yaml')
    ex = AICityReIDExtractor(cfg=config)
    res = ex(data_)
    print(res.shape)
