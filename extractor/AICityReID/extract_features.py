import os
from shutil import copyfile
import time
import math
import yaml
import argparse

from tqdm import tqdm
import numpy as np
import scipy.io
from sklearn.cluster import DBSCAN
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.optim import lr_scheduler

from .model import (
    ft_net, ft_net_angle, ft_net_dense,
    ft_net_NAS, PCB, PCB_test, CPB)
from .evaluate_gpu import calculate_result
from .evaluate_rerank import calculate_result_rerank
from .re_ranking import re_ranking, re_ranking_one
from .utils import load_network
from .losses import L2Normalization


class AICityReIDExtractor(object):

    def __init__(self, model_path=None, cfg=None, use_cuda=True):
        super().__init__()

        self.device = 'cuda:0' if use_cuda else 'cpu'
        if use_cuda:
            torch.backends.cudnn.benchmark = True

        self.height, self.width = cfg.AICITYREID.INPUT.SIZE_TEST
        if self.height == self.width:
            self.data_transforms = transforms.Compose([
                transforms.Resize(
                    (round(cfg.AICITYREID.INPUTSIZE * 1.1),
                     round(cfg.AICITYREID.INPUTSIZE * 1.1)),
                    interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.data_transforms = transforms.Compose([
                transforms.Resize(
                    (round(self.height * 1.1), round(self.width * 1.1)),
                    interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        self.net, _ = load_network(None, cfg)
        self.net.classifier.classifier = nn.Sequential()
        self.net.to(self.device).eval()

    def _proprocess(self, im_crops):
        im_crops_ = [
            Image.fromarray(np.uint8(dat)).convert('RGB')
            for dat in im_crops]
        im_batch = torch.cat(
            [self.data_transforms(im).unsqueeze(0)
            for im in im_crops_], dim=0).float()
        return im_batch

    def _fliplr(self, img):
        """flip horizontal
        """
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().to(self.device)
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    def _extract_feature(self, data):
        with torch.no_grad():
            n, c, h, w = data.size()
            data = data.to(self.device)
            # ff = torch.tensor((n, 512), dtype=torch.float32).zero_().to(self.device)
            flip_data = self._fliplr(data)
            f = self.net(flip_data)
            ff = f + self.net(data)

            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
        return ff.cpu().numpy()

    def __call__(self, img_crops):
        im_batch = self._proprocess(img_crops)
        feats = self._extract_feature(im_batch)
        return feats
