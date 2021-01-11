#coding=utf-8
import numpy as np
from torch.utils.data import Dataset

class ACP_dataset(Dataset):
    def __init__(self, feature_list, target_list):
        self.features = feature_list
        self.label = target_list
        self.set_length = len(self.label)
    def __getitem__(self, index):
        fea = self.features[index]
        lab = self.label[index]
        return fea, lab
    def __len__(self):
        return len(self.features)

class ACP_MT_dataset(Dataset):
    def __init__(self, feature_list, target_list, mask_list=None):
        self.features = feature_list
        self.label = target_list
        self.flag = 0
        if type(mask_list) == np.ndarray:
            self.mask = mask_list
            self.flag = 1
        self.set_length = len(self.label)
    def __getitem__(self, index):
        fea = self.features[index]
        lab = self.label[index]
        if self.flag: msk = self.mask[index]
        else: msk = -1
        return fea, lab, msk
    def __len__(self):
        return len(self.features)

