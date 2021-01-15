#coding=utf-8
from torch.utils.data import Dataset
import numpy as np

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

def matrix_generator(data_file_path, pad_len=38):
    fea_data = np.load(data_file_path, allow_pickle=True)
    pad_fea_data = np.array([np.pad(data[1], ((0, pad_len - data[1].shape[0]), (0, 0)), mode='constant') for data in fea_data])
    task_fea = pad_fea_data
    task_id = fea_data[:, 0]
    print('Output feature dimension: ', task_fea.shape)
    return task_fea, task_id



