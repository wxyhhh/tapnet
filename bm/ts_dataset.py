import torch
from torch.utils import data
from utils import *

class TSDataset(data.Dataset):

    'Characterizes a dataset for PyTorch'
    # def __init__(self, list_ids, labels):
    #     'Initialization'
    #     self.labels = labels
    #     self.list_ids = list_ids

    def __init__(self, features, list_ids, labels):
        'Initialization'
        self.features = features
        self.list_ids = list_ids
        self.labels = labels

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_ids)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        id = self.list_ids[index]

        # Load data and get label
        X = self.features[id, ]
        y = self.labels[id]

        return X, y