import torch
import numpy
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pickle
import os

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data, labels = sample['data'],sample['labels']
        return {'data': torch.from_numpy(data),
                'labels': torch.from_numpy(labels)}

class TwentyFiveMixtureDataset(Dataset):
    def __init__(self,train=True,transform=None):
        self.train_pickle_file = 'tmp/train_dataset.pk'
        self.test_pickle_file = 'tmp/test_dataset.pk'
        self.transform=transform
        self.train = train
        if self.train:
            file = open(self.train_pickle_file,'rb')
        else:
            file = open(self.test_pickle_file,'rb')
        dataset = pickle.load(file)
        file.close()
        self.data = dataset['data']
        self.labels = dataset['labels']


    def __len__(self):
        if self.train:
            return 100000
        else:
            return 10000

    def __getitem__(self, idx):
        data = self.data[idx,:]
        label = self.labels[idx]
        # sample = {'data': data, 'labels': label}
        sample = (data,label)
        if self.transform:
            sample = self.transform(sample)

        return sample

