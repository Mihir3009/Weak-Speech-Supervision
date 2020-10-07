
from os import listdir
from os.path import join
from scipy.io import loadmat
import numpy as np
import torch


class speech_data(Dataset):

    def __init__(self, folder_path):

        self.path = folder_path
        self.files = listdir(self.path)
        self.length = len(self.files)

    def __getitem__(self, index):

        a = torch.LongTensor(1).random_(0, len(self.files))

        folder = join(self.path, self.files[int(a)])
        rand_index = torch.LongTensor(1).random_(0, len(listdir(folder)))
        d = loadmat(join(folder, listdir(folder)[rand_index]))

        # print("Selected - ",self.files[int(a)])
        if ((self.files[int(a)])[0:2] == "UA"):
            e = 1                               # for trusted data
        else:
            e = 0                               # for weakly-labeled data

        return np.array(d['mcc']), np.array(d['label']), np.array(e)

    def __len__(self):
        return self.length
