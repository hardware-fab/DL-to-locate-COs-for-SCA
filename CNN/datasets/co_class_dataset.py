"""
Authors : 
    Giuseppe Chiari (giuseppe.chiari@polimi.it),
    Davide Galli (davide.galli@polimi.it),
    Francesco Lattari (francesco.lattari@polimi.it),
    Matteo Matteucci (matteo.matteucci@polimi.it),
    Davide Zoni (davide.zoni@polimi.it)
"""

import os
import numpy as np
from torch.utils.data import Dataset


class CoClassifierDataset(Dataset):
    def __init__(self, data_dir, labels, which_subset='train'):

        self.labels = labels
        self.type = type

        print("Loading {} set..".format(which_subset) + " in " +
              data_dir + '/' + '{}_set.npy'.format(which_subset))

        self.windows = np.load(
            os.path.join(data_dir, '{}_set.npy'.format(which_subset)), mmap_mode='r+')

        self.target = np.load(
            os.path.join(data_dir, '{}_labels.npy'.format(which_subset)),  mmap_mode='r+')

        print("Number of windows in the {} set: {}".format(
            which_subset, self.windows.shape[0]*self.windows.shape[1]))

        self.index_map = []
        index = 0
        for trace_index, wins in enumerate(self.windows):  # First dimension
            for window_index in range(0, len(wins)):  # Second dimension
                self.index_map.append([trace_index, window_index])
                index += 1

    def __len__(self):
        return self.windows.shape[0]*self.windows.shape[1]

    def __getitem__(self, index):
        trace_index, window_index = self.index_map[index]
        x = self.windows[trace_index, window_index]
        which_target = int(self.target[trace_index, window_index])

        if which_target == 0 or which_target == 2 or which_target == 1:
            y = int(self.labels[0])
        elif which_target == 3:
            y = int(self.labels[1])

        return x, y
