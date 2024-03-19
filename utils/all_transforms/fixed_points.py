from os import replace
import numpy as np
from torchvision import transforms
from PIL import Image
import random
import logging
import torch
import re
import math
import numbers

class FixedPoints(object):

    # extension of the Fixed points from torch_geometric with a given list of item to sample

    def __init__(self, num, replace=True, allow_duplicates=False, item_list=None):
        self.num = num
        self.replace = replace
        self.allow_duplicates = allow_duplicates
        self.item_list = item_list

    def __call__(self, data):
        if self.item_list is None:
            num_nodes = data.num_nodes
        else:
            num_nodes = data[self.item_list[0]].shape[0]

        # Sampling
        if self.replace:
            choice = np.random.choice(num_nodes, self.num, replace=True)
            choice = torch.from_numpy(choice).to(torch.long)
        elif not self.allow_duplicates:
            choice = torch.randperm(num_nodes)[:self.num]
        else:
            choice = torch.cat([
                torch.randperm(num_nodes)
                for _ in range(math.ceil(self.num / num_nodes))
            ], dim=0)[:self.num]

        # selecting elements
        if self.item_list is None:
            for key, item in data:
                if bool(re.search('edge', key)):
                    continue
                if (torch.is_tensor(item) and item.size(0) == num_nodes
                        and item.size(0) != 1):
                    data[key] = item[choice]
        else:
            for key, item in data:
                if key in self.item_list:
                    if bool(re.search('edge', key)):
                        continue
                    if (torch.is_tensor(item) and item.size(0) != 1):
                        data[key] = item[choice]
        return data

    def __repr__(self):
        return '{}({}, replace={})'.format(self.__class__.__name__, self.num,
                                           self.replace)