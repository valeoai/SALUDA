import torch


class Transpose(object):
    
    # extension of the Fixed points from torch_geometric with a given list of item to sample

    def __init__(self, item_list):
        self.item_list = item_list

    def __call__(self, data):

        for key in self.item_list:
            data[key] = data[key].transpose(0,1)
        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.item_list)