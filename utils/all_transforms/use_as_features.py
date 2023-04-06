import torch


class UseAsFeatures(object):
    def __init__(self, item_list):
        self.item_list = item_list
    
    def __call__(self, data):

        features = []
        for key in self.item_list:
            value = data[key]
            if len(value.shape)==1:
                value = value.unsqueeze(1)
            features.append(value)

        features = torch.cat(features, dim=1)
        data["x"] = features

        return data