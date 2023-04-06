import logging

import torch


class CreateInputs(object):
    
    def __init__(self, item_list):
        self.item_list = item_list
        logging.info(f"CreateInputs -- {item_list}")
    
    def __call__(self, data):
        
        features = []
        for key in self.item_list:
            features.append(data[key])

        data["x"] = torch.cat(features, dim=1) 
        return data