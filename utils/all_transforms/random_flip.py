import torch


class RandomFlip(object):
    
    def __init__(self, item_list) -> None:
        self.item_list = item_list

    def __call__(self, data):

        if torch.randint(0, 2, size=(1,)).item():
            for item in self.item_list:
                data[item][:,0] = -data[item][:,0]
                if ("second_" + item) in data.keys():
                    data["second_" + item][:,0] = -data["second_" + item][:,0]
        
        return data