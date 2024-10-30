import torch.nn.functional as F


class CreateDirs(object):
    
    def __call__(self, data):

        dirs = F.normalize(data["pos"], dim=1)
        data["dirs"] = dirs


        # if second frame --> decimation of the second frame
        if "second_pos" in data.keys():
            second_dirs = F.normalize(data["second_pos"], dim=1)
            data["second_dirs"] = second_dirs

        return data

class CreateDirs_non_manifold(object):
    def __call__(self, data):
    
        dirs_non_manifold = F.normalize(data["pos_non_manifold"], dim=1)
        data["dirs_non_manifold"] = dirs_non_manifold
        
        return data