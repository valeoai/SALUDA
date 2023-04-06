import importlib

torchsparse_found = importlib.util.find_spec("torchsparse") is not None
import logging

logging.info(f"Torchsparse found - {torchsparse_found}")
if torchsparse_found:
    from torchsparse.utils.quantize import sparse_quantize
    from torchsparse import SparseTensor

import numpy as np
import torch


class TorchSparseQuantize(object):
    
    def __init__(self, voxel_size) -> None:
        self.voxel_size = voxel_size

    def __call__(self, data):

        pc_ = np.round(data["pos"].numpy() / self.voxel_size).astype(np.int32)
        pc_ -= pc_.min(0, keepdims=1) 
        coords, indices, inverse_map = sparse_quantize(pc_,
                                            return_index=True,
                                            return_inverse=True)
        coords = torch.tensor(coords, dtype=torch.int)
        indices = torch.tensor(indices)
        feats = data["x"][indices]
        inverse_map = torch.tensor(inverse_map, dtype=torch.long)
        data["sparse_input"] = SparseTensor(coords=coords, feats=feats)
        data["sparse_input_invmap"] = inverse_map

        return data