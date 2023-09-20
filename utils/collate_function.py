import importlib
import logging

import torch

torchsparse_found = importlib.util.find_spec("torchsparse") is not None
logging.info(f"Torchsparse found - {torchsparse_found}")
if torchsparse_found:
    from torchsparse.utils.collate import sparse_collate



def collate_function(list_data, 
        cat_item_list, 
        stack_item_list,
        sparse_item_list,
        ):

    data = {}

    for key in stack_item_list:
        tmp = [d[key] for d in list_data]
        tmp = torch.stack(tmp, dim=0)
        data[key] = tmp

    for key in cat_item_list:

        if "pos" in key or "dirs" in key or "dirs_non_manifold" in key: # denote a position --> add the batch as a first dim
            tmp = []
            for b_id, d in enumerate(list_data):
                pos = d[key]
                batch = torch.full((pos.shape[0], 1), fill_value=b_id)
                pos = torch.cat([batch, pos], dim=1)
                tmp.append(pos)
        else:
            tmp = [d[key] for d in list_data]
        tmp = torch.cat(tmp, dim=0)
        data[key] = tmp
    
    # sparse
    for key in sparse_item_list:
        if key in list_data[0]:
            key_inv = key+"_invmap"
            pos = 0
            tmp = []
            tmp_inv = []
            for d in list_data:
                tmp.append(d[key])
                tmp_inv.append(d[key_inv] + pos)
                pos += d[key].C.shape[0]
            data[key] = sparse_collate(tmp)
            data[key_inv] = torch.cat(tmp_inv, dim=0)

    return data