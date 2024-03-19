import torch
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.collate import sparse_collate

def collate_function_merged(list_data):

    data = {}

    for key in ["source_labels", "source_features",  "source_per_point_labels", "source_coordinates", "target_labels", "target_features",  "target_per_point_labels", "target_coordinates"]:       
        if key in list_data[0]:
            tmp = []
            for b_id, d in enumerate(list_data):
                pos = d[key]

                batch = torch.full((pos.shape[0], 1), fill_value=b_id)
                if "labels" in key or "per_point_labels" in key:
                    pos = torch.cat([batch, pos[:,None]], dim=1)
                else:
                    pos = torch.cat([batch, pos], dim=1)     
                tmp.append(pos)
            tmp = torch.cat(tmp, dim=0)
            data[key] = tmp
    
    # sparse
    sparse_item_list = ["source_sparse_input", "target_sparse_input"]
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