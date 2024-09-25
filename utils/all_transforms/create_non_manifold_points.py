import torch
import torch.nn.functional as F


class CreateNonManifoldPoints(object):

    def __call__(self, data):

        pos = data["pos"]
        label = data["y"]

        # build the non manifold points
        if "dirs" in data.keys():
            dirs = data["dirs"]
        else:
            dirs = F.normalize(pos, dim=1)
        
        pos_in = pos + 0.1 * dirs * torch.rand((pos.shape[0],1))
        pos_out = pos - 0.1 * dirs * torch.rand((pos.shape[0],1))
        pos_out_far = (pos - 0.1 * dirs) * torch.rand((pos.shape[0],1))
        data["pos_non_manifold"] = torch.cat([pos_in, pos_out, pos_out_far], dim=0)
        data["label_non_manifold"] = torch.cat([label, label, torch.zeros_like(label)], dim=0)

        occ_in = torch.ones(pos_in.shape[0], dtype=torch.long)
        occ_out = torch.zeros(pos_out.shape[0], dtype=torch.long)
        occ_out_far = torch.zeros(pos_out_far.shape[0], dtype=torch.long)
        data["occupancies"] = torch.cat([occ_in, occ_out, occ_out_far], dim=0)

        if "intensities" in data:
            intensities_in = data["intensities"]
            intensities_out = data["intensities"]
            intensities_out_far = torch.full_like(intensities_out, fill_value=-1)
            data["intensities_non_manifold"] = torch.cat([intensities_in, intensities_out, intensities_out_far], dim=0)
        
        return data