import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import avg_pool_x, knn, radius


class InterpNetBase(torch.nn.Module):

    def __init__(self, latent_size, out_channels, K=1, spatial_prefix="", 
            contrast_loss=False,
            intensity_loss=False,
            patch_similarity_loss=False,
            predict_translation=False,
            all_prediction=False,
            bow=False,
            use_dirs=True, radius_search=True, config=None):

        super().__init__()

        logging.info(f"InterpNet - Mean - radius={K}")

        self.contrast_loss = contrast_loss
        self.intensity_loss = intensity_loss
        self.predict_translation = predict_translation
        self.bow = bow
        self.all_prediction = all_prediction
        self.dirs = use_dirs
        self.patch_similarity_loss = patch_similarity_loss

        self.out_channels = out_channels
        if self.intensity_loss:
            self.out_channels += 1

        in_size = latent_size
        if self.dirs:
            in_size += 3
        if not self.predict_translation:
            in_size += 3
        
        self.fc_in = torch.nn.Linear(in_size, latent_size)
        mlp_layers = [torch.nn.Linear(latent_size, latent_size) for _ in range(2)]
        self.mlp_layers = nn.ModuleList(mlp_layers)
        self.fc_out = torch.nn.Linear(latent_size, self.out_channels)

        self.activation = torch.nn.ReLU()
        self.spatial_prefix = spatial_prefix

        if radius_search:
            self.radius = K
            self.K = None
        else:
            self.K = int(K)
            self.radius = None

    def get_stack_item_list(self):
        return []

    def get_cat_item_list(self):
        return []

    def forward(self, data, inference_mode=False):

        # get the data
        pos = data["pos"]
        latents = data["latents"]
        dirs = data["dirs"][:,1:]
        pos_non_manifold = data["pos_non_manifold"]

        # create batch and pos
        batch_source = pos[:, 0].long()
        pos_source = pos[:, 1:]
        batch_target = pos_non_manifold[:, 0].long()
        pos_target = pos_non_manifold[:, 1:]

        # neighborhood search
        if self.radius is not None:
            row, col = radius(pos_source, pos_target, self.radius, batch_source, batch_target)
        else:
            row, col = knn(pos_source, pos_target, self.K, batch_source, batch_target)

        # Moved translation after nn computation
        # --> collapse to xy plane
        if self.predict_translation:
            # operateTranslation
            pos_source = pos_source - latents[:, :3]
            latents = latents[:, 3:]

        # compute reltive position between query and input point cloud
        pos_relative = pos_target[row] - pos_source[col]

        # get the corresponding latent vectors
        latents = latents[col]

        # create the input of the decoder
        if self.dirs:
            x = torch.cat([latents, pos_relative, dirs[col]], dim=1)
        else:
            x = torch.cat([latents, pos_relative], dim=1)

        if (not inference_mode) and ("occupancies" in data):
            occupancies = data["occupancies"][row]

        # MLP
        x = self.fc_in(x.contiguous())
        for i, l in enumerate(self.mlp_layers):
            x = l(self.activation(x))

        # Final layer
        x = self.fc_out(x)

        # inference mode
        if inference_mode:
            # average pooling over all predictions
            x, _ = avg_pool_x(row, x, batch_source[col])

            output_pos_count = 0
            predictions = torch.full((pos_target.shape[0],), fill_value=-1e7, dtype=torch.float, device=x.device)

            target_point_ids = torch.unique(row)
            predictions[target_point_ids] = x[:, output_pos_count]
            
            return_data = {"predictions":predictions}
            if ("occupancies" in data):
                return_data["occupancies"] = data["occupancies"]
            return return_data

        output_pos_count = 0
        return_data = {"predictions":x[:, output_pos_count], "occupancies": occupancies}

        # Reconstruction loss
        recons_loss = F.binary_cross_entropy_with_logits(x[:,output_pos_count], occupancies.float())
        return_data["recons_loss"] = recons_loss
        output_pos_count += 1
 
        return return_data


    def forward_inference(self, data):

        return self.forward(data, inference_mode=True)


# All neighbors in radius
class InterpAllRadiusNoDirsNet(InterpNetBase):
    def __init__(self, latent_size, out_channels, K=1, spatial_prefix="", config=None):
        super().__init__(latent_size, out_channels, K, spatial_prefix, all_prediction=True, use_dirs=False, config=config)

# With directions
class InterpAllRadiusNet(InterpNetBase):
    def __init__(self, latent_size, out_channels, K=1, spatial_prefix="", config=None):
        super().__init__(latent_size, out_channels, K, spatial_prefix, all_prediction=True, config=config)

# Original POCO head
class InterpAttentionKHeadsNet(torch.nn.Module):
    
    def __init__(self, latent_size, out_channels, K=16, spatial_prefix="", config=None):
        super().__init__()
        
        self.latent_size = latent_size
        self.config = config
        logging.info(f"InterpNet - Simple - K={K}")
        if "use_no_dirs_rec_head_flag" in self.config and self.config["use_no_dirs_rec_head_flag"]:
            logging.info(f"No dirs used for reconstruction head used")
            self.fc1 = torch.nn.Linear(latent_size+3, latent_size)
        else:
            self.fc1 = torch.nn.Linear(latent_size+6, latent_size)
        self.fc2 = torch.nn.Linear(latent_size, latent_size)
        self.fc3 = torch.nn.Linear(latent_size, latent_size)

        self.fc8 = torch.nn.Linear(latent_size, out_channels)
        self.activation = torch.nn.ReLU()
        self.spatial_prefix = spatial_prefix
        self.fc_query = torch.nn.Linear(latent_size, 64)
        self.fc_value = torch.nn.Linear(latent_size, latent_size)

        self.K = K
        self.spatial_prefix = spatial_prefix
        self.decoder_ratio = config["decoder_ratio"]

        if "rotation_head_flag" in self.config and self.config['rotation_head_flag']:
            logging.info(f"Rotation of the reconstruction ball")

    def get_stack_item_list(self):
        return []

    def get_cat_item_list(self):
        return []
        
    def forward(self, data, no_occupancies=False, inference=False, return_feature=False, inference_mode=None):

        # get the data
        pos = data["pos"]
        latents = data["latents"]

        dirs = data["dirs"][:,1:]
        pos_non_manifold = data["pos_non_manifold"]
        pos_source = pos[:,1:]
        pos_target = pos_non_manifold[:,1:]

        batch_source = pos[:,0].long()
        batch_target = pos_non_manifold[:,0].long()
        row, col = knn(pos_source, pos_target, self.K, batch_source, batch_target)
        
        pos = pos_target[row] - pos_source[col]
        
        if "use_no_dirs_rec_head_flag" in self.config and self.config["use_no_dirs_rec_head_flag"]:
            x = torch.cat([latents[col], pos], dim=1)
        else:
            x = torch.cat([latents[col], pos, dirs[col]], dim=1)

        if not return_feature:
            occupancies = data["occupancies"]#[row]

        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))

        query = self.fc_query(x)
        value = self.fc_value(x)

        attention = torch.nn.functional.softmax(query, dim=-1).mean(dim=1)

        attention = attention.view(-1,1)
        product = torch.mul(attention, value)
        product = product.view(pos_non_manifold.shape[0], -1, self.latent_size)

        x = torch.sum(product, dim=1)
        pred = self.fc8(x)
        
        
        if return_feature: 
            return_data = {"predictions":pred, "feature":x}
        else:
            return_data = {"predictions":pred, 
                "occupancies": occupancies} 
        return return_data

    def forward_inference(self, data):
        #inference mode attribute is here only used for 
        return self.forward(data, inference=True)

    def forward_inference_feature(self, data):
        
        return self.forward(data, inference=True, return_feature=True)