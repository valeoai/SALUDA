import logging

import torch
import torch.nn as nn

from .backbone import *
from .decoder import *


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Network(torch.nn.Module):

    def __init__(self, in_channels, latent_size, out_channels, backbone, decoder=None, 
                frozen_backbone=False,
                no_head = False, voxel_size = 1,
                dual_seg_head = False, nb_classes = 1, da_flag = False, 
                target_in_channels = None, config=None, no_decoder = False,
                **kwargs):
        super().__init__()

        self.backbone = backbone
        self.dual_seg_head_flag = dual_seg_head
        self.config = config
        self.da_flag = da_flag
        self.latent_size = latent_size
        self.activation = nn.ReLU(inplace=True)
        self.net = get_backbone(backbone)(in_channels, self.latent_size, 
                            segmentation=True, dropout=0, spatial_prefix="encoder_",
                            voxel_size = voxel_size, nb_classes=self.config["nb_classes"], 
                            da_flag= self.da_flag, target_in_channels=target_in_channels)
        
        logging.info(f"Network -- backbone -- {count_parameters(self.net)} parameters")

        self.no_decoder = no_decoder
        if not no_decoder:
            self.decoder_name = decoder["name"]
            self.projection = eval(decoder["name"])(self.latent_size, out_channels, decoder["k"], spatial_prefix="projection_", config=config)
            logging.info(f"Network -- Surface head -- {count_parameters(self.projection)} parameters")

        if self.dual_seg_head_flag:
            #Head for semantic segmentation
            logging.info("Network -- Latent size creation {} and nb classes creation {}, larger seg head: False ".format(self.latent_size, self.config["nb_classes"]))
            self.dual_seg_head = self.net.get_linear_layer(self.latent_size, self.config["nb_classes"])
            logging.info(f"Network -- Semantic head -- {count_parameters(self.dual_seg_head)} parameters")

        if self.da_flag:
            #Add the target backbone 
            self.target_backbone = get_backbone(backbone)(target_in_channels, self.latent_size, segmentation=True, dropout=0, spatial_prefix="encoder_")
            logging.info(f"Network -- Target backbone -- {count_parameters(self.target_backbone)} parameters")


    def train(self, mode=True):
        r"""Sets the module in training mode."""      
        self.training = mode
        self.net.train(mode)

        if not self.no_decoder:
            self.projection.train(mode)
        elif not self.no_decoder:
            self.final.train(mode)
        return self


    def get_stack_item_list(self):
        item_list = self.net.get_stack_item_list()
        if not self.no_decoder:
            item_list += self.projection.get_stack_item_list()
        return item_list


    def get_cat_item_list(self):
        item_list = self.net.get_cat_item_list()
        if not self.no_decoder:
            item_list += self.projection.get_cat_item_list()
        return item_list


    def forward_pretraining(self, data, return_latents=False, return_projection=True, get_latent=False, inference=False):
        if self.dual_seg_head_flag: 
            #Head for semantic segmentation
            data["latents"] = self.net(data)
            outs_sem = self.dual_seg_head(data["latents"][:,:,None])

            if get_latent: 
                return self.projection(data, inference_mode=inference), outs_sem, data
            else:
                return self.projection(data, inference_mode=inference), outs_sem


    def from_latent(self, data,no_occupancies=False):
        return self.projection(data,no_occupancies)


    def get_latent(self, data, return_latents=False, return_projection=True):
        data["latents"] = self.net(data)
        return data 


    def forward_latents(self, data, return_latents=False, return_projection=True):
        return {"latents": self.net(data)}


    def forward_projection_inference(self, data):
        return self.projection.forward_inference(data)


    def forward_inference_feature(self, data):
        return self.projection.forward_inference_feature(data)