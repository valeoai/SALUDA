import glob
import importlib
import logging
import os
import pickle
import sys
from pathlib import Path

# Basic libs
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.spatial import cKDTree
from torch._C import Value
from torch_geometric.data import Data, Dataset

def class_mapping_da(config):
    #Learning map from SyntheticLIDAR to SemanticPOSS
        if (config["source_dataset_name"] == "SynLidar" or  config["target_dataset_name"] == "SynLidar") or \
            (config["source_dataset_name"] == "SemanticPOSS" and  config["target_dataset_name"] == "SemanticPOSS"):
            maps = {0: 0,   # "unlabeled",
                    1:0,
                    2: 0,
                    3: 0,
                    4: 1,   # "person",
                    5: 1,  # "person",
                    6: 2,   # "rider",
                    7: 3,   # "car",
                    8: 4,   # "trunk",
                    9: 5,   # "plants",
                    10: 6,   #  "traffic sign"
                    11: 6,   #  "traffic sign"
                    12: 6,   #  "traffic sign"
                    13: 7,   #  "pole",
                    14: 8,   #  "garbage-can",
                    15: 9,   #  "building",
                    16: 10,   #  "cone/stone",
                    17: 11,   #  "fence",
                    18: 0,
                    19: 0,
                    20: 0,
                    21: 12,  #  "bike",
                    22: 13}   #  "ground"
        elif (config["source_dataset_name"] == "NuScenes" or config["target_dataset_name"] == "NuScenes"):
            maps = {0:0, # unlabeled-->unlabeled
                    1:0,#-->unlabeled
                    2:0,#-->unlabeled
                    3:0,#-->unlabeled
                    4:1,# person-->person
                    5:1,# person-->person
                    6:2,# rider-->bike
                    7:3,# car-->car
                    8:0,# trunk-->unlabeled
                    9:5,# plants-->vegetation
                    10:6,# traffic sign-->manmade
                    11:6,# traffic sign-->manmade
                    12:6,# traffic sign-->manmade
                    13:6,# pole-->manmade
                    14:6,# garbage-can-->manmade
                    15:6,# building-->manmade
                    16:6,# cone/stone-->manmade
                    17:6,# fence-->manmade
                    18:0,#-->unlabeled
                    19:0,#-->unlabeled
                    20:0,#-->unlabeled
                    21:2,# bike-->bike
                    22:4 }# ground-->ground
        else: 
            raise ValueError(f"No mapping for source {config['source_dataset_name']} and target {config['target_dataset_name']}")
        
        return maps 
    
class SemanticPOSS(Dataset):
    def __init__(self,root,
                 split="training",
                 transform=None, 
                 dataset_size=None,
                 skip_ratio=1,
                 da_flag =False, config=None,
                 **kwargs):

        
        super().__init__(root, transform, None)


        self.split = split
        self.n_frames = 1
        self.da_flag = da_flag
        self.config = config

        
        logging.info(f"SemanticPoss - split {split}")

        assert(split in ["train", "val", "test", "verifying", "parametrizing"])
        if split == "verifying":
            self.sequences = ['{:02d}'.format(i) for i in range(6) if i == 3]
        elif split == "parametrizing":
            self.sequences = ['{:02d}'.format(i) for i in range(6) if (i == 0 and i == 5)]
        elif split == "train":
            self.sequences = ['{:02d}'.format(i) for i in range(6) if i != 3]
        elif split == "val":
            self.sequences = ['{:02d}'.format(i) for i in range(6) if i ==3]
        elif split == "test":
            raise ValueError('Unknown set for SemanticPoss data: ', split)
        else:
            raise ValueError('Unknown set for SemanticPoss data: ', split)

        self.name = 'SemanticPOSSDataset'
        self.maps = class_mapping_da(config)
        self.all_files = []
        self.all_labels = []

        remap_dict_val = self.maps
        max_key = max(remap_dict_val.keys())
        remap_lut_val = np.zeros((max_key + 100), dtype=np.int32)
        remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())

        self.remap_lut_val = remap_lut_val

        for sequence in self.sequences:
            num_frames = len(os.listdir(os.path.join(self.root, "sequences", sequence, 'labels')))

            for f in np.arange(num_frames):
                files = os.path.join(self.root, "sequences", sequence, 'velodyne', f'{int(f):06d}.bin')
                labels = os.path.join(self.root, "sequences", sequence, 'labels', f'{int(f):06d}.label')

                if os.path.exists(files) and os.path.exists(labels):
                    self.all_files.append(files)
                    self.all_labels.append(labels)
 
        self.color_map = np.array([(255, 255, 255),  # unlabelled
                                    (250, 178, 50),  # person
                                    (255, 196, 0),  # rider
                                    (25, 25, 255),  # car
                                    (107, 98, 56),  # trunk
                                    (157, 234, 50),  # plants
                                    (173, 23, 121),  # traffic-sign
                                    (83, 93, 130),  # pole
                                    (23, 173, 148),  # garbage-can
                                    (233, 166, 250),  # building
                                    (173, 23, 0),  # traffic-cone
                                    (255, 214, 251),  # fence
                                    (187, 0, 255),  # bicycle
                                    (164, 173, 104)])/255.  # other-ground

    def __len__(self):
        return len(self.all_files)

    def get(self, idx):
        pcd_tmp = self.all_files[idx]
        label_tmp = self.all_labels[idx]

        pcd = np.fromfile(pcd_tmp, dtype=np.float32).reshape((-1, 4))
        label = self.load_label_poss(label_tmp)
        points = pcd[:, :3]
        colors = np.ones((points.shape[0], 1), dtype=np.float32)
        #colors = points[:, 3][..., np.newaxis] #intensieties, not sure if available for SemanticPOSS
        data = {'points': points, 'colors': colors, 'labels': label}

        
        pos = torch.tensor(data['points'], dtype=torch.float)
        intensities = torch.tensor(data['colors'], dtype=torch.float)
        
        y = torch.tensor(data['labels'],dtype=torch.long)
        x = torch.ones((points.shape[0],1), dtype=torch.float)

        
        return Data(x=x, intensities=intensities, pos=pos, y=y, 
                    shape_id=idx, )

    def load_label_poss(self, label_path):
        label = np.fromfile(label_path, dtype=np.uint32)
        label = label.reshape((-1))
        sem_label = label & 0xFFFF  # semantic label in lower half
        inst_label = label >> 16  # instance id in upper half
        assert ((sem_label + (inst_label << 16) == label).all())
        sem_label = self.remap_lut_val[sem_label]
        return sem_label.astype(np.int32)

    @staticmethod
    def get_mask_filter_valid_labels(y):
        return (y>0)

    @property
    def raw_file_names(self):
        return []

    def _download(self): # override _download to remove makedirs
        pass

    def download(self):
        pass

    def process(self):
        pass

    def _process(self):
        pass

    def len(self):
        return len(self.all_files)

    def get_category(self, f_id):
        return str(self.all_files[f_id]).split("/")[-3]

    def get_object_name(self, f_id):
        return str(self.all_files[f_id]).split("/")[-1]

    def get_class_name(self, f_id):
        return "lidar"

    def get_save_dir(self, f_id):
        return os.path.join(str(self.all_files[f_id]).split("/")[-3], str(self.all_files[f_id]).split("/")[-2])