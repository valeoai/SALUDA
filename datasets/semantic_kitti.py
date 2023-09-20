import importlib
import os

import torch
import torch.nn.functional as F
from torch._C import Value
from torch_geometric.data import Data, Dataset

import logging
from pathlib import Path

# Basic libs
import numpy as np
import yaml


class SemanticKITTI(Dataset):
    def __init__(self, root, split="training", transform=None, dataset_size=None,
     multiframe_range=None, da_flag =False, config=None, **kwargs):

        super().__init__(root, transform, None)

        self.split = split
        self.n_frames = 1
        self.da_flag = da_flag
        self.config = config
        self.multiframe_range = multiframe_range
        self.N_LABELS = self.config["nb_classes"] if self.config is not None else 11

        
        logging.info(f"SemanticKITTI - split {split}")

        # get the scenes
        assert(split in ["train", "val", "test"])
        
        if split == "train":
            self.sequences = ['{:02d}'.format(i) for i in range(11) if i != 8]
        elif split == "val":
            self.sequences = ['{:02d}'.format(i) for i in range(11) if i == 8]
        elif split == "test":
            self.sequences = ['{:02d}'.format(i) for i in range(11, 22)]
        else:
            raise ValueError('Unknown set for SemanticKitti data: ', split)

        # get the filenames
        self.all_files = []
        for sequence in self.sequences:
            self.all_files += [path for path in Path(os.path.join(self.root, "dataset", "sequences", sequence, "velodyne")).rglob('*.bin')]
        
        # Sort for verifying and parametrizing 
        if split == "verifying" or split == "val" or split == "parametrizing": 
            self.all_files = sorted(self.all_files, key=lambda i:str(i).lower())

        self.all_labels = []
        for fname in self.all_files:
            fname = str(fname).replace("/velodyne/", "/labels/")
            fname = str(fname).replace(".bin", ".label")
            self.all_labels.append(fname)


        # Read labels
        if self.n_frames == 1:
            config_file = os.path.join(self.root, 'semantic-kitti.yaml')
        elif self.n_frames > 1:
            config_file = os.path.join(self.root, 'semantic-kitti-all.yaml')
        else:
            raise ValueError('number of frames has to be >= 1')

        with open(config_file, 'r') as stream:
            doc = yaml.safe_load(stream)
            all_labels = doc['labels']
            
            if self.da_flag: 
                #Changes of mapping in DA case
                if self.config["source_dataset_name"] == "NuScenes" or  self.config["target_dataset_name"] == "NuScenes":
                    # Original class mapping from Complete&Label paper
                    learning_map={
                        0 : 0,     # "unlabeled"
                        1 : 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
                        10: 1,     # "car"
                        11: 2,    # "bicycle"
                        13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
                        15: 3,     # "motorcycle"
                        16: 0,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
                        18: 4,     # "truck"
                        20: 5,     # "other-vehicle"
                        30: 6,     # "person"
                        31: 0,     # "bicyclist"
                        32: 0,     # "motorcyclist"
                        40: 7,     # "road"
                        44: 7,    # "parking"
                        48: 8,    # "sidewalk"
                        49: 0,    # "other-ground"
                        50: 0,    # "building"
                        51: 0,    # "fence"
                        52: 0,    # "other-structure" mapped to "unlabeled" ------------------mapped
                        60: 7,     # "lane-marking" to "road" ---------------------------------mapped
                        70: 10,    # "vegetation"
                        71: 10,    # "trunk"
                        72: 9,    # "terrain"
                        80: 0,    # "pole"
                        81: 0,    # "traffic-sign"
                        99: 0,    # "other-object" to "unlabeled" ----------------------------mapped
                        252: 1,    # "moving-car" to "car" ------------------------------------mapped
                        253: 0,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
                        254: 6,    # "moving-person" to "person" ------------------------------mapped
                        255: 0,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
                        256: 0,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
                        257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
                        258: 4,    # "moving-truck" to "truck" --------------------------------mapped
                        259: 5    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
                    }
                    learning_map_inv ={ # inverse of previous map
                        0: 0,      # "unlabeled", and others ignored
                        1: 10,     # "car"
                        2: 11,     # "bicycle"
                        3: 15,     # "motorcycle"
                        4: 18,     # "truck"
                        5: 20,     # "other-vehicle"
                        6: 30,     # "person"
                        #: 31,     # "bicyclist" No differentitation to bicycle
                        #8: 32,     # "motorcyclist" No differentiation to motorcycle
                        7: 40,     # "road"
                        #: 44,    # "parking" No differentation to road
                        8: 48,    # "sidewalk"
                        #: 49,    # "other-ground" Ignored
                        #: 50,    # "building"
                        #: 51,    # "fence"
                        10: 70,    # "vegetation"
                        #: 71,    # "trunk"is in vegetation
                        9: 72,    # "terrain"
                        #: 80,    # "pole"
                        #: 81,    # "traffic-sign"
                    
                    }
                elif self.config["source_dataset_name"] == "SynLidar" or  self.config["target_dataset_name"] == "SynLidar":
                    # Mapping with SynLiDAR
                    learning_map = doc['learning_map']
                    learning_map_inv = doc['learning_map_inv']

            else:
                learning_map = doc['learning_map']
                learning_map_inv = doc['learning_map_inv']
            
            self.learning_map = np.zeros((np.max([k for k in learning_map.keys()]) + 1), dtype=np.int32)
            
            for k, v in learning_map.items():
                self.learning_map[k] = v
            
            self.learning_map_inv = np.zeros((np.max([k for k in learning_map_inv.keys()]) + 1), dtype=np.int32)
            for k, v in learning_map_inv.items():
                self.learning_map_inv[k] = v
          
    def get_weights(self):
        weights = torch.ones(self.N_LABELS)
        weights[0] = 0
        return weights

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

    def get_filename(self, idx):
        return self.all_files[idx]

    def get(self, idx):
        """Get item."""

        fname_points = self.all_files[idx]
        frame_points = np.fromfile(fname_points, dtype=np.float32)
        pos = frame_points.reshape((-1, 4))
        intensities = pos[:,3:]
        pos = pos[:,:3]

        if self.split in ["test", "testing"]:
            # Fake labels
            y = np.zeros((pos.shape[0],), dtype=np.int32)
        else:
            # Read labels
            label_file = self.all_labels[idx]
            frame_labels = np.fromfile(label_file, dtype=np.int32)
            y = frame_labels & 0xFFFF  # semantic label in lower half
            y = self.learning_map[y]

        # points are annotated only until 50 m
        mask = np.linalg.norm(pos, axis=1)<50
        pos = pos[mask]
        y = y[mask]
        intensities = intensities[mask]

       
        pos = torch.tensor(pos, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        intensities = torch.tensor(intensities, dtype=torch.float)
        x = torch.ones((pos.shape[0],1), dtype=torch.float)
        return Data(x=x, intensities=intensities, pos=pos, y=y, shape_id=idx)

