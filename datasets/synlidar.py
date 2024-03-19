import logging
import os
from pathlib import Path

# Basic libs
import numpy as np
import torch
import yaml
from torch_geometric.data import Data, Dataset


def class_mapping_da(config):
    #Changes of mapping in DA case
    if config["source_dataset_name"] == "SemanticKITTI" or  config["target_dataset_name"] == "SemanticKITTI":
        learning_map={
            0: 0,  # "unlabeled"
            1: 1,  # "car"
            2: 4,  # "pick-up"
            3: 4,  # "truck"
            4: 5,  # "bus"
            5: 2,  # "bicycle"
            6: 3,  # "motorcycle"
            7: 5,  # "other-vehicle"
            8: 9,  # "road"
            9: 11,  # "sidewalk"
            10: 10,  # "parking"
            11: 12,  # "other-ground"
            12: 6, # "female"
            13: 6,  # "male"
            14: 6,  # "kid"
            15: 6,  # "crowd"
            16: 7,  # "bicyclist"
            17: 8,  # "motorcyclist"
            18: 13,  # "building"
            19: 0,  # "other-structure"
            20: 15,  # "vegetation"
            21: 16,  # "trunk"
            22: 17,  # "terrain"
            23: 19,  # "traffic-sign"
            24: 18,  # "pole"
            25: 0,  # "traffic-cone"
            26: 14,  # "fence"
            27: 0,  # "garbage-can"
            28: 0,  # "electric-box"
            29: 0,  # "table"
            30: 0,  # "chair"
            31: 0,  # "bench"
            32: 0  # "other-object"
        }
        learning_map_inv ={ # inverse of previous map
            0: 0,     
            1: 1,     
            2: 5,    
            3: 6,     
            4: 2,     
            5: 7,     
            6: 12,
            7: 16,  
            8: 17,
            9: 8, 
            10: 10, 
            11: 9,
            12:11,
            13:18,
            14:26,
            15:20,
            16:21,
            17:22,
            18:24,
            19:23,
        
            }
    elif config["source_dataset_name"] == "SemanticPOSS" or  config["target_dataset_name"] == "SemanticPOSS":
        learning_map={0: 0,  # "unlabeled"
                1: 3,  # "car"
            2: 0,  # "pick-up"
            3: 0,  # "truck"
            4: 0,  # "bus"
            5: 12,  # "bicycle"
            6: 0,  # "motorcycle"
            7: 0,  # "other-vehicle"
            8: 13,  # "road"
            9: 0,  # "sidewalk"
            10: 0,  # "parking"
            11: 0,  # "other-ground"
            12: 1,  # "female"
            13: 1,  # "male"
            14: 1,  # "kid"
            15: 1,  # "crowd"
            16: 2,  # "bicyclist"
            17: 2,  # "motorcyclist"
            18: 9,  # "building"
            19: 0,  # "other-structure"
            20: 5,  # "vegetation"
            21: 4,  # "trunk"
            22: 0,  # "terrain"
            23: 6,  # "traffic-sign"
            24: 7,  # "pole"
            25: 10,  # "traffic-cone"
            26: 11,  # "fence"
            27: 8,  # "garbage-can"
            28: 0,  # "electric-box"
            29: 0,  # "table"
            30: 0,  # "chair"
            31: 0,  # "bench"
            32: 0}  # "other-object"
    else: 
        raise ValueError("No mapping")


    return learning_map

class SynLidar(Dataset):
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

        
        logging.info(f"SynLidar - split {split}")

        # get the scenes
        assert(split in ["train", "val", "test", "verifying", "parametrizing"])
        
        if split == "train":
            self.sequences = ['{:02d}'.format(i) for i in range(13)]
        elif split == "val":
            self.sequences = ['{:02d}'.format(i) for i in range(13) if i == 8] # Pseudo-Validation set, as seen during training.
        elif split == "test":
            raise ValueError('Unknown set for SynLidar data: ', split)
        
        # get the filenames
        self.all_files = []
        for sequence in self.sequences:
            self.all_files += [path for path in Path(os.path.join(self.root, "SubDataset", "sequences", sequence, "velodyne")).rglob('*.bin')]
        
        # Sort for verifying and parametrizing 
        if split == "verifying" or split == "val" or split == "parametrizing": 
            self.all_files = sorted(self.all_files, key=lambda i:str(i).lower())
        
        self.all_files = self.all_files[::skip_ratio]

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

        labels_names={0 : "unlabeled",
            1: "car",
            2: "pick-up",
            3: "truck",
            4: "bus",
            5: "bicycle",
            6: "motorcycle",
            7: "other-vehicle",
            8: "road",
            9: "sidewalk",
            10: "parking",
            11: "other-ground",
            12: "female",
            13: "male",
            14: "kid",
            15: "crowd",  # multiple person that are very close
            16: "bicyclist",
            17: "motorcyclist",
            18: "building",
            19: "other-structure",
            20: "vegetation",
            21: "trunk",
            22: "terrain",
            23: "traffic-sign",
            24: "pole",
            25: "traffic-cone",
            26: "fence",
            27: "garbage-can",
            28: "electric-box",
            29: "table",
            30: "chair",
            31: "bench",
            32: "other-object"}


        learning_map = class_mapping_da(config)
        self.learning_map = np.zeros((np.max([k for k in learning_map.keys()]) + 1), dtype=np.int32)
        
        for k, v in learning_map.items():
            self.learning_map[k] = v
  

    def get_weights(self):
        weights = torch.ones(self.config['nb_classes'])
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
            label_path = self.all_labels[idx]
            label = np.fromfile(label_path, dtype=np.uint32)
            label = label.reshape((-1))
            y = self.learning_map[label]

        
        mask = np.linalg.norm(pos, axis=1)<50
        pos = pos[mask]
        y = y[mask]
        intensities = intensities[mask]

        pos = torch.tensor(pos, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        intensities = torch.tensor(intensities, dtype=torch.float)
        x = torch.ones((pos.shape[0],1), dtype=torch.float)

        return Data(x=x, intensities=intensities, pos=pos, y=y, shape_id=idx,)