
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset

import logging

import numpy as np
from nuscenes import NuScenes as NuScenes_
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.data_io import load_bin_file
from nuscenes.utils.splits import create_splits_scenes



def class_mapping_ns():
    return {0: 'noise',
            1: 'animal',
            2: 'human.pedestrian.adult',
            3: 'human.pedestrian.child',
            4: 'human.pedestrian.construction_worker',
            5: 'human.pedestrian.personal_mobility',
            6: 'human.pedestrian.police_officer',
            7: 'human.pedestrian.stroller',
            8: 'human.pedestrian.wheelchair',
            9: 'movable_object.barrier',
            10: 'movable_object.debris',
            11: 'movable_object.pushable_pullable',
            12: 'movable_object.trafficcone',
            13: 'static_object.bicycle_rack',
            14: 'vehicle.bicycle',
            15: 'vehicle.bus.bendy',
            16: 'vehicle.bus.rigid',
            17: 'vehicle.car',
            18: 'vehicle.construction',
            19: 'vehicle.emergency.ambulance',
            20: 'vehicle.emergency.police',
            21: 'vehicle.motorcycle',
            22: 'vehicle.trailer',
            23: 'vehicle.truck',
            24: 'flat.driveable_surface',
            25: 'flat.other',
            26: 'flat.sidewalk',
            27: 'flat.terrain',
            28: 'static.manmade',
            29: 'static.other',
            30: 'static.vegetation',
            31: 'vehicle.ego'
            }
def class_mapping_da(config):
    #Adaption of mapping of classes in DA Case
            if config["source_dataset_name"] == "SemanticKITTI" or  config["target_dataset_name"] == "SemanticKITTI" \
            or (config["source_dataset_name"] == "NuScenes" and  config["target_dataset_name"] == "NuScenes"):
                # Official class mappings from Complet&Label (CVPR 2021)
                return {
                0: 0,
                1: 0,
                2: 6,
                3: 6,
                4: 6,
                5: 6, 
                6: 6,
                7: 0,
                8: 0,
                9: 0,
                10: 0,
                11: 0,
                12: 0,
                13: 0,
                14: 2,
                15: 5,
                16: 5,
                17: 1,
                18: 5,
                19: 5, 
                20: 5, 
                21: 3,
                22: 5,
                23: 4,
                24: 7,
                25: 0,
                26: 8,
                27: 9,
                28: 0,
                29: 0,
                30: 10,
                31: 0,
                }
            else: 
                raise ValueError("No mapping")

class NuScenes(Dataset):

    N_LABELS=11

    def __init__(self, root, split="training", transform=None,
                 da_flag = False, config=None, **kwargs):

        super().__init__(root, transform, None)

        self.config = config
        #self.nusc = NuScenes_(version=self.config['ns_dataset_version'], dataroot=self.root, verbose=True)
        self.nusc = NuScenes_(version='v1.0-mini', dataroot=self.root, verbose=True)

        self.da_flag = da_flag
        logging.info("Nuscenes dataset - creating splits")

        # get the scenes
        assert(split in ["train", "val", "test"])
        phase_scenes = create_splits_scenes()[split]

        # create a list of camera & lidar scans
        self.list_keyframes = []
        for scene_idx in range(len(self.nusc.scene)):
            scene = self.nusc.scene[scene_idx]
            if scene["name"] in phase_scenes:
                current_sample_token = scene["first_sample_token"]

                # Loop to get all successive keyframes
                list_data = []
                while current_sample_token != "":
                    current_sample = self.nusc.get("sample", current_sample_token)
                    list_data.append(current_sample["data"])
                    current_sample_token = current_sample["next"]

                # Add new scans in the list
                self.list_keyframes.extend(list_data)

        

        print(f"Nuscnes dataset split {split} - {len(self.list_keyframes)} frames")

        self.label_to_name = class_mapping_ns()
        
        if self.da_flag: 
            self.label_to_reduced = class_mapping_da(self.config)
        
        self.label_to_reduced_np = np.zeros(32, dtype=np.int)
        for i in range(32):
            self.label_to_reduced_np[i] = self.label_to_reduced[i]
        

    def get_weights(self):
        weights = torch.ones(self.N_LABELS)
        weights[0] = 0
        return weights

    @staticmethod
    def get_mask_filter_valid_labels(y):
        return (y>0)

    @staticmethod
    def get_ignore_index():
        return 0

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
        return len(self.list_keyframes)

    def get_category(self, f_id):
        raise NotImplementedError

    def get_object_name(self, f_id):
        raise NotImplementedError

    def get_class_name(self, f_id):
        raise NotImplementedError

    def get_save_dir(self, f_id):
        raise NotImplementedError


    def get(self, idx):
        """Get item."""

        data = self.list_keyframes[idx]

        # get the lidar
        lidar_token = data['LIDAR_TOP']
        lidar_rec = self.nusc.get('sample_data', lidar_token)
        pc = LidarPointCloud.from_file(os.path.join(self.nusc.dataroot, lidar_rec['filename']))
        pc = pc.points.T

        pos = pc[:,:3]
        intensities = pc[:,3:] / 255 # intensities are not used

        # get the labels
        lidarseg_label_filename = os.path.join(self.nusc.dataroot, self.nusc.get('lidarseg', lidar_token)['filename'])
        y_complete_labels = load_bin_file(lidarseg_label_filename)

        y = self.label_to_reduced_np[y_complete_labels]
        
        # convert to torch
        pos = torch.tensor(pos, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        intensities = torch.tensor(intensities, dtype=torch.float)
        x = torch.ones((pos.shape[0],1), dtype=torch.float)

        return Data(x=x, intensities=intensities, pos=pos, y=y, shape_id=idx)