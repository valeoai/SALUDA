import os
#from cv2 import add
from matplotlib.pyplot import fill
import numpy as np
import yaml
from tqdm import tqdm
import logging
import shutil
import torch_geometric.transforms as T

# torch imports
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# lightconvpoint imports
#from lightconvpoint.datasets.dataset import get_dataset
#import lightconvpoint.utils.transforms as lcp_T
#from lightconvpoint.utils.logs import logs_file
#from lightconvpoint.utils.misc import dict_to_device

from utils.utils import wblue, wgreen
import utils.metrics as metrics

from utils.transforms import da_get_inputs, da_get_transforms
from utils.utils import get_savedir_root, get_bbdir_root
from utils.logging_files_functions import val_log_data_da, train_log_data_da
from utils.utils import validation
from utils.shared_funcs import (get_savedir_root, get_bbdir_root, collate_function,
                                count_parameters, save_config_file, metrics_holder,
                                da_get_dataloader, ignore_selection, optimizer_selection,
                                learning_rate_scheduler_selection, validation_process_training,
                                construct_network, calculation_metrics)

from utils.shared_funcs import resume_model, save_val_model
from utils.transforms import CreateDirs, CreateInputs, CreateNonManifoldPoints, UseAsFeatures, Transpose
import datasets
import networks
#import argparse

#from datetime import datetime
#from functools import partial
