import logging
import os

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

import datasets
import networks
from config import ex
from utils.logging_files_functions import val_log_data_da
from utils.shared_funcs import (count_parameters, da_get_dataloader,
                                get_bbdir_root, get_savedir_root,
                                ignore_selection, logs_file, save_config_file)
from utils.transforms import da_get_inputs
from utils.utils import get_bbdir_root, get_savedir_root, validation


def load_student_model(checkpoint_path):
    # reloads model
    def clean_state_dict(state):
        # clean state dict from names of PL
        for k in list(ckpt.keys()):
            if "student_model" in k:
                if "dual_seg_head" in k:
                    ckpt[k.replace("student_model.", "")] = ckpt[k]
                else: 
                    ckpt[k.replace("student_model.", "net.")] = ckpt[k]
            del ckpt[k]
        return state

    ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))["state_dict"]
    print(ckpt.keys())
    ckpt = clean_state_dict(ckpt)

    
    return ckpt

def main_run(_config, _run):

    # experiment_name
    experiment_name = _run.experiment_info["name"]

    # writer for tensorboard
    writer = SummaryWriter('runs/{}'.format(experiment_name))

    # convert config to dict
    config = eval(str(_config))

    # device
    device = torch.device(config['device'])
    if config["device"] == "cuda":
        torch.backends.cudnn.benchmark = True

    # define the logging
    logging.getLogger().setLevel(config["logging"])

    # get the savedir
    savedir_root = get_savedir_root(config, experiment_name)
    bb_dir_root = get_bbdir_root(config)
    # create the network
    disable_log = not config["interactive_log"]
    N_LABELS = 2  # For occupancy
    latent_size = config["network_latent_size"]
    backbone = config["network_backbone"]
    decoder = {'name': config["network_decoder"], 'k': config['network_decoder_k']}

    in_channels_source, _, in_channels_target, _ = da_get_inputs(config)
    logging.info("Creating the network")

    def network_function():
        return networks.Network(in_channels_source, latent_size, config["network_n_labels"], backbone, decoder,
                    voxel_size=config["voxel_size"], dual_seg_head=config["dual_seg_head"],
                    nb_classes=config["nb_classes"], da_flag=False, target_in_channels=in_channels_target, config=config)
    
    net = network_function()
    net.to(device)
    logging.info(f"Network -- Number of parameters {count_parameters(net)}")
    datasets_prefix = "datasets."
    source_DatasetClass = datasets.get_dataset(eval(datasets_prefix+config["source_dataset_name"]))
    target_DatasetClass = datasets.get_dataset(eval(datasets_prefix+config["target_dataset_name"]))
    if config['test_flag_eval']:
        val_number = -1
    else:
        val_number = 1  # 1: verifying split, 2 train split, else: test split
    dataloader_dict = da_get_dataloader(source_DatasetClass, target_DatasetClass, config, net)
    source_train_loader = dataloader_dict ["source_train_loader"]
    source_test_loader = dataloader_dict ["source_test_loader"]
    target_train_loader = dataloader_dict ["target_train_loader"]
    target_test_loader = dataloader_dict ["target_test_loader"]

    os.makedirs(savedir_root, exist_ok=True)
    save_config_file(eval(str(config)), os.path.join(savedir_root, "config.yaml"))
    epoch_start = 0
    train_iter_count = 0


    ckpt_path = os.path.join(bb_dir_root, 'checkpoint.pth')
    prefix_save_name = "last_ckpt_"

    logging.info(f"CKPT -- Load ckpt from {ckpt_path}")

    # Load the checkpoint for the backbone

    if "cosmix_backbone" in config and config["cosmix_backbone"]:
        # For models refined with CoSMix
        checkpoint = load_student_model(os.path.join(bb_dir_root, "checkpoint.pth"))
    else:
        checkpoint = torch.load(os.path.join(bb_dir_root, "checkpoint.pth"), map_location=device)
        checkpoint = checkpoint["state_dict"]
        


    try:
        net.load_state_dict(checkpoint)
    except Exception as e:
        logging.info(e)
        logging.info(f"Loaded parameters do not match exactly net architecture, switching to load_state_dict strict=false")
        net.load_state_dict(checkpoint, strict=False)

    epoch_start = 0
    os.makedirs(savedir_root, exist_ok=True)
    save_config_file(eval(str(config)), os.path.join(savedir_root, "config.yaml"))
    epoch_start = 0
    train_iter_count = 0

    # create the loss layer
    loss_layer = torch.nn.BCEWithLogitsLoss()
    weights_ss = torch.ones(config["nb_classes"])
    list_ignore_classes = ignore_selection(config["ignore_idx"])
    for idx_ignore_class in list_ignore_classes: 
        weights_ss[idx_ignore_class] = 0
    logging.info(f"Ignored classes {list_ignore_classes}")
    logging.info(f"Weights of the different classes {weights_ss}")
    weights_ss=weights_ss.to(device)
    ce_loss_layer = torch.nn.CrossEntropyLoss(weight=weights_ss)
    epoch = epoch_start

    # Validation on SOURCE
    return_data_val_source = \
        validation(net, config, source_test_loader, N_LABELS, epoch, disable_log, device, ce_loss_layer, loss_layer, target_flag=False, list_ignore_classes=list_ignore_classes)
    logging.info("Source mIoU per class: {}".format(return_data_val_source["seg_iou_per_class"]))

    # Validation on TARGET
    return_data_val_target = \
        validation(net, config, target_test_loader, N_LABELS, epoch, disable_log, device, ce_loss_layer, loss_layer, target_flag=False, list_ignore_classes=list_ignore_classes)
    logging.info("Target per class mIoU: {}".format(return_data_val_target["seg_iou_per_class"]))
    
    # save the logs
    val_log_data = val_log_data_da(val_data_src=return_data_val_source, val_data_trg=return_data_val_target, train_iter_count=train_iter_count, _run= _run, writer=writer, config=config, long_prefix="validation", short_prefix="val")
    logs_file(os.path.join(savedir_root, prefix_save_name+"logs_val.csv"), train_iter_count, val_log_data)
    df_cm_source = pd.DataFrame(data=return_data_val_source["cm_seg_head"].astype(float))
    df_cm_source.to_csv(os.path.join(savedir_root, prefix_save_name+"validation_cm_source.csv"), sep=' ', header=False, float_format='%.2f', index=False)
    df_cm_target = pd.DataFrame(data=return_data_val_target["cm_seg_head"].astype(float))
    df_cm_target.to_csv(os.path.join(savedir_root, prefix_save_name+"validation_cm_target.csv"), sep=' ', header=False, float_format='%.2f', index=False)

    logging.info("Save the results with prefix: {}".format(prefix_save_name))

@ex.automain
def main(_config, _run):
    # Run the validation for the last epoch (or the one selected)
    main_run(_config, _run)