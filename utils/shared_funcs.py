#Shared functions between the different networks
import logging
import os
from functools import partial

import numpy as np
import scipy
import torch
import yaml
from scipy.stats import entropy
from sklearn.metrics import confusion_matrix

import networks
import utils.metrics as metrics
from utils.collate_function import collate_function
from utils.logging_files_functions import (train_log_data_da,
                                           val_log_data_da)
from utils.transforms import da_get_inputs, da_get_transforms
from utils.utils import validation
from utils.lcp import logs_file


def da_get_dataloader(source_DatasetClass, target_DatasetClass, config, net, train_shuffle=True, cat_list_changed=[]):
    source_train_transforms = da_get_transforms(config, train=True, source_flag=True)
    source_test_transforms = da_get_transforms(config, train=False, source_flag=True)
    target_train_transforms = da_get_transforms(config, train=True, source_flag=False)
    target_test_transforms = da_get_transforms(config, train=False, source_flag=False)

    # build the Train datasets for target and source
    source_train_dataset=source_DatasetClass(config["source_dataset_root"],
                split=config["train_split"], transform=source_train_transforms,
                da_flag=True, config=config)
    target_train_dataset=target_DatasetClass(config["target_dataset_root"],
               split=config["train_split"], transform=target_train_transforms,
               da_flag=True, config=config)

    # Loading the test datasets
    source_test_dataset=source_DatasetClass(config["source_dataset_root"],
                split=config["val_split"], transform=source_test_transforms,
                da_flag=True, config=config)
    target_test_dataset=target_DatasetClass(config["target_dataset_root"],
            split=config["val_split"], transform=target_test_transforms,
            da_flag=True, config=config)

    # create the collate function
    if len(cat_list_changed) > 0:
        cat_item_list = cat_list_changed + net.get_cat_item_list()
    else: 
        cat_item_list = ["pos", "x", "y", "dirs", "pos_non_manifold", "occupancies"] + net.get_cat_item_list()

    stack_item_list = [] + net.get_stack_item_list()
    sparse_item_list = ["sparse_input"]

    collate_fn = partial(collate_function, 
                cat_item_list=cat_item_list,
                stack_item_list=stack_item_list,
                sparse_item_list=sparse_item_list
                )

    # build the data loaders for SOURCE
    source_train_loader = torch.utils.data.DataLoader(
        source_train_dataset, batch_size=config["training_batch_size"],
        shuffle=train_shuffle, pin_memory=True,
        num_workers=config["threads"], collate_fn=collate_fn,
        )

    source_test_loader = torch.utils.data.DataLoader(
        source_test_dataset, batch_size=config["test_batch_size"],
        shuffle=False, num_workers=config["threads"],
        pin_memory=True, collate_fn=collate_fn,
        )

    # build the data loaders for TARGET
    target_train_loader = torch.utils.data.DataLoader(
        target_train_dataset,
        batch_size=config["training_batch_size"],
        shuffle=train_shuffle,
        pin_memory=True,
        num_workers=config["threads"],
        collate_fn=collate_fn,
        )

    target_test_loader = torch.utils.data.DataLoader(
        target_test_dataset,
        batch_size=config["test_batch_size"],
        shuffle=False,
        pin_memory=True,
        num_workers=config["threads"],
        collate_fn=collate_fn,
        )
    return {"source_train_loader": source_train_loader, "source_test_loader": source_test_loader, "target_train_loader": target_train_loader, "target_test_loader": target_test_loader}


def optimizer_selection(logging, config, net, net_parameters=None):
    if net_parameters is None: 
        # Train the complete network, otherwise only the parameters in net_parameters are trained
        net_parameters = net.parameters()
    if config["optimizer"] == "AdamW":
        logging.info(f"Selected optimizer: {config['optimizer']}, LR(start): {config['training_lr_start']}")
        optimizer = torch.optim.AdamW(net_parameters,config["training_lr_start"])
    elif config["optimizer"] == "Adam":
        betas = (0.9, 0.99)
        logging.info(f"Selected optimizer: {config['optimizer']}, LR(start): {config['training_lr_start']}, betas: {betas}")
        optimizer = torch.optim.Adam(net_parameters, lr=config["training_lr_start"], betas=betas)
    else: 
        raise NotImplementedError

    return optimizer


def learning_rate_scheduler_selection(logging, config, optimizer):
    if config['lr_scheduler'] == "cos_an_half_lr":
        # Cosine annealing
        T_max = config['training_iter_nbr']
        eta_min = 0
        logging.info(f"Cosine annealing LR rate scheduling with only half period T_max: {T_max}, eta_min: {eta_min}")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    else: 
        raise NotImplementedError

    return scheduler


def validation_process_training(net, config, source_test_loader, target_test_loader, N_LABELS, epoch, disable_log, device, ce_loss_layer, loss_layer, list_ignore_classes, logging):
    # Validation on SOURCE
    val_data_src = \
        validation(net, config, source_test_loader, N_LABELS, epoch, disable_log, device, ce_loss_layer, loss_layer, target_flag=False, list_ignore_classes=list_ignore_classes)
    logging.info("Source mIoU per class: {}".format(val_data_src["seg_iou_per_class"]))
    
    # Validation on TARGET
    val_data_trg = \
        validation(net, config, target_test_loader, N_LABELS, epoch, disable_log, device, ce_loss_layer, loss_layer, target_flag=False, list_ignore_classes=list_ignore_classes)
    logging.info("Target per class mIoU: {}".format(val_data_trg["seg_iou_per_class"]))
    return val_data_src, val_data_trg


def construct_network(config, logging):
        latent_size = config["network_latent_size"]
        backbone = config["network_backbone"]
        decoder = {'name':config["network_decoder"], 'k': config['network_decoder_k']}
        logging.info("Backbone {}".format(backbone))
        logging.info("Decoder {}".format(decoder))
        in_channels_source, _, in_channels_target, _ = da_get_inputs(config)
        logging.info("In channels source: {}".format(in_channels_source))
        logging.info("in channels target: {}".format(in_channels_target))
        logging.info("Creating the network:")
        def network_function():
            return networks.Network(in_channels_source, latent_size, config["network_n_labels"], backbone, decoder, 
                        voxel_size=config["voxel_size"],
                        dual_seg_head=config["dual_seg_head"],
                        nb_classes=config["nb_classes"],
                        da_flag=False, target_in_channels=in_channels_target, config=config)
        net = network_function()
        return net, network_function


def get_savedir_root(config, experiment_name):
    savedir = f"{experiment_name}_{config['network_backbone']}_{config['network_decoder']}_{config['network_decoder_k']}"
    savedir += f"_{config['train_split']}Split"
    if ("desc" in config) and config["desc"]:
        savedir += f"_{config['desc']}"

    savedir_root = os.path.join(config["save_dir"], savedir)

    return savedir_root


def get_bbdir_root(config):
    # Gives the backbone dir 
    savedir_root = config["ckpt_path_model"]
    return savedir_root


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_config_file(config, filename):
    with open(filename, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def ignore_selection(idx=-1):
    if idx == -1:
        return []
    elif idx == 0:
        return [0]


class metrics_holder():
    def __init__(self, N_LABELS, config, target_flag): 
        self.cm = np.zeros((N_LABELS, N_LABELS))  # Used for occupancy
        self.cm_occ_seg = np.zeros((config["nb_classes"], config["nb_classes"]))
        self.error = 0
        self.error_seg_head = 0
        self.error_recons = 0
        self.error_additional = 0
        self.cm_seg_head = np.zeros((config["nb_classes"], config["nb_classes"]))
        self.config = config
        self.target = target_flag
        self.counter = 0
        self.entropy_average_sum = 0
        self.information_maximation_sum = 0



def resume_model(net, savedir_root, device, optimizer, scheduler, source_train_loader):
    checkpoint = torch.load(os.path.join(savedir_root, "checkpoint.pth"), map_location=device)
    net.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    epoch_start = checkpoint["epoch"]
    train_iter_count = len(source_train_loader) * epoch_start
    logging.info(f"Resume on epoch {epoch_start} with iteration count {train_iter_count}")
    current_lr = optimizer.param_groups[0]["lr"]
    logging.info(f"LR is at {current_lr} ")
    try: 
        best_checkpoint = torch.load(os.path.join(savedir_root, "best_checkpoint.pth"))
    except: 
        logging.info(f"!! No best checkpoint found, go on without it")
        best_checkpoint=None
    return net, optimizer, scheduler, epoch_start, train_iter_count, current_lr, best_checkpoint


def save_val_model(config, data_saver):
    metrics = data_saver["metrics"]
    metrics_target = data_saver["metrics_target"]
    train_iter_count = data_saver["train_iter_count"]
    _run = data_saver["_run"]
    writer = data_saver["writer"]
    epoch = data_saver["epoch"]
    net = data_saver["net"]
    source_test_loader =data_saver["source_test_loader"]
    target_test_loader =data_saver["target_test_loader"]
    N_LABELS = data_saver["N_LABELS"]
    disable_log =data_saver["disable_log"]
    disable_log =data_saver["disable_log"]
    ce_loss_layer =data_saver["ce_loss_layer"]
    loss_layer =data_saver["loss_layer"]
    list_ignore_classes =data_saver["list_ignore_classes"]
    list_ignore_classes = data_saver["list_ignore_classes"]
    device = data_saver["device"]
    optimizer = data_saver["optimizer"]
    scheduler = data_saver["scheduler"]
    savedir_root = data_saver["savedir_root"]
    best_ckpt_mioU_target = data_saver["best_ckpt_mioU_target"]
    best_ckpt_epoch = data_saver["best_ckpt_epoch"]
    
    
    ###################################### Validation during Training ####################################################
    if (epoch+1)%config["val_interval"]==0 or train_iter_count >= config["training_iter_nbr"]:
        
        #Validation
        val_data_src, val_data_trg= validation_process_training(net, config, source_test_loader,\
                target_test_loader, N_LABELS, epoch, disable_log, device, ce_loss_layer, loss_layer, list_ignore_classes, logging)
            
        ######################################
        # save the validation logs
        val_log_data = val_log_data_da(val_data_src, val_data_trg, train_iter_count, _run, writer, config=config)
        logs_file(os.path.join(savedir_root, "logs_val.csv"), train_iter_count, val_log_data)

        # Save only the ckpts in the val generation duration
        if (epoch+1)%config["val_interval"]==0 or (train_iter_count >= config["training_iter_nbr"]):
            torch.save({"epoch": epoch + 1, "state_dict": net.state_dict(), "optimizer": optimizer.state_dict(),"scheduler":scheduler.state_dict(),},
            os.path.join(savedir_root, f"checkpoint_{epoch}_epoch.pth"),)

        if val_log_data["trg_validation.seg_mIoU"] > best_ckpt_mioU_target:
            best_ckpt_mioU_target = val_log_data["trg_validation.seg_mIoU"]
            best_ckpt_epoch = epoch
            logging.info("New best target mIou: {}".format(best_ckpt_mioU_target))
            torch.save({"epoch": epoch + 1, "state_dict": net.state_dict(),"optimizer": optimizer.state_dict(),"scheduler":scheduler.state_dict(), "best_mIoU":best_ckpt_mioU_target},
            os.path.join(savedir_root, f"best_checkpoint.pth"),)
        else:
            logging.info("No new best target mIou, best remains: {} from epoch {}".format(best_ckpt_mioU_target, best_ckpt_epoch))
    
    return best_ckpt_mioU_target, best_ckpt_epoch

def calculation_metrics(metrics_holder, outputs, occupancies, loss_seg, loss, recons_loss, output_seg=None, source_data=None, ignore_list=[], output_data=None):
    output_np = (torch.sigmoid(outputs).cpu().detach().numpy() > 0.5).astype(int)
    target_np = occupancies.cpu().numpy().astype(int)
    
    metrics_holder.counter +=1
    #### Semantic Segmentation
    output_seg_np = np.argmax(output_seg[:,1:].cpu().detach().numpy(), axis=1) + 1 #As the 0 dimension (class ignore) should not be taken into account for finding the max value
    target_seg_np = source_data["y"].cpu().numpy().astype(int)

    cm_seg_head_ = confusion_matrix(target_seg_np.ravel(), output_seg_np.ravel(), labels=list(range(metrics_holder.config["nb_classes"])))
    metrics_holder.cm_seg_head += cm_seg_head_
    train_seg_head_oa = metrics.stats_overall_accuracy(metrics_holder.cm_seg_head, ignore_list=ignore_list)
    train_seg_head_maa, train_acc_per_class = metrics.stats_accuracy_per_class(metrics_holder.cm_seg_head, ignore_list=ignore_list) #First return value is the mean IoU
    train_seg_head_miou, train_seg_iou_per_class = metrics.stats_iou_per_class(metrics_holder.cm_seg_head, ignore_list=ignore_list) #First return value is the mean IoU
    if not(loss_seg is None):
        metrics_holder.error_seg_head += loss_seg.item()
    # point wise scores on training segmentation head
    train_seg_head_loss = metrics_holder.error_seg_head / metrics_holder.cm_seg_head.sum()
    
    
    #### Occupany
    cm_ = confusion_matrix(
        target_np.ravel(), output_np.ravel(), labels=list(range(metrics_holder.cm.shape[0]))
    )
    metrics_holder.cm += cm_

    metrics_holder.error += loss.item()
    metrics_holder.error_recons += recons_loss.item()

    # point wise scores on training
    train_oa = metrics.stats_overall_accuracy(metrics_holder.cm)
    train_aa = metrics.stats_accuracy_per_class(metrics_holder.cm)[0]
    train_iou = metrics.stats_iou_per_class(metrics_holder.cm)[0]
    train_aloss = metrics_holder.error / metrics_holder.cm.sum()
    train_aloss_recons = metrics_holder.error_recons / metrics_holder.cm.sum()
    train_aloss_additional = metrics_holder.error_additional / metrics_holder.cm.sum()

    
    return_data =  {"train_oa":train_oa, "train_aa":train_aa, "train_aloss":train_aloss,"train_aloss_recons":train_aloss_recons, "train_aloss_additional":train_aloss_additional,\
        "train_iou":train_iou, "train_seg_head_miou":train_seg_head_miou,"train_seg_head_maa":train_seg_head_maa, "train_seg_head_loss": train_seg_head_loss,\
    "accuracy_per_class": train_acc_per_class, "seg_iou_per_class": train_seg_iou_per_class}
    return return_data

    