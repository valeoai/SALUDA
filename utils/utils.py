import os

import numpy as np
import scipy
import torch
from scipy.stats import entropy
from sklearn.metrics import confusion_matrix
#from lightconvpoint.utils.misc import dict_to_device
from torchsparse import SparseTensor
from tqdm import tqdm

import utils.metrics as metrics


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# wrap blue / green
def wblue(str):
    return bcolors.OKBLUE+str+bcolors.ENDC


def wgreen(str):
    return bcolors.OKGREEN+str+bcolors.ENDC


def wred(str):
    return bcolors.FAIL+str+bcolors.ENDC

def dict_to_device(data, device):
    for key, value in data.items():
        if torch.is_tensor(value):
            data[key] = value.to(device)
        elif isinstance(value, list):
            data[key] = list_to_device(value, device)
        elif isinstance(value, dict):
            data[key] = dict_to_device(value, device)
        elif isinstance(value, SparseTensor):
            data[key] = data[key].to(device)
    return data


def get_savedir_root(config, experiment_name):
    savedir = f"{experiment_name}_{config['network_decoder']}_{config['network_decoder_k']}"
    savedir += f"_{config['train_split']}Split"
    if ("desc" in config) and config["desc"]:
        savedir += f"_{config['desc']}"
    print(savedir)
    savedir_root = os.path.join(config["save_dir"], savedir)

    return savedir_root


def get_bbdir_root(config):
    # Gives the backbone dir 
    savedir_root = config["ckpt_path_model"]
    return savedir_root


def validation(net, config, test_loader, N_LABELS, epoch, disable_log, device, ce_loss_layer, loss_layer, target_flag=False, list_ignore_classes=[]):
    net.eval()
    error = 0
    error_recons = 0
    error_additional = 0
    error_seg_head = 0
    cm = np.zeros((N_LABELS, N_LABELS))
    cm_seg_head = np.zeros((config["nb_classes"],config["nb_classes"]))
    
    with torch.no_grad():
        count_iter = 0
        t = tqdm(
            test_loader,
            desc="  Test " + str(epoch),
            ncols=200,
            disable=disable_log,
        )
        for data in t:

            data = dict_to_device(data, device)                    
            output_data, output_seg = net.forward_pretraining(data, get_latent=config["get_latent"])     
            loss_seg = ce_loss_layer(output_seg, data["y"][:, None])   

            
            outputs = output_data["predictions"].squeeze(-1)
            occupancies = output_data["occupancies"].float()
            recons_loss = loss_layer(outputs, occupancies)
            loss = recons_loss
            if "loss" in output_data:
                additionnal_loss = output_data["loss"]
                loss = loss + additionnal_loss
            else:
                additionnal_loss = torch.zeros((1,))

            output_np = (torch.sigmoid(outputs).cpu().detach().numpy() > 0.5).astype(int)
            target_np = occupancies.cpu().numpy().astype(int)
            cm_ = confusion_matrix(
                target_np.ravel(), output_np.ravel(), labels=list(range(N_LABELS))
            )
            cm += cm_

            error += loss.item()
            error_additional += additionnal_loss.item()
            error_recons += recons_loss.item()

            # point-wise scores on testing
            test_oa = metrics.stats_overall_accuracy(cm)
            test_aa, _ = metrics.stats_accuracy_per_class(cm)
            test_iou, iou_per_class = metrics.stats_iou_per_class(cm)
            test_aloss = error / cm.sum()
            test_aloss_recons = error_recons / cm.sum()
            test_aloss_additional = error_additional / cm.sum()

            
            output_seg_np = np.argmax(output_seg[:,1:].cpu().detach().numpy(), axis=1) + 1 # As 0 is ignored, only looked in the non 0 part
            target_seg_np = data["y"].cpu().numpy().astype(int)
            cm_seg_head_ = confusion_matrix(target_seg_np.ravel(), output_seg_np.ravel(), labels=list(range(config["nb_classes"])))
            cm_seg_head += cm_seg_head_
            
            error_seg_head += loss_seg.item()
            # point wise scores on training segmentation head
            test_seg_head_maa, accuracy_per_class = metrics.stats_accuracy_per_class(cm_seg_head, ignore_list=list_ignore_classes)
            test_seg_head_miou, seg_iou_per_class = metrics.stats_iou_per_class(cm_seg_head, ignore_list=list_ignore_classes)
            test_seg_head_loss = error_seg_head / cm_seg_head.sum()

            count_iter += 1
            if count_iter % 10 == 0:
                torch.cuda.empty_cache()

            description = f"Epoch {epoch}|| TARGET {target_flag}|| Rec-IoU {test_iou*100:.2f} | Rec-Loss {test_aloss:.3e} | LossR {test_aloss_recons:.3e} | LossA {test_aloss_additional:.3e} | Seg-IoU {test_seg_head_miou*100:.2f}| Seg-AA {test_seg_head_maa*100:.2f} | Seg-Loss {test_seg_head_loss:.3e}"

            t.set_description_str(wgreen(description))

    
    return_data = {"test_oa": test_oa, "test_aa": test_aa, "test_iou": test_iou, "test_aloss": test_aloss, "test_aloss_recons": test_aloss_recons,
        "test_aloss_additional": test_aloss_additional, "test_seg_head_miou": test_seg_head_miou, "test_seg_head_maa": test_seg_head_maa,
        "test_seg_head_loss": test_seg_head_loss, "seg_iou_per_class": seg_iou_per_class, "accuracy_per_class": accuracy_per_class,
        "cm_seg_head": cm_seg_head}
    return return_data