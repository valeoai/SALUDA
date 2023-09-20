import os

import numpy as np
import scipy
import torch
from scipy.stats import entropy
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import utils.metrics as metrics
#from lightconvpoint.utils.misc import dict_to_device
from torchsparse import SparseTensor

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
    bnm_value_sum = torch.zeros((1))
    entropy_mean_sum = 0
    information_maximization_sum = 0
    torch_softmax = torch.nn.Softmax(dim=1)
    cm = np.zeros((N_LABELS, N_LABELS))
    cm_seg_head = np.zeros((config["nb_classes"],config["nb_classes"]))
    list_data_source = []
    list_data_target = []
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
            if config["dual_seg_head"]:
                if target_flag: 
                    if config["get_latent"]:
                        output_data, output_seg, data = net.forward_test_target(data, get_latent=config["get_latent"]) # TARGET data
                        latents = data["latents"].detach().cpu().numpy()
                        label = data["y"].detach().cpu().numpy()
                        pos = data["pos"].detach().cpu().numpy()
                        if count_iter % 5 == 0: 
                            list_data_target.append(list([pos, label, latents]))
                    else:
                        output_data, output_seg = net.forward_test_target(data, get_latent=config["get_latent"]) # TARGET data
                else:
                    if config["get_latent"]:
                        output_data, output_seg, data = net.forward_pretraining(data, get_latent=config["get_latent"]) # SOURCE data
                        latents = data["latents"].detach().cpu().numpy()
                        label = data["y"].detach().cpu().numpy()
                        pos = data["pos"].detach().cpu().numpy()
                        if count_iter % 5 == 0:
                            list_data_source.append(list([pos, label, latents]))
                    else:
                        output_data, output_seg = net.forward_pretraining(data, get_latent=config["get_latent"]) # SOURCE data

                if config["network_backbone"] == "MinkEng_Res16UNet34C":
                    loss_seg = ce_loss_layer(output_seg, data["y"]) 
                else:
                    loss_seg = ce_loss_layer(output_seg, data["y"][:, None])   

            else:
                output_data = net.forward_pretraining(data)
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

            # Batch Norm calculation
            bnm_value_sum = bnm_value_sum + torch.norm(torch_softmax(output_seg[:, :, 0]),p='nuc').detach().cpu()/(output_seg.size()[0])


            # Entropy and Information maximization calculation
            softmaxed_value = scipy.special.softmax(output_seg.cpu().detach().numpy(), axis=1)
            entropy_calculated = entropy(softmaxed_value, axis=1)
            entropy_mean = np.average(entropy_calculated, axis=0)
            entropy_on_average_prob = entropy(np.average(softmaxed_value, axis=0))
            information_maximation = entropy_on_average_prob - entropy_mean

            entropy_mean_sum = entropy_mean_sum + entropy_mean
            information_maximization_sum = information_maximization_sum + information_maximation
            
            if config["dual_seg_head"]:
                output_seg_np = np.argmax(output_seg[:,1:].cpu().detach().numpy(), axis=1) + 1 # As 0 is ignored, only looked in the non 0 part
                target_seg_np = data["y"].cpu().numpy().astype(int)
                cm_seg_head_ = confusion_matrix(target_seg_np.ravel(), output_seg_np.ravel(), labels=list(range(config["nb_classes"])))
                cm_seg_head += cm_seg_head_
                if config["network_backbone"] == "MinkEng_Res16UNet34C":
                    error_seg_head += loss_seg.item()
                else:
                    error_seg_head += loss_seg.item()
                # point wise scores on training segmentation head
                test_seg_head_maa, accuracy_per_class = metrics.stats_accuracy_per_class(cm_seg_head, ignore_list=list_ignore_classes)
                test_seg_head_miou, seg_iou_per_class = metrics.stats_iou_per_class(cm_seg_head, ignore_list=list_ignore_classes)
                test_seg_head_loss = error_seg_head / cm_seg_head.sum()
            else:
                test_seg_head_maa = 0
                test_seg_head_miou = 0
                test_seg_head_loss = 0

            count_iter += 1
            if count_iter % 10 == 0:
                torch.cuda.empty_cache()

            if config["dual_seg_head"]:
                description = f"Epoch {epoch}|| TARGET {target_flag}|| Rec-IoU {test_iou*100:.2f} | Rec-Loss {test_aloss:.3e} | LossR {test_aloss_recons:.3e} | LossA {test_aloss_additional:.3e} | Seg-IoU {test_seg_head_miou*100:.2f}| Seg-AA {test_seg_head_maa*100:.2f} | Seg-Loss {test_seg_head_loss:.3e}"

            else:
                description = f"Epoch {epoch}|| TARGET {target_flag}|| Rec-IoU {test_iou*100:.2f} | Rec-Loss {test_aloss:.3e} | LossR {test_aloss_recons:.3e} | LossA {test_aloss_additional:.3e} |"
            t.set_description_str(wgreen(description))

        entropy_mean_avg = entropy_mean_sum / count_iter
        information_maximization_avg = information_maximization_sum / count_iter
        bnm_value_avg = bnm_value_sum.numpy() / count_iter
    print("Confusion matrix")
    np.set_printoptions(threshold=np.inf)
    print(cm_seg_head)
    np.set_printoptions(threshold=1000)
    lists_latents = (list_data_source, list_data_target)
    if config["dual_seg_head"]:
        return_data = {"test_oa": test_oa, "test_aa": test_aa, "test_iou": test_iou, "test_aloss": test_aloss, "test_aloss_recons": test_aloss_recons,
            "test_aloss_additional": test_aloss_additional, "test_seg_head_miou": test_seg_head_miou, "test_seg_head_maa": test_seg_head_maa,
            "test_seg_head_loss": test_seg_head_loss, "seg_iou_per_class": seg_iou_per_class, "accuracy_per_class": accuracy_per_class,
            "lists_latents": lists_latents, "cm_seg_head": cm_seg_head, "entropy_mean_avg": entropy_mean_avg[0],
            "information_maximization_avg": information_maximization_avg[0], "bnm_value_avg": bnm_value_avg[0]}
        return return_data

    else:
        return test_oa, test_aa, test_iou, test_aloss, test_aloss_recons, test_aloss_additional, lists_latents
