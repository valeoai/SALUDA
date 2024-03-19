import os

import numpy as np


def default_logger_basic(prefix, prefix2, logger_func, metrics, metrics_target, train_iter_count, trg_mIoU=False):
  logger_func(f'{prefix}.OA', metrics[f"{prefix2}_oa"], train_iter_count)
  logger_func(f"{prefix}.AA", metrics[f"{prefix2}_aa"], train_iter_count)
  logger_func(f"{prefix}.IoU", metrics[f"{prefix2}_iou"], train_iter_count)
  logger_func(f"{prefix}.loss", metrics[f"{prefix2}_aloss"], train_iter_count)
  logger_func(f"{prefix}.lossRecons", metrics[f"{prefix2}_aloss_recons"], train_iter_count)
  logger_func(f"{prefix}.lossAdditional", metrics[f"{prefix2}_aloss_additional"], train_iter_count)
  logger_func(f"{prefix}.seg_MAA", metrics[f"{prefix2}_seg_head_maa"], train_iter_count)
  logger_func(f"{prefix}.seg_mIoU", metrics[f"{prefix2}_seg_head_miou"], train_iter_count)
  logger_func(f"{prefix}.seg_loss", metrics[f"{prefix2}_seg_head_loss"], train_iter_count)
  if not (metrics_target is None):
    logger_func(f"trg_{prefix}.OA", metrics_target[f"{prefix2}_oa"], train_iter_count)
    logger_func(f"trg_{prefix}.AA", metrics_target[f"{prefix2}_aa"], train_iter_count)
    logger_func(f"trg_{prefix}.IoU", metrics_target[f"{prefix2}_iou"], train_iter_count)
    logger_func(f"trg_{prefix}.loss", metrics_target[f"{prefix2}_aloss"], train_iter_count)
    logger_func(f"trg_{prefix}.lossRecons", metrics_target[f"{prefix2}_aloss_recons"], train_iter_count)
    logger_func(f"trg_{prefix}.lossAdditional", metrics_target[f"{prefix2}_aloss_additional"], train_iter_count)
    if trg_mIoU:
        logger_func(f"trg_{prefix}.seg_MAA", metrics_target[f"{prefix2}_seg_head_maa"], train_iter_count)
        logger_func(f"trg_{prefix}.seg_mIoU", metrics_target[f"{prefix2}_seg_head_miou"], train_iter_count)
        logger_func(f"trg_{prefix}.seg_loss", metrics_target[f"{prefix2}_seg_head_loss"], train_iter_count) 



def writer_logger_inside_seg(prefix, writer, metrics, metrics_target, train_iter_count):
  writer.add_scalar(f'{prefix}.inside_seg_iou', metrics["seg_inside_iou"], train_iter_count)
  writer.add_scalar(f'{prefix}.inside_seg_acc', metrics["seg_inside_acc"], train_iter_count)
  writer.add_scalar(f'trg_{prefix}.inside_seg_iou', metrics_target["seg_inside_iou"], train_iter_count)
  writer.add_scalar(f'trg_{prefix}.inside_seg_acc', metrics_target["seg_inside_acc"], train_iter_count)


def get_names_list(config):
  if (config["source_dataset_name"] == "SynLidar" and config['target_dataset_name'] == "SemanticPOSS") or \
    (config["source_dataset_name"] == "SemanticPOSS" and config['target_dataset_name'] == "SemanticPOSS"):
    names_list = ["ignore", "person", "rider", "car", "trunk", "plants", "traffic_sign", "pole",\
      "garbage_can", "building", "cone", "fence", "bike", "ground"]
    
  elif (config["source_dataset_name"] == "NuScenes" and config['target_dataset_name'] == "SemanticPOSS") or \
    (config["target_dataset_name"] == "NuScenes" and config['source_dataset_name'] == "SemanticPOSS"):
    names_list = ["ignore", "person", "bike", "car", "ground", "vegetation", "manmade"]
  
  elif config["source_dataset_name"] == "SynLidar" and config['target_dataset_name'] == "SemanticKITTI":
    names_list = ["ignore","car","bicycle","motorcycle","truck","other_vehicle","pedestrian","bicyclist",\
    "motorcyclist","road","parking","sidewalk","other_ground","building","fence","vegetation","trunk","terrain","pole","traffic_sign"]
  else:
    names_list=["ignore", "car", "bicycle", "motorcycle", "truck", "other_vehicle", "pedestrian", "driveable_surface", "sidewalk", "terrain", "vegetation"]

  return names_list

def default_logger_per_class(prefix, logger_func, val_data, train_iter_count, config):
  
  names_list=get_names_list(config)
  

  for idx, class_name in enumerate(names_list):
      logger_func(f"{prefix}_{class_name}", val_data["seg_iou_per_class"][idx], train_iter_count)
      logger_func(f"{prefix}_acc_{class_name}", val_data["accuracy_per_class"][idx], train_iter_count)


def run_logger_per_class(prefix, _run, val_data, train_iter_count, config):
  names_list=get_names_list(config)
  
  for idx, class_name in enumerate(names_list):
    _run.log_scalar(f"{prefix}_{class_name}", val_data["seg_iou_per_class"][idx],train_iter_count)
    _run.log_scalar(f"{prefix}_acc_{class_name}", val_data["accuracy_per_class"][idx],train_iter_count)
      
def dataset_config(names_list, val_log_data, val_data, prefix="", short_prefix='val'):
  for idx, class_name in enumerate(names_list):
    val_log_data[prefix+f"{short_prefix}_{class_name}"]= val_data["seg_iou_per_class"][idx]
    val_log_data[prefix+f"{short_prefix}_acc_{class_name}"]=val_data["accuracy_per_class"][idx]
  return val_log_data



def val_log_data_da(val_data_src, val_data_trg, train_iter_count, _run, writer, config, long_prefix="validation", short_prefix="val"):
  #For the DA case with 2 datasets (source and target)
  val_log_data = {
      f"{long_prefix}.OA": val_data_src["test_oa"],
      f"{long_prefix}.AA": val_data_src["test_aa"],
      f"{long_prefix}.IoU": val_data_src["test_iou"],
      f"{long_prefix}.loss": val_data_src["test_aloss"],
      f"{long_prefix}.lossRecons": val_data_src["test_aloss_recons"],
      f"{long_prefix}.lossAdditional": val_data_src["test_aloss_additional"],
      f"{long_prefix}.seg_MAA":val_data_src["test_seg_head_maa"],
      f"{long_prefix}.seg_mIoU":val_data_src["test_seg_head_miou"],
      f"{long_prefix}.seg_loss":val_data_src["test_seg_head_loss"],
      f"trg_{long_prefix}.OA": val_data_trg["test_oa"],
      f"trg_{long_prefix}.AA": val_data_trg["test_aa"],
      f"trg_{long_prefix}.IoU": val_data_trg["test_iou"],
      f"trg_{long_prefix}.loss": val_data_trg["test_aloss"],
      f"trg_{long_prefix}.lossRecons": val_data_trg["test_aloss_recons"],
      f"trg_{long_prefix}.lossAdditional": val_data_trg["test_aloss_additional"],
      f"trg_{long_prefix}.seg_MAA":val_data_trg["test_seg_head_maa"],
      f"trg_{long_prefix}.seg_mIoU":val_data_trg["test_seg_head_miou"],
      f"trg_{long_prefix}.seg_loss":val_data_trg["test_seg_head_loss"],
  }

  names_list=get_names_list(config)
  val_log_data=dataset_config(names_list, val_log_data, val_data_src, prefix="", short_prefix=short_prefix)
  val_log_data=dataset_config(names_list, val_log_data, val_data_trg, prefix="trg_", short_prefix=short_prefix)

  default_logger_basic(long_prefix, "test", _run.log_scalar, val_data_src, val_data_trg, train_iter_count)
  default_logger_per_class(short_prefix, _run.log_scalar, val_data_src, train_iter_count, config=config)
  default_logger_per_class(f"trg_{short_prefix}", _run.log_scalar, val_data_trg, train_iter_count, config=config)
  
  #Tensorboard logger
  default_logger_basic(long_prefix, "test", writer.add_scalar, val_data_src, val_data_trg, train_iter_count, trg_mIoU=True)
  default_logger_per_class(short_prefix, writer.add_scalar, val_data_src, train_iter_count, config=config)
  default_logger_per_class(f"trg_{short_prefix}", writer.add_scalar, val_data_trg, train_iter_count, config=config)

  return val_log_data



def  train_log_data_da(metrics, metrics_target, train_iter_count, _run, writer, config):

  if metrics is None:
    train_log_data = {}
  else:   
    train_log_data = {
        "training.OA": metrics["train_oa"],
        "training.AA": metrics["train_aa"],
        "training.IoU": metrics["train_iou"],
        "training.loss": metrics["train_aloss"],
        "training.lossRecons": metrics["train_aloss_recons"],
        "training.lossAdditional": metrics["train_aloss_additional"],
        
    }

  if metrics_target is None:
    pass
  else:
    #Performance on target
    train_log_data["trg_training.OA"]= metrics_target["train_oa"]
    train_log_data["trg_training.AA"]= metrics_target["train_aa"]
    train_log_data["trg_training.IoU"]= metrics_target["train_iou"]
    train_log_data["trg_training.loss"]= metrics_target["train_aloss"]
    train_log_data["trg_training.lossRecons"]= metrics_target["train_aloss_recons"]
    train_log_data["trg_training.lossAdditional"]= metrics_target["train_aloss_additional"]    
  if metrics is None:
    ###Make it more flexbile
    if not ("train_seg_head_maa" in metrics):
      #Add NaN
      metrics["train_seg_head_maa"] = np.nan

    train_log_data["training.seg_MAA"] = metrics["train_seg_head_maa"]

    if not ("train_seg_head_miou" in metrics):
      metrics["train_seg_head_miou"] = np.nan
      
    train_log_data["training.seg_mIoU"] = metrics["train_seg_head_miou"]

    if not ("train_seg_head_loss" in metrics):
      metrics["train_seg_head_loss"]=np.nan
    
    train_log_data["training.seg_loss"] =metrics["train_seg_head_loss"]

    names_list=get_names_list(config)
    train_log_data=dataset_config(names_list, train_log_data, metrics, prefix="")

    #Write the basic information for sacred and tensorboard
    default_logger_basic("training", "train", _run.log_scalar, metrics, metrics_target, train_iter_count)
    default_logger_basic("training", "train",  writer.add_scalar, metrics, metrics_target, train_iter_count)

  return train_log_data
