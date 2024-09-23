#Training with 1 Backbone but source and target
from torchsparse import SparseTensor

from config import ex
from general_imports import *
from utils.lcp import count_parameters, dict_to_device, get_dataset


@ex.automain
def main(_config, _run):
    experiment_name = _run.experiment_info["name"]
    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('runs/{}'.format(experiment_name))
    # convert config to dict
    config = eval(str(_config))
    # define the logging
    logging.getLogger().setLevel(config["logging"])
    # device
    device = torch.device(config['device'])
    if config["device"] == "cuda":
        torch.backends.cudnn.benchmark = True
    # get the savedir
    savedir_root = get_savedir_root(config, experiment_name)
    # create the network
    disable_log = not config["interactive_log"]
    N_LABELS = 2 #For occupancy
    
    net, network_function = construct_network(config, logging)
    net.to(device)
    logging.info(f"Network -- Number of parameters {count_parameters(net)}")
    logging.info("Getting the dataset")
    
    config['test_batch_size'] = 8
    source_DatasetClass = get_dataset(eval("datasets."+config["source_dataset_name"]))
    target_DatasetClass = get_dataset(eval("datasets."+config["target_dataset_name"]))
    dataloader_dict = da_get_dataloader(source_DatasetClass, target_DatasetClass, config, net, network_function)
    source_train_loader = dataloader_dict ["source_train_loader"]
    source_test_loader = dataloader_dict ["source_test_loader"]
    target_train_loader = dataloader_dict ["target_train_loader"]
    target_test_loader = dataloader_dict ["target_test_loader"]
 
    # Optimizer
    optimizer = optimizer_selection(logging, config, net)
    ##Learning rate scheduler 
    scheduler = learning_rate_scheduler_selection(logging, config, optimizer)

    # save the config file in the directory to restore the configuration
    if ("resume" in config) and (config["resume"]) and (os.path.exists(savedir_root)):
        net, optimizer, scheduler, epoch_start, train_iter_count, current_lr, best_checkpoint =\
             resume_model(net=net, savedir_root=savedir_root, device=device, optimizer=optimizer, scheduler=scheduler, source_train_loader=source_train_loader)
        if best_checkpoint is None: 
            best_ckpt_mioU_target = 0.0
            best_ckpt_epoch = 0
        else: 
            best_ckpt_mioU_target = best_checkpoint["best_mIoU"]
            best_ckpt_epoch = best_checkpoint["epoch"]

        logging.info(f"Best ckpt mIoU is set to {best_ckpt_mioU_target}, at epoch {best_ckpt_epoch}")
        
    else:
        #IF the training starts from new
        if os.path.exists(savedir_root):
            shutil.rmtree(savedir_root)
        os.makedirs(savedir_root, exist_ok=True)
        save_config_file(eval(str(config)), os.path.join(savedir_root, "config.yaml"))
        epoch_start = 0
        train_iter_count = 0    
        best_ckpt_mioU_target = 0.0
        best_ckpt_epoch = 0
        
           
    # create the loss layer
    loss_layer = torch.nn.BCEWithLogitsLoss()
    weights_ss = torch.ones(config["nb_classes"])
    list_ignore_classes = ignore_selection(config["ignore_idx"])
    for idx_ignore_class in list_ignore_classes: 
        weights_ss[idx_ignore_class] = 0
    

    logging.info(f"Ignored classes {list_ignore_classes}")
    logging.info(f"Weights of the different classes {weights_ss}")
    weights_ss= weights_ss.to(device)
    ce_loss_layer = torch.nn.CrossEntropyLoss(weight = weights_ss)
    epoch = epoch_start

    
    max_iteration_per_epoch = max(len(source_train_loader),0)
    train_iter_src = enumerate(source_train_loader)
    train_iter_trg = enumerate(target_train_loader)
    
    while True:
        net.train()
        if train_iter_count >= config["training_iter_nbr"]:
            break
        
        #Metrics for SOURCE
        metrics_holder_source = metrics_holder(N_LABELS=N_LABELS, config=config, target_flag=False)
        #Metrics for TARGET
        metrics_holder_target = metrics_holder(N_LABELS=N_LABELS, config=config, target_flag=True) 

        start_iteration = 0
        t = tqdm(range(start_iteration, max_iteration_per_epoch),desc="Epoch " + str(epoch), ncols=200, disable=disable_log,)
        for _ in t:
            
            # Load source and target data
            try:
                _, source_data = train_iter_src.__next__()
            except:
                train_iter_src = enumerate(source_train_loader)
                _, source_data = train_iter_src.__next__()
            try:
                _, target_data = train_iter_trg.__next__()
            except: 
                train_iter_trg = enumerate(target_train_loader)
                _, target_data = train_iter_trg.__next__()

            source_data = dict_to_device(source_data, device)
            target_data = dict_to_device(target_data, device)
            
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar(f"training.lr", current_lr, train_iter_count)
            

            #######################################################################################
            # Training on source                                                                  #
            # #####################################################################################    
            optimizer.zero_grad()
                
            output_data, output_seg = net.forward_pretraining(source_data)
            #Semantic Segmentation loss
            
            loss_seg = ce_loss_layer(output_seg, source_data["y"][:,None])
            
            outputs = output_data["predictions"].squeeze(-1)
            occupancies = output_data["occupancies"].float()
            #Reconstruction loss
            recons_loss = config["weight_rec_src"]*loss_layer(outputs, occupancies)
            writer.add_scalar(f"training.src.recons_loss",recons_loss, train_iter_count)
            

            loss_seg = config["weight_ss_src"]*loss_seg
            writer.add_scalar(f"training.src.seg_loss", loss_seg, train_iter_count)
            loss = recons_loss + loss_seg
            writer.add_scalar(f"training.src.loss",loss, train_iter_count)
            
            loss.backward()
            optimizer.step()
            metrics = calculation_metrics(metrics_holder_source, outputs, occupancies, loss_seg, loss,\
                        recons_loss, output_seg=output_seg, source_data=source_data, ignore_list=list_ignore_classes, output_data=output_data)
            
            del source_data
            del output_seg
            del output_data
            del loss
            torch.cuda.empty_cache()

            #######################################################################################
            # Training on target                                                                  #
            # #####################################################################################
    
            #Training on the same backbone as source (but only reconstruction loss)
            optimizer.zero_grad()
            output_data, output_seg_target = net.forward_pretraining(target_data)
            outputs = output_data["predictions"].squeeze(-1)
            occupancies = output_data["occupancies"].float()
            recons_loss = loss_layer(outputs, occupancies)
            loss = config["weight_rec_trg"] * recons_loss
            writer.add_scalar(f"training.trg.recons_loss",loss, train_iter_count)
                
            loss.backward()
            optimizer.step()
            scheduler.step()
                       
            metrics_target = calculation_metrics(metrics_holder_target, outputs, occupancies, None, loss, loss,\
                output_seg=output_seg_target, source_data=target_data, ignore_list=list_ignore_classes)
            
            description = f"Epoch {epoch} | SOURCE: Rec-IoU {metrics['train_iou']*100:.2f} | Seg-IoU {metrics['train_seg_head_miou']*100:.2f} ||TARGET: Rec-IoU {metrics_target['train_iou']*100:.2f} || LR: {current_lr:.3e}"    
            t.set_description_str(wblue(description))

            train_iter_count += 1

            if train_iter_count >= config["training_iter_nbr"]:
                break
            
            
            del target_data
            del output_seg_target
            del output_data
            del loss
            torch.cuda.empty_cache()
        
            
        
        ######################################
        
        #Save the current weights, optimizer and scheduler
        torch.save({"epoch": epoch + 1,"state_dict": net.state_dict(),"optimizer": optimizer.state_dict(),"scheduler":scheduler.state_dict()
            },os.path.join(savedir_root, "checkpoint.pth"),)

        data_saver={"metrics":metrics, "metrics_target":metrics_target, "train_iter_count":train_iter_count,"_run":_run,
                    "writer":writer,  "epoch":epoch,"net":net, "source_test_loader":source_test_loader, "target_test_loader":target_test_loader,
                    "N_LABELS":N_LABELS, "disable_log":disable_log, "disable_log":disable_log, "ce_loss_layer":ce_loss_layer,"loss_layer":loss_layer,
                    "list_ignore_classes":list_ignore_classes, "list_ignore_classes":list_ignore_classes, "device":device, "optimizer":optimizer, 
                    "scheduler":scheduler, "savedir_root":savedir_root, "best_ckpt_mioU_target":best_ckpt_mioU_target, "best_ckpt_epoch":best_ckpt_epoch}    
        best_ckpt_mioU_target, best_ckpt_epoch = save_val_model(config, data_saver)


        epoch += 1
    ####################  When training is finished      ##########################################
    torch.save({"epoch": epoch + 1,"state_dict": net.state_dict(),"optimizer": optimizer.state_dict(), "scheduler":scheduler.state_dict()},
            os.path.join(savedir_root, "checkpoint.pth"),)
                    