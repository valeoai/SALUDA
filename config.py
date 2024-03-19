from sacred import SETTINGS, Experiment
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

SETTINGS.CAPTURE_MODE = 'sys'  # for tqdm

ex = Experiment("DA_experiments", save_git_info=False)
ex.observers.append(FileStorageObserver('my_runs'))
ex.captured_out_filter = apply_backspaces_and_linefeeds  # for tqdm

@ex.config
def config():
    # Dataset configs
    dataset_name = 'UnknwonDataset'
    dataset_root = 'UnknownDatasetPath'
    source_dataset_name = 'UnknwonDataset'
    source_dataset_root = 'UnknownDatasetPath'
    target_dataset_name = 'UnknwonDataset'
    target_dataset_root = 'UnknownDatasetPath'
    save_dir = 'results_ckpt'
    ns_dataset_version='v1.0-trainval'

    # splits
    train_split = 'train'
    val_split = 'val'
    test_split = 'val'
    nb_classes = 1

    # Method parameters
    input_intensities = False
    input_dirs = False
    input_normals = False
    source_input_intensities = False
    source_input_dirs = False    
    target_input_intensities = False
    target_input_dirs = False

    manifold_points = 10000
    non_manifold_points = 2048
    
    # Training parameters
    da_flag = True
    dual_seg_head = True
    training_iter_nbr=300000
    training_batch_size = 4
    test_batch_size = 1
    training_lr_start = 0.001
    training_lr_start_head = None
    optimizer = "AdamW"
    lr_scheduler = "cos_an_half_lr"
    voxel_size = 0.1
    val_interval = 10
    resume = False

    # Network parameter
    network_backbone = 'TorchSparseMinkUNet'
    network_latent_size = 128
    network_decoder = 'InterpAllRadiusNoDirsNet'
    network_decoder_k = 1.0
    network_n_labels = 1
    use_no_dirs_rec_head_flag = False
    rotation_head_flag = False

    # Technical parameter
    device = 'cuda'
    threads = 6
    interactive_log = True
    logging = 'INFO'

    # Data augmentation
    randRotationZ = True
    randFlip = True
    no_augmentation = False
    
    # Ckpt path 
    ckpt_path_model = "UnknownPath"

    # Weighting parameter for loss
    weight_rec_src = 1.0
    weight_rec_trg = 1.0
    weight_ss_src = 1.0
    weight_ss_trg = 1.0
    weight_inside_seg_src = 1.0

    # Ignorance idx
    ignore_idx = 0
    get_latent = False

    # Test flag
    test_flag_eval = False
    target_training = True
    source_training = True

    # Which ckpt to load from in eval
    ckpt_number = -1

@ex.named_config
def da_ns_sk():
    source_dataset_name = 'NuScenes'
    source_dataset_root = 'data/nuscenes'
    nb_classes = 11
    target_dataset_name = 'SemanticKITTI'
    target_dataset_root = 'data/SemanticKITTI'
    weight_rec_src=0.00001
    weight_rec_trg=0.00001
    


@ex.named_config
def da_syn_sk():
    source_dataset_name = 'SynLidar'
    source_dataset_root = 'data/synlidar'
    nb_classes = 20
    target_dataset_name = 'SemanticKITTI'
    target_dataset_root = 'data/SemanticKITTI'
    


@ex.named_config
def da_syn_poss():
    source_dataset_name = 'SynLidar'
    source_dataset_root = 'data/synlidar'
    nb_classes = 14
    target_dataset_name = 'SemanticPOSS'
    target_dataset_root = 'data/SemanticPOSS'
    weight_rec_src=0.00001
    weight_rec_trg=0.00001
    voxel_size = 0.05
    


@ex.named_config
def da_ns_poss():
    source_dataset_name = 'NuScenes'
    source_dataset_root = 'data/nuscenes'
    nb_classes = 7
    target_dataset_name = 'SemanticPOSS'
    target_dataset_root = 'data/SemanticPOSS'
    weight_rec_src=0.00001
    weight_rec_trg=0.00001