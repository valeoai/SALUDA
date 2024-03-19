## Training - Step 2: Self-training
We base our self-training on the code of [CoSMix](https://github.com/saltoricristiano/cosmix-uda). Therefore, the first step is to clone the repository, and then do a few adaptations to the original code: 
1. Replace the backbone (as we use the TorchSparse library, instead of MinkoswkiEngine)
2. Replace the collation function 
3. Deactivate the mixing of source and target data (for the ease of the code adaptation, we solely deactivate the function. However, there could be a significant speed-up if we completely remove it)

After the self-training it is important to evaluate the student model with the by us provided eval.py as we calculate the mIoU on the global dataset and not as an average of the mIoUs of the scenes. 

A more detailed description:

1. Copy the folder models_torchsparse/ into cosmix-uda/utils/models/ and make sure that 'TorchSparseMinkUNet' is loaded as a backbone in adapt_cosmix.py, as a student model as well as a teacher model. In 'functions_to-exchange.py' we also provide a load_model.py that should be used to load the checkpoints that are trained with our training step 1. Furthermore, we use a different quantization function. Therefore, also the class 'ConcatDataset' in cosmix-uda/utils/datasets/concat_dataset.py has to be adapted. In detail, the 'voxelize' function has to be replaced with the one that we provide in quantization_concatenation.py  

2. Add in cosmix-uda/utils/collation.py the by us provied collate function 'collate_function_merged' that we provide in the collate.py. Make sure that you use this collate function for training_collation (at adapt_cosmix.py).

3. For deactivating the copying mechanism go to the 'SimMaskedAdaptation' class in comsix-uda/utils/pipelines/masked_simm_pipeline.py. You can uncomment the entire content of the 'mask' function, and just add 'mask = np.ones(dest_pts.shape[0])' at the end of the function. As we are using TorchSparse we also have to change the quantization. Therefore, the lines of quantization for MinkowskiEngine of 360 to 394 in comsix-uda/utils/pipelines/masked_simm_pipeline.py, have to be replaced by the quantization steps that we provide in our quantization_adaptation.py.

With these code changes you can then do a self-training with a model that you have trained in the first step. You load the checkpoints into the student and teacher model and start the adaptation process as described in the CoSMix repository. When the training is finished, evaluate the student model with the by us provided eval.py.

This self-training would not have been possible without the repository of [CoSMix](https://github.com/saltoricristiano/cosmix-uda/). Please, consider acknowleding also this project.