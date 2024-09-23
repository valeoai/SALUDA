
<div align='center'>

# <span style="color:#54c0e8">SALUDA</span>: <span style="color:#54c0e8">S</span>urface-based <span style="color:#54c0e8">A</span>utomotive <span style="color:#54c0e8">L</span>idar <span style="color:#54c0e8">U</span>nsupervised <span style="color:#54c0e8">D</span>omain <span style="color:#54c0e8">A</span>daptation

[Björn Michele](https://bjoernmichele.com)<sup>1,3</sup>&nbsp;&nbsp;
[Alexandre Boulch](https://boulch.eu/)<sup>1</sup>&nbsp;&nbsp;&nbsp;
[Gilles Puy](https://sites.google.com/site/puygilles/)<sup>1</sup>&nbsp;&nbsp;&nbsp;
[Tuan-Hung Vu](https://tuanhungvu.github.io/)<sup>1</sup>&nbsp;&nbsp;&nbsp;
[Renaud Marlet](http://imagine.enpc.fr/~marletr/)<sup>1,2</sup>&nbsp;&nbsp;
[Nicolas Courty](https://people.irisa.fr/Nicolas.Courty/)<sup>3</sup>&nbsp;&nbsp;&nbsp;

<sub>
<sup>1</sup> Valeo.ai, Paris, France&nbsp;
<sup>2</sup> LIGM, Ecole des Ponts, Univ Gustave Eiffel, CNRS, Marne-la-Vallée, France

<sup>3</sup> CNRS, IRISA, Univ. Bretagne Sud, Vannes, France
</sub>

<br/>

[![Arxiv](https://img.shields.io/badge/paper-arxiv.2304.03251-B31B1B.svg)](https://arxiv.org/abs/2304.03251)


<br/>

SALUDA has been accepted as a SPOTLIGHT at 3DV 2024

<br/>


![Overview](doc/architecture2.png)

</div>

<br/>


## 💡 Overview
Learning models on one labeled dataset that generalize well on another domain is a difficult task, as several shifts might happen between the data domains. This is notably the case for lidar data, for which models can exhibit large performance discrepancies due for instance to different lidar patterns or changes in acquisition conditions. This paper addresses the corresponding Unsupervised Domain Adaptation (UDA) task for semantic segmentation. To mitigate this problem, we introduce an unsupervised auxiliary task of learning an implicit underlying surface representation simultaneously on source and target data. As both domains share the same latent representation, the model is forced to accommodate discrepancies between the two sources of data. This novel strategy differs from classical minimization of statistical divergences or lidar-specific state-of-the-art domain adaptation techniques. Our experiments demonstrate that our method achieves a better performance than the current state of the art in synthetic-to-real and real-to-real scenarios.

More resources: [Slides](doc/Slides_SALUDA.pdf), [Poster](doc/Poster_SALUDA.pdf)

---

## 🎓 Citation

```
@article{michele2024saluda,
  title={{SALUDA}: Surface-based Automotive Lidar Unsupervised Domain Adaptation},
  author={Michele, Bjoern and Boulch, Alexandre and Puy, Gilles and Vu, Tuan-Hung and Marlet, Renaud and Courty, Nicolas},
  booktitle={2024 International Conference on 3D Vision (3DV)},
  year={2024},
  organization={IEEE}
}
```



## 🧰 Dependencies

This code was implemented and tested with python 3.10, PyTorch 1.11.0 and CUDA 11.3.
The backbone is implemented with version 1.4.0 of [Torchsparse](https://github.com/mit-han-lab/torchsparse.)([Exact commit](https://github.com/mit-han-lab/torchsparse/commit/69c1034ddb285798619380537802ea0ff03aeba6))
Additionally, [Sacred](https://github.com/IDSIA/sacred) 0.8.3 is used. 


---
## 💾 Datasets 

For our experiments we use the following datasets: [nuScenes](https://www.nuscenes.org/nuscenes), [SemanticKITTI](http://www.semantic-kitti.org/dataset.html), [SynthLiDAR](https://github.com/xiaoaoran/SynLiDAR) and [SemanticPOSS](http://www.poss.pku.edu.cn/semanticposs.html)

Please note that we use in all our experiments the official SubDataset of SynthLiDAR. 
The datasets should be placed in: data/

---

## 💪 Training 

1. Step:  Source/Target training with surface reconstruction regularization
```
nuScenes to SemanticKITTI
python train_single_back.py --name='SALUDA_ns_sk' with da_ns_sk

SynthLiDAR to SemanticKITTI
python train_single_back.py --name='SALUDA_syn_sk' with da_syn_sk"

nuScenes to SemanticPOSS
python train_single_back.py --name='SALUDA_ns_poss' with da_ns_poss

SynthLiDAR to SemanticPOSS (5 cm voxel size)
python train_single_back.py --name='SALUDA_syn_poss' with da_syn_poss
```

In a second step the previously obtained models are further refined with a self-training. For this we rely on the code-basis of CoSMix, but adapt the code so that it contains only a simple self-training. We provide more details in the folder [self_training](self_training/README.md)

---

## 🏁 Evaluation

Evaluation of a SALUDA model on nuScenes to SemanticKITTI: 

```
python eval.py --name='EVAL_SALUDA_ns_sk' with da_ns_sk network_decoder=InterpAllRadiusNoDirsNet network_decoder_k=1.0 save_dir=results_val/ ckpt_path_model=path/to/folder
```

Evaluation on SyntheticLiDAR to SemanticKITTI:

```
python eval.py --name='EVAL_SALUDA_syn_sk' with da_syn_sk network_decoder=InterpAllRadiusNoDirsNet network_decoder_k=1.0 save_dir=results_val/ ckpt_path_model=path/to/folder
```

---
## 🐘 Model zoo

DA Setting | Method | Backbone | Link |
---|---|---|---|
nuScenes to SemanticKITTI | SALUDA w/o ST |TorchSparse-MinkUNet  | [CKPT](https://github.com/valeoai/SALUDA/releases/download/v0.0.0/ns_sk_saluda_wo_st.zip) |
SyntheticLiDAR to SemanticKITTI | SALUDA  w/o ST |TorchSparse-MinkUNet  |  [CKPT](https://github.com/valeoai/SALUDA/releases/download/v0.0.0/syn_sk_saluda_wo_st.zip) |

The checkpoint should be placed in a folder. The link to this folder should be given to the "ckpt_path_model" parameter.  

---

## 🏅 Acknowledgments

This project would not have been possible without many community resources and repositories. Among them:

- [ALSO](https://github.com/valeoai/ALSO/)
- [POCO](https://github.com/valeoai/POCO)
- [CoSMix](https://github.com/saltoricristiano/cosmix-uda/)
- [SynLiDAR](https://github.com/xiaoaoran/SynLiDAR)
- [Torchsparse](https://github.com/mit-han-lab/torchsparse)

Please, consider acknowleding these projects.

---

## 📝 License

This work released under the terms of the [Apache 2.0 license](LICENSE).
