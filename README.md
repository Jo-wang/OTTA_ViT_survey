# Online Test-time Adaptation
This is the official implementation of paper In Search of Lost Online Test-time Adaptation: A Survey.
This implementation based on ViT backbones.


## Prerequisites
To use the repository, we provide a conda environment.
```bash
conda update conda
conda env create -f environment.yml
conda activate tta 
```

## Classification

<details open>
<summary>Features</summary>

This repository allows studying a wide range of datasets, models, settings, and methods. A quick overview is given below:

- **Datasets**
  - `cifar10_c` [CIFAR10-C](https://zenodo.org/record/2535967#.ZBiI7NDMKUk)
  - `cifar100_c` [CIFAR100-C](https://zenodo.org/record/3555552#.ZBiJA9DMKUk)
  - `imagenet_c` [ImageNet-C](https://zenodo.org/record/2235448#.Yj2RO_co_mF)
  
- **Settings**
  - `reset_each_shift` Reset the model state after the adaptation to a domain.
 
- **Batch Size**
  - 1, 16, 32, 64, 128

- **Backbone**
  - ViT B-16 224
  
  
- **Methods**
  - LN - Tent
  - CoTTA
  - MEMO
  - SAR
  - Conjugate PL
  - RoTTA
  - TAST



### Run Experiments

We provide config files for all experiments and methods. Simply run the following Python file with the corresponding config file.
```bash
python test_time.py --cfg cfgs/[cifar10_c/cifar100_c/imagenet_c/imagenet_others/domainnet126]/[source/norm_test/norm_alpha/tent/memo/eata/cotta/adacontrast/lame/sar/rotta/gtta/rmt/roid].yaml
```

For imagenet_others, the argument CORRUPTION.DATASET has to be passed:
```bash
python test_time.py --cfg cfgs/imagenet_others/[source/norm_test/norm_alpha/tent/memo/eata/cotta/adacontrast/lame/sar/rotta/gtta/rmt/roid].yaml CORRUPTION.DATASET [imagenet_a/imagenet_r/imagenet_k/imagenet_d109]
```

E.g., run the following command to run ROID for the ImageNet-to-ImageNet-R benchmark.
```bash
python test_time.py --cfg cfgs/imagenet_others/roid.yaml CORRUPTION.DATASET imagenet_r
```

Alternatively, you can reproduce our experiments by running the `run.sh` in the subdirectory `classification`. For the different settings, modify `setting` within `run.sh`.

To run the different continual DomainNet-126 sequences, you must pass the `CKPT_PATH` argument. When not specifying a `CKPT_PATH`, the sequence using the *real* domain as the source domain will be used.
The checkpoints are provided by [AdaContrast](https://github.com/DianCh/AdaContrast) and can be downloaded [here](https://drive.google.com/drive/folders/1OOSzrl6kzxIlEhNAK168dPXJcHwJ1A2X). Structurally, it is best to download them into the directory `./ckpt/domainnet126`.
```bash
python test_time.py --cfg cfgs/domainnet126/rmt.yaml CKPT_PATH ./ckpt/domainnet126/best_clipart_2020.pth
```

For GTTA, we provide checkpoint files for the style transfer network. The checkpoints are provided on Google Drive ([download](https://drive.google.com/file/d/1IpkUwyw8i9HEEjjD6pbbe_MCxM7yqKBq/view?usp=sharing)); extract the zip-file within the `classification` subdirectory.


### Changing Configurations
All the hyperparameter could be changed in the folder classification/cfgs/


### Acknowledgements
+ Test-time Adaptation [official](https://github.com/mariodoebler/test-time-adaptation)

### Acknowledgements
+ We would like to thank all the authors mentioned in the paper. Thank you for contributing Test-time Adaptation community.

Please consider cite:
```
@article{DBLP:journals/corr/abs-2310-20199,
  author       = {Zixin Wang and
                  Yadan Luo and
                  Liang Zheng and
                  Zhuoxiao Chen and
                  Sen Wang and
                  Zi Huang},
  title        = {In Search of Lost Online Test-time Adaptation: {A} Survey},
  journal      = {CoRR},
  volume       = {abs/2310.20199},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2310.20199},
  doi          = {10.48550/ARXIV.2310.20199},
  eprinttype    = {arXiv},
  eprint       = {2310.20199},
  timestamp    = {Fri, 03 Nov 2023 10:56:40 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2310-20199.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```