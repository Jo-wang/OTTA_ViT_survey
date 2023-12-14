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

- **Datasets**
  - This repository allows studying a wide range of datasets, models, settings, and methods. A quick overview is given below:
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
  - Tent
  - CoTTA
  - MEMO
  - SAR
  - Conjugate PL
  - RoTTA
  - TAST



### Changing Configurations
All the hyperparameter could be changed in the folder classification/cfgs/


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

### Acknowledgements
+ Test-time Adaptation [official](https://github.com/mariodoebler/test-time-adaptation)
+ We would like to thank all the authors mentioned in the paper. Thank you for contributing Test-time Adaptation community.