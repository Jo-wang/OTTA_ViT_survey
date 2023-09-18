import os
import time
import torch
import wandb
import logging
import numpy as np

from models.model import get_model
from utils import get_accuracy, eval_domain_dict
from datasets.data_loading import get_test_loader
from conf import cfg, load_cfg_from_args, get_num_classes, adaptation_method_lookup

from methods.tent import Tent 
from methods.ttaug import TTAug
from methods.memo import MEMO
from methods.cotta import CoTTA
from methods.gtta import GTTA
from methods.adacontrast import AdaContrast
from methods.rmt import RMT
from methods.eata import EATA
# ! from methods.norm import Norm
from methods.lame import LAME
from methods.sar import SAR
from methods.rotta import RoTTA
from methods.roid import ROID
from methods.source_only import SO
from methods.conjugatePL import ConjugatePL

from gpu_mem_track import MemTracker


logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(description, path):
    # name = "cifar100c-vitb16_L5_eval"
    
    # name = "CIFAR10C_TTA_bs1_CoTTA_NoParaReset_UpdateALL_vitb16_L5_episodic"
    name = "CIFAR10C_TTA_bs16_ConjPL_Poly_vitb16_L5_episodic"
    
    # MemTracker.init_tracker(detail=False, path='mem_track/', verbose=False, device=1, filename=name + '.txt')
    
    load_cfg_from_args(description, path)
    valid_settings = ["reset_each_shift",           # reset the model state after the adaptation to a domain
                      "continual",                  # train on sequence of domain shifts without knowing when a shift occurs
                      "mixed_domains",              # consecutive test samples are likely to originate from different domains
                      "correlated",                 # sorted by class label
                      "mixed_domains_correlated",   # mixed domains + sorted by class label
                      "reset_each_shift_correlated"
                      ]
    check = False
    if check:
        wandb.init(project="test-time-adaptation", name=name, config=cfg, mode="disabled")
    else:
        wandb.init(project="test-time-adaptation", name=name, config=cfg)
    
    assert cfg.SETTING in valid_settings, f"The setting '{cfg.SETTING}' is not supported! Choose from: {valid_settings}"

    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)
    
    if cfg.MODEL.CHECKPOINT != None:        
        base_model = get_model(cfg, num_classes, ckpt=cfg.MODEL.CHECKPOINT)
    else:
        base_model = get_model(cfg, num_classes, ckpt=None)
        
    # MemTracker.track("Before loading model to GPU")
    
    base_model.to(device)

    # MemTracker.track("After lodaing model to GPU")
    
    # setup test-time adaptation method
    model = eval(f'{adaptation_method_lookup(cfg.MODEL.ADAPTATION)}')(cfg=cfg, model=base_model, num_classes=num_classes)
    logger.info(f"Successfully prepared test-time adaptation method: {cfg.MODEL.ADAPTATION.upper()}")

    # get the test sequence containing the corruptions or domain names
    dom_names_all = cfg.CORRUPTION.TYPE
    logger.info(f"Using the following domain sequence: {dom_names_all}")

    # prevent iterating multiple times over the same data in the mixed_domains setting
    dom_names_loop = ["mixed"] if "mixed_domains" in cfg.SETTING else dom_names_all

    # setup the severities for the gradual setting
    if "gradual" in cfg.SETTING and cfg.CORRUPTION.DATASET in {"cifar10_c", "cifar100_c", "imagenet_c"} and len(cfg.CORRUPTION.SEVERITY) == 1:
        severities = [1, 2, 3, 4, 5, 4, 3, 2, 1]
        logger.info(f"Using the following severity sequence for each domain: {severities}")
    else:
        severities = cfg.CORRUPTION.SEVERITY

    errs = []
    errs_5 = []
    domain_dict = {}
    
    # start evaluation
    for i_dom, domain_name in enumerate(dom_names_loop):
        if i_dom == 0 or "reset_each_shift" in cfg.SETTING:
            model.reset()
            logger.info("resetting model")
            
        else:
            logger.warning("not resetting model")

        for severity in severities:
            test_data_loader = get_test_loader(setting=cfg.SETTING,
                                               adaptation=cfg.MODEL.ADAPTATION,
                                               dataset_name=cfg.CORRUPTION.DATASET,
                                               root_dir=cfg.DATA_DIR,
                                               domain_name=domain_name,
                                               severity=severity,
                                               num_examples=cfg.CORRUPTION.NUM_EX,
                                               rng_seed=cfg.RNG_SEED,
                                               domain_names_all=dom_names_all,
                                               alpha_dirichlet=cfg.TEST.ALPHA_DIRICHLET,
                                               batch_size=cfg.TEST.BATCH_SIZE,
                                               shuffle=False,
                                               workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count()))

            acc, domain_dict, flops, params, elipsed_time = get_accuracy(
                model, data_loader=test_data_loader, dataset_name=cfg.CORRUPTION.DATASET,
                domain_name=domain_name, setting=cfg.SETTING, domain_dict=domain_dict)
            
            logger.info(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
            wandb.log({domain_name + " GFLOPs": flops / 1e9})
            logger.info(f"parameter amount: {params / 1e6:.2f} M")
            wandb.log({domain_name + " parameter amount (M)": params / 1e6})  
            logger.info(f"time spend: {elipsed_time:.2f} s")
            wandb.log({domain_name + " time spend (s)": elipsed_time})

            err = 1. - acc
            errs.append(err)
            if severity == 5 and domain_name != "none":
                errs_5.append(err)

            logger.info(f"{cfg.CORRUPTION.DATASET} error % [{domain_name}{severity}][#samples={len(test_data_loader.dataset)}]: {err:.2%}")
            wandb.log({f"{cfg.CORRUPTION.DATASET} error % [{domain_name}{severity}][#samples={len(test_data_loader.dataset)}]": err})

    # logger.info(f"time spend: {elapsed_time}")
    if len(errs_5) > 0:
        logger.info(f"mean error: {np.mean(errs):.2%}, mean error at 5: {np.mean(errs_5):.2%}")
        wandb.log({"mean error": np.mean(errs), "mean error at 5": np.mean(errs_5)})
    else:
        logger.info(f"mean error: {np.mean(errs):.2%}")
        wandb.log({"mean error": np.mean(errs)})

    if "mixed_domains" in cfg.SETTING:
        # print detailed results for each domain
        eval_domain_dict(domain_dict, domain_seq=dom_names_all)

    # import copy
    # best_model_weights = copy.deepcopy(model.state_dict())
    # save_path = cfg.SAVE_PATH + cfg.SETTING + "_BS" + str(cfg.TEST.BATCH_SIZE) + "_" + str(cfg.CORRUPTION.SEVERITY[0]) + "_" + cfg.CORRUPTION.DATASET + "_err" + str(np.mean(errs)*100) + "_" + cfg.MODEL.ARCH +  ".pth"
    # torch.save(best_model_weights, save_path)


if __name__ == '__main__':
    evaluate('"Evaluation.', '/home/uqzxwang/code/test-time-adaptation/classification/cfgs/cifar10_c/conjPL.yaml')

