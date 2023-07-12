import os
import time
import torch
import yaml
import copy
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

logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_order(f):
   
    # Load the YAML file
    with open(f, 'r') as file:
        yaml_data = yaml.safe_load(file)
    orders = []
    # Loop over the types and read their content
    for i in range(1, 11):
        type_name = f"TYPE{i}"
        if type_name in yaml_data:
            orders.append(list(yaml_data[type_name]))
    return orders
            
        
def evaluate(description, path, orders):
    load_cfg_from_args(description, path)
    valid_settings = ["continual"]
    assert cfg.SETTING in valid_settings, f"The setting '{cfg.SETTING}' is not supported! Choose from: {valid_settings}"

    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)
    
    if cfg.MODEL.CHECKPOINT != None:        
        base_model = get_model(cfg, num_classes, ckpt=cfg.MODEL.CHECKPOINT)
    else:
        base_model = get_model(cfg, num_classes, ckpt=None)

    base_model.to(device)

    # setup test-time adaptation method
    model = eval(f'{adaptation_method_lookup(cfg.MODEL.ADAPTATION)}')(cfg=cfg, model=base_model, num_classes=num_classes)
    logger.info(f"Successfully prepared test-time adaptation method: {cfg.MODEL.ADAPTATION.upper()}")

    for i in range(len(orders)):
        # get the test sequence containing the corruptions or domain names
        dom_names_all = orders[i]
        logger.info(f"Using the following domain sequence: {dom_names_all}")
    
        severities = cfg.CORRUPTION.SEVERITY

        errs = []
        domain_dict = {}

        start_time = time.time()
        # start evaluation
        for i_dom, domain_name in enumerate(dom_names_all):
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

                acc, domain_dict = get_accuracy(
                    model, data_loader=test_data_loader, dataset_name=cfg.CORRUPTION.DATASET,
                    domain_name=domain_name, setting=cfg.SETTING, domain_dict=domain_dict)

                err = 1. - acc
                errs.append(err)
                logger.info(f"Type {i}: {cfg.CORRUPTION.DATASET} error % [{domain_name}{severity}][#samples={len(test_data_loader.dataset)}]: {err:.2%}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"time spend: {elapsed_time}")
        logger.info(f"mean error: {np.mean(errs):.2%}")

        best_model_weights = copy.deepcopy(model.state_dict())
        save_path = cfg.SAVE_PATH + "TYPE_" + str(i+1) + "_" + cfg.SETTING + "_BS" + str(cfg.TEST.BATCH_SIZE) + "_" + str(cfg.CORRUPTION.SEVERITY[0]) + "_" + cfg.CORRUPTION.DATASET + "_err" + str(np.mean(errs)*100) + "_" + cfg.MODEL.ARCH +  ".pth"
        torch.save(best_model_weights, save_path)
        model.reset()
        logger.info("resetting model")


if __name__ == '__main__':
    orders = read_order('/home/uqzxwang/code/test-time-adaptation/classification/cfgs/cifar10_c/10orders/tent0.yaml')
    evaluate('Evaluation.', '/home/uqzxwang/code/test-time-adaptation/classification/cfgs/cifar10_c/10orders/tent0.yaml', orders)

