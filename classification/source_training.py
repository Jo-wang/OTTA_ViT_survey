import os
import logging
import copy
import torch

import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from models.source_model import get_model

from conf import cfg, load_cfg_fom_args

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

logger = logging.getLogger(__name__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(description, path):

    load_cfg_fom_args(description, path)
    
    # num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)
    model = get_model(cfg).to(device)

    logger.info(f"Using the backbone: {cfg.MODEL.ARCH}")
    logger.info(f"Using the following source dataset: {cfg.CORRUPTION.DATASET}")

    
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()])

    criterion = nn.CrossEntropyLoss()
    if cfg.OPTIM.METHOD == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=cfg.OPTIM.LR, momentum=0.9)
    elif cfg.OPTIM.METHOD == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=int(0.1*cfg.OPTIM.ITER), T_mult=2, eta_min=0.001)
    
    if cfg.CORRUPTION.DATASET == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=cfg.DIR, train=True,
                                        download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=cfg.DIR, train=False,
                                           download=True, transform=transform)
    elif cfg.CORRUPTION.DATASET == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=cfg.DIR, train=True,
                                        download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root=cfg.DIR, train=False,
                                           download=True, transform=transform)
    elif cfg.CORRUPTION.DATASET == 'imagenet':
        trainset = torchvision.datasets.ImageNet(root=cfg.DIR, train=True,
                                        download=True, transform=transform)
        testset = torchvision.datasets.ImageNet(root=cfg.DIR, train=False,
                                           download=True, transform=transform)
        
    
    dataset = torch.utils.data.ConcatDataset([trainset, testset])

    # Define data loaders
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.TEST.BATCH_SIZE,
                                              shuffle=True, num_workers=4, pin_memory=True)
   
    for iteration in range(cfg.OPTIM.ITER):
        # Set learning rate for warm-up phase
        if iteration < cfg.OPTIM.WARMUP:
            lr = 0.1 * (iteration / cfg.OPTIM.WARMUP)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Move data to device
        inputs, labels = next(iter(dataloader))
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()

        # Update the scheduler
        scheduler.step()

        # Print progress
        if (iteration + 1) % 100 == 0:
            print(f"Iteration [{iteration + 1}/{cfg.OPTIM.ITER}]\tLoss: {loss.item()}")

       
    best_model_weights = copy.deepcopy(model.state_dict())

    # Load the best model weights
    
    model.load_state_dict(best_model_weights)

    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_acc = 100.0 * correct / total
    avg_test_loss = test_loss / len(dataloader)
    save_path = cfg.SAVE_PATH + cfg.CORRUPTION.DATASET + "_acc" + str(test_acc) + "_" + cfg.MODEL.ARCH +  ".pth"
    torch.save(best_model_weights, save_path)
   
    print(f"Final Loss: {avg_test_loss:.4f} - Final Acc: {test_acc:.2f}%")
    logger.info(f"Final Loss: {avg_test_loss:.4f} - Final Acc: {test_acc:.2f}%")



if __name__ == '__main__':
    train('"Training.', '/home/uqzxwang/code/test-time-adaptation/classification/cfgs/cifar100_c/source.yaml')

