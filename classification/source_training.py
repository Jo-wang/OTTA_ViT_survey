import os
import logging
import copy
import torch

import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models.source_model import get_model

from conf import cfg, load_cfg_fom_args, get_num_classes

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

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
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    elif cfg.OPTIM.METHOD == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    trainset = torchvision.datasets.CIFAR10(root=cfg.DIR, train=True,
                                        download=True, transform=transform)
    # Split trainset into train and validation sets
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])

    testset = torchvision.datasets.CIFAR10(root=cfg.DIR, train=False,
                                           download=True, transform=transform)

    # Define data loaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=200,
                                              shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=100,
                                            shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=2)

    for epoch in range(cfg.OPTIM.EPOCH):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        best_model_weights = None

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * correct / total
        avg_train_loss = train_loss / len(trainloader)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data in valloader:
                inputs, labels = data[0].to(device), data[1].to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * correct / total
        avg_val_loss = val_loss / len(valloader)

        # Print epoch results
        print(f"Epoch [{epoch+1}/{cfg.OPTIM.EPOCH}] - "
              f"Train Loss: {avg_train_loss:.4f} - Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.2f}%")
        
        logger.info(f"Epoch [{epoch+1}/{cfg.OPTIM.EPOCH}] - "
              f"Train Loss: {avg_train_loss:.4f} - Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.2f}%")

        # Save the best model weights based on validation accuracy
        if epoch == 0:
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())

    # Load the best model weights
    
    model.load_state_dict(best_model_weights)

    # Ev    aluate on test set
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_acc = 100.0 * correct / total
    avg_test_loss = test_loss / len(testloader)
    save_path = cfg.SAVE_PATH + cfg.CORRUPTION.DATASET + str(test_acc) + ".pth"
    torch.save(best_model_weights, save_path)
   
    print(f"Test Loss: {avg_test_loss:.4f} - Test Acc: {test_acc:.2f}%")
    logger.info(f"Test Loss: {avg_test_loss:.4f} - Test Acc: {test_acc:.2f}%")



if __name__ == '__main__':
    train('"Training.', '/home/uqzxwang/code/test-time-adaptation/classification/cfgs/cifar10_c/source.yaml')

