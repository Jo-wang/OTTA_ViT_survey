import numpy as np
import os
import random
from torch.utils.data import Dataset
from PIL import Image

class CIFAR10_1_Dataset(Dataset):
    def __init__(self, root_dir, transform, rng_seed=0):
        self.root_dir = root_dir
        self.seed = rng_seed
        self.transform = transform
        # Load the data into memory
        self.data = np.load(os.path.join(root_dir, "cifar10.1_v6_data.npy"), allow_pickle=True)
        self.targets = np.load(os.path.join(root_dir, "cifar10.1_v6_labels.npy"))
        
        # Shuffle the data
        self.indices = list(range(len(self.data)))
        random.seed(self.seed)
        random.shuffle(self.indices)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        
        img, target = self.data[actual_idx], self.targets[actual_idx]
        
        # Convert the image to PIL for easier transformation
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
        
        return img, target


def create_cifar10_1_dataset(root, transform=None, rng_seed=0):
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    dataset = CIFAR10_1_Dataset(root_dir=root, transform=transform, rng_seed=rng_seed)

    return dataset