import os
import random
import numpy as np
from torchvision.datasets import ImageFolder

def create_cifarw_dataset(root, domain_name, transform=None, domain_names_all=None, rng_seed=0):
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    # Transformations
    transform = transform

    path = os.path.join(root, domain_name)
    if os.path.exists(path):
        dataset = CustomDataset(root=path, transform=transform, seed=rng_seed)

        return dataset
    else:
        raise ValueError(f"Domain {domain_name} does not exist in {root}")
    

from torch.utils import data
from PIL import Image

class CustomDataset(data.Dataset):
    def __init__(self, root, transform=None, seed=0):
        self.root = root
        self.transform = transform
        self.class_to_id = {
            "airplane": 0,
            "sedan": 1,
            "bird": 2,
            "cat": 3,
            "deer": 4,
            "dog": 5,
            "frog": 6,
            "horse": 7,
            "ship": 8,
            "truck": 9,
            "suv": 1
        }
        self.classes = os.listdir(self.root)
        self.classes.sort()

        self.img_paths = []
        self.labels = []
        
        for class_name in os.listdir(self.root):
            class_dir = os.path.join(self.root, class_name)
            label = self.class_to_id.get(class_name, None)
            if label is not None:
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    self.img_paths.append(img_path)
                    self.labels.append(label)
                    
        random.seed(seed)
        np.random.seed(seed)
        zipped_list = list(zip(self.img_paths, self.labels))
        random.shuffle(zipped_list)
        self.img_paths, self.labels = zip(*zipped_list)


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label





