import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from deeplearning.datasets.cifar10 import CIFAR10
from deeplearning.datasets.fmnist import FashionMNIST


dataset_registry = {
    "cifar10": CIFAR10,
    "fmnist": FashionMNIST,
}

def fetch_dataloader(config, train, test):
    train_loader = DataLoader(MyDataset(train["images"], train["labels"]), 
                    batch_size=config.batch_size, shuffle=True,
                    num_workers=config.workers, pin_memory=False)
    
    test_loader = DataLoader(MyDataset(test["images"], test["labels"]), 
                    batch_size=config.batch_size, shuffle=False,
                    num_workers=config.workers, pin_memory=False)

    return train_loader, test_loader

def fetch_dploader(config, train):
    if len(train) == 0 or config.use_sensitive==False:
        return []

    train_loader = DataLoader(MyDataset(train["images"], train["labels"]), 
                    batch_size=config.dp_batch_size, shuffle=True,
                    num_workers=config.workers, pin_memory=False)

    return train_loader


# def fetch_dataloader(config, train, test):
#     train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True,
#                                                 num_workers=config.workers, pin_memory=False)
#     test_loader = DataLoader(test, batch_size=config.batch_size, shuffle=False,
#                                                 num_workers=config.workers, pin_memory=False)

#     return train_loader, test_loader

# def fetch_dploader(config, train):
#     if len(train) == 0 or config.use_sensitive==False:
#         return []

#     train_loader = DataLoader(train, batch_size=config.dp_batch_size, shuffle=True,
#                                                 num_workers=config.workers, pin_memory=False)

#     return train_loader


def fetch_dataset(config):
    dataset = dataset_registry[config.dataset](config.data_path)

    config.num_classes = dataset.num_classes
    config.im_size = dataset.im_size
    config.channel = dataset.channel
    config.n_train = dataset.n_train

    return dataset


def fetch_subsets(dst_train, indices, pytorch_dataset=False):
    if pytorch_dataset:
        insensitive_subset = torch.utils.data.Subset(dst_train, indices["insensitive_idx"])
        
        # for sensitive dataset, no augmentation
        lowscore_sensitive_subset = torch.utils.data.Subset(dst_train, indices["lowscore_sensitive_idx"])
        highscore_sensitive_subset = torch.utils.data.Subset(dst_train, indices["highscore_sensitive_idx"])
        random_sensitive_subset = torch.utils.data.Subset(dst_train, indices["random_sensitive_idx"])

    else:
        insensitive_subset = {"images": dst_train["images"][indices["insensitive_idx"]],
                              "labels": dst_train["labels"][indices["insensitive_idx"]]}
        lowscore_sensitive_subset = {"images": dst_train["images"][indices["lowscore_sensitive_idx"]],
                                     "labels": dst_train["labels"][indices["lowscore_sensitive_idx"]]}
        highscore_sensitive_subset = {"images": dst_train["images"][indices["highscore_sensitive_idx"]],
                                      "labels": dst_train["labels"][indices["highscore_sensitive_idx"]]}

        n_array = np.arange(len(dst_train["images"]))
        np.random.seed(0)
        random_idx = np.random.permutation(n_array)[:len(indices["lowscore_sensitive_idx"])] 

        random_sensitive_subset = {"images": dst_train["images"][random_idx],
                                   "labels": dst_train["labels"][random_idx]}

    subsets = {
        "insensitive": insensitive_subset,
        "lowscore_sensitive": lowscore_sensitive_subset,
        "highscore_sensitive": highscore_sensitive_subset,
        "random_sensitive": random_sensitive_subset,
    }

    return subsets


class MyDataset(Dataset):
    def __init__(self, images, labels):
        """Construct a customized dataset
        """
        if min(labels) < 0:
            labels = (labels).reshape((-1,1)).astype(np.float32)
        else:
            labels = (labels).astype(np.int64)

        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)
        self.num_samples = images.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx] 
        return (image, label)


