import os
import pickle
import numpy as np

from torchvision import datasets
from torch.utils.data.dataset import Dataset
from torchvision import transforms as T
from torch import tensor, long

def CIFAR10(data_path):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    channel = 3
    im_size = (32, 32)
    num_classes = 10

    with open(os.path.join(data_path, "train.dat"), "rb") as fp:
        dst_train = pickle.load(fp)
    
    with open(os.path.join(data_path, "test.dat"), "rb") as fp:
        dst_test = pickle.load(fp)

    # apply normalization
    train_images, test_images = dst_train["images"], dst_test["images"]
    train_images, test_images = train_images.astype(np.float32)/255, test_images.astype(np.float32)/255

    # reshape images    
    train_images = train_images.reshape(-1, 3, 32, 32)
    test_images = test_images.reshape(-1, 3, 32, 32)

    for i, mean_i in enumerate(mean):
        train_images[:, i] = (train_images[:, i] - mean[i])/std[i]
        test_images[:, i] = (test_images[:, i] - mean[i])/std[i]

    dst_train["images"], dst_test["images"] = train_images, test_images

    properties = {
        "channel": channel,
        "im_size": im_size,
        "num_classes": num_classes,
        "n_train": len(dst_train["images"]),
        "dst_train": dst_train,
        "dst_test": dst_test,
    }

    class dataset_properties: pass
    for key, value in properties.items():
        setattr(dataset_properties, key, value)

    return dataset_properties


# class MyCIFAR10(Dataset):
#     def __init__(self, file_path, train, download, transform):
#         self.cifar10 = datasets.CIFAR10(file_path, train=train, download=download, transform=transform)
#         self.targets = self.cifar10.targets
#         self.classes = self.cifar10.classes
#         self.clone_ver = 0      

#     def __getitem__(self, index):
#         data, target = self.cifar10[index]
#         return data, target, index + int(256*self.clone_ver)

#     def __len__(self):
#         return len(self.cifar10)

#     def update_clone_ver(self, cur_ver):
#         self.clone_ver = cur_ver

# def CIFAR10(data_path):
#     channel = 3
#     im_size = (32, 32)
#     num_classes = 10
#     mean = [0.4914, 0.4822, 0.4465]
#     std = [0.2470, 0.2435, 0.2616]

#     train_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(size=32, padding=4), T.ToTensor(), T.Normalize(mean=mean, std=std)])
#     test_transform = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
    
#     dst_train = MyCIFAR10(data_path, train=True, download=True, transform=train_transform)
#     dst_test = datasets.CIFAR10(data_path, train=False, download=False, transform=test_transform)
#     class_names = dst_train.classes
#     dst_train.targets = tensor(dst_train.targets, dtype=long)
#     dst_test.targets = tensor(dst_test.targets, dtype=long)
    
#     properties = {
#         "channel": channel,
#         "im_size": im_size,
#         "num_classes": num_classes,
#         "class_names": class_names,
#         "dst_train": dst_train,
#         "dst_test": dst_test
#     }
    
#     class dataset_properties: pass
#     for key, value in properties.items():
#         setattr(dataset_properties, key, value)

#     return dataset_properties