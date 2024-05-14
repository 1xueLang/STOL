import os
import torch
import torchvision
from torch.utils.data import DataLoader

def imagenet_dataset(data_dir, batch_size, test_batch_size, transform=None):
    if transform:
        transform_train, transform_test = transform
    else:
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.AutoAugment(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    tr_set = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'),
        transform=transform_train
    )
    ts_set = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'val'),
        transform=transform_test
    )
    
    return DataLoader(tr_set, shuffle=True, num_workers=16, batch_size=batch_size, pin_memory=True), \
        DataLoader(ts_set, shuffle=False, num_workers=16, batch_size=test_batch_size, pin_memory=True)
