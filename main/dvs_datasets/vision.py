from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from . import myTransform


def cifar10dvs_dataset(data_dir, batch_size1, batch_size2, T):
    transform_train = transforms.Compose([
        myTransform.ToTensor(),
        transforms.Resize(size=(48, 48)),
        transforms.RandomCrop(48, padding=4),
        transforms.RandomHorizontalFlip(),])
        # transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]],std=[n / 255. for n in [68.2, 65.4, 70.4]]),
        # Cutout(n_holes=1, length=16)])

    transform_test = transforms.Compose([
        myTransform.ToTensor(),
        transforms.Resize(size=(48, 48))])
        # transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]], std=[n / 255. for n in [68.2, 65.4, 70.4]])
    trainset = CIFAR10DVS(root=data_dir, train=True, data_type='frame', frames_number=T, split_by='number', transform=transform_train)
    testset = CIFAR10DVS(root=data_dir, train=False, data_type='frame', frames_number=T, split_by='number',  transform=transform_test)
    train_data_loader = DataLoader(trainset, batch_size=batch_size1, shuffle=True, num_workers=2)
    test_data_loader = DataLoader(testset, batch_size=batch_size2, shuffle=False, num_workers=2)
    return train_data_loader, test_data_loader


def dvs128gesture_dataset(data_dir, batch_size1, batch_size2, T):
    trainset = DVS128Gesture(data_dir, train=True, data_type='frame', frames_number=T, split_by='number', 
                             transform=transforms.Compose([myTransform.SkipFrames(T), myTransform.GestureRoll(T)]))
    train_data_loader = DataLoader(trainset, batch_size=batch_size1, shuffle=True, num_workers=2)
    
    testset = DVS128Gesture(data_dir, train=False, data_type='frame', frames_number=T, split_by='number')
    test_data_loader = DataLoader(testset, batch_size=batch_size2, shuffle=False, num_workers=2)
    return train_data_loader, test_data_loader