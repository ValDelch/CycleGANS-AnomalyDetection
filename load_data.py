import random

import torch
from torch import nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import glob
import os
from PIL import Image
import json



class RandomChoice(torch.nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def __call__(self, imgs):
        t = random.choice(self.transforms)
        return [t(img) for img in imgs]

class ImageDataset(Dataset):
    """
    files is a list containing the path to the images to load
        ex: [
            path_to_image_xxx,
            path_to_image_xxx,
            ...
        ]
    """
    def __init__(self, files=[], transform=None, color_mode="RGB"):
        self.transform = transform
        self.color_mode = color_mode

    def __getitem__(self, index):
        item = self.transform(Image.open(self.files[index]).convert(self.color_mode))
        return {"image": item}

    def __len__(self):
        return len(self.files)

class ImageDatasetPaired(Dataset):
    """
    files is a list of tupples containing the paired path to the images to load
        ex: [
            (path_to_image_normal_xxx, path_to_image_abnormal_xxx),
            (path_to_image_normal_xxx, path_to_image_abnormal_xxx),
            ...
        ]
    """
    def __init__(self, files=[()], transform=None, color_mode="RGB"):
        self.transform = transform
        self.color_mode = color_mode

        if len(self.files_A) >= len(self.files_B):
            # Files in files_B should be taken at least once
            self.cache = self.files_B.copy()
        else:
            self.cache = self.files_A.copy()

    def __getitem__(self, index):
        if len(self.files_A) >= len(self.files_B):
            if self.cache:
                item_B = self.transform(Image.open(self.cache.pop(random.randint(0, len(self.cache) - 1))).convert(self.color_mode))
            else:
                item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert(self.color_mode))

            item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert(self.color_mode))

        else:
            if self.cache:
                item_A = self.transform(Image.open(self.cache.pop(random.randint(0, len(self.cache) - 1))).convert(self.color_mode))
            else:
                item_A = self.transform(Image.open(self.files_A[random.randint(0, len(self.files_A) - 1)]).convert(self.color_mode))

            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert(self.color_mode))

        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
    


def return_dataset(name, path, always_RGB, max_image_size, loader):

    with open('datasets_config.json', 'r') as json_file:
        dataset_config = json.load(json_file)[name]

    """ Setup """

    image_size = min([max_image_size, dataset_config["image_size"]])

    if loader == 'ImageDatasetPaired':
        train_dataloader = ImageDatasetPaired

        # Generate the splits
        normal_images = glob.glob(os.path.join())

    else:
        train_dataloader = ImageDataset

        # Generate the splits
    test_dataloader = ImageDataset

    if always_RGB or dataset_config["RGB"]:
        color_mode = "RGB"
        normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    else:
        color_mode = "L"
        normalize = transforms.Normalize([0.5], [0.5])

    if not dataset_config["hard_aug"]:
        train_tf = transforms.Compose([
            transforms.Resize((image_size, image_size), Image.Resampling.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomChoice([
                transforms.RandomRotation((90,90)),
                transforms.RandomRotation((180,180)),
                transforms.RandomRotation((270,270)),
                transforms.RandomRotation((0,0))
            ]),
            transforms.ToTensor(),
            normalize
        ])
        test_tf = transforms.Compose([
            transforms.Resize((image_size, image_size), Image.Resampling.BICUBIC),
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform_ori = transforms.Compose([
            transforms.Resize((image_size, image_size), Image.Resampling.BICUBIC),
            transforms.ToTensor(),
            normalize
        ])
        transform_ori_rot90 = transforms.Compose([
            transforms.Resize((image_size, image_size), Image.Resampling.BICUBIC),
            transforms.RandomRotation((90, 90)),
            transforms.ToTensor(),
            normalize
        ])
        transform_ori_rot180 = transforms.Compose([
            transforms.Resize((image_size, image_size), Image.Resampling.BICUBIC),
            transforms.RandomRotation((180,180)),
            transforms.ToTensor(),
            normalize
        ])
        transform_ori_rot270 = transforms.Compose([
            transforms.Resize((image_size, image_size), Image.Resampling.BICUBIC),
            transforms.RandomRotation((270,270)),
            transforms.ToTensor(),
            normalize
        ])
        transform_hflip = transforms.Compose([
            transforms.Resize((image_size, image_size), Image.Resampling.BICUBIC),
            transforms.RandomHorizontalFlip(p=1),
            transforms.ToTensor(),
            normalize
        ])
        transform_hflip_rot90 = transforms.Compose([
            transforms.Resize((image_size, image_size), Image.Resampling.BICUBIC),
            transforms.RandomRotation((90,90)),
            transforms.RandomHorizontalFlip(1),
            transforms.ToTensor(),
            normalize
        ])
        transform_hflip_rot180 = transforms.Compose([
            transforms.Resize((image_size, image_size), Image.Resampling.BICUBIC),
            transforms.RandomRotation((180,180)),
            transforms.RandomHorizontalFlip(p=1),
            transforms.ToTensor(),
            normalize
        ])
        transform_hflip_rot270 = transforms.Compose([
            transforms.Resize((image_size, image_size), Image.Resampling.BICUBIC),
            transforms.RandomRotation((270,270)),
            transforms.RandomHorizontalFlip(p=1),
            transforms.ToTensor(),
            normalize
        ])
        transform_vflip = transforms.Compose([
            transforms.Resize((image_size, image_size), Image.Resampling.BICUBIC),
            transforms.RandomVerticalFlip(p=1),
            transforms.ToTensor(),
            normalize
        ])

        transform_ori_test = transforms.Compose([
            transforms.Resize((image_size, image_size), Image.Resampling.BICUBIC),
            transforms.ToTensor(),
            normalize
        ])

    return (train_dataset, test_normal_dataset, test_abnormal_dataset)