import random

import torch
from torch import nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os
from PIL import Image
import json

valid_extensions = ["png", "tif", "jpg", "jpeg"]

class RandomChoice(nn.Module):
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
        self.files = files
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
        self.files = files
        self.transform = transform
        self.color_mode = color_mode

    def __getitem__(self, index):
        normal_image = self.transform(Image.open(self.files[index][0]).convert(self.color_mode))
        abnormal_image = self.transform(Image.open(self.files[index][1]).convert(self.color_mode))
        return {"normal": normal_image, "abnormal": abnormal_image}

    def __len__(self):
        return len(self.files)

def return_dataset(name, path, model_config, dataset_config, id=0):

    """ Setup """

    image_size = min([model_config["max_image_size"], dataset_config["image_size"]])

    # Generate the splits
    normal_images = [f for f in os.listdir(os.path.join(path, name, 'normal')) 
                     if f.split('.')[-1].lower() in valid_extensions]
    abnormal_images = [f for f in os.listdir(os.path.join(path, name, 'abnormal')) 
                       if f.split('.')[-1].lower() in valid_extensions]
    random.Random(id).shuffle(normal_images)
    random.Random(id).shuffle(abnormal_images)
    test_size = min([len(normal_images), len(abnormal_images)]) // 2
    
    normal_images_test = normal_images[:test_size]
    abnormal_images_test = abnormal_images[:test_size]
    normal_images_train = normal_images[test_size:]
    abnormal_images_train = abnormal_images[test_size:]

    if model_config["data_loader"] == 'ImageDatasetPaired':
        train_dataloader = ImageDatasetPaired

        # Generate the training pairs
        images_train = []
        if len(normal_images_train) >= len(abnormal_images_train):
            # Files in files_B should be taken at least once
            cache = abnormal_images_train.copy() 
            for i in range(len(normal_images_train)):
                if cache:
                    images_train.append((os.path.join(path, name, 'normal', normal_images_train[i]),
                                         os.path.join(path, name, 'abnormal', cache.pop(random.randint(0,len(cache)-1)))))
                else:
                    images_train.append((os.path.join(path, name, 'normal', normal_images_train[i]),
                                         os.path.join(path, name, 'abnormal', abnormal_images_train[random.randint(0, len(abnormal_images_train)-1)])))
        else:
            cache = normal_images_train.copy()
            for i in range(len(abnormal_images_train)):
                if cache:
                    images_train.append((os.path.join(path, name, 'normal', cache.pop(random.randint(0,len(cache)-1))),
                                         os.path.join(path, name, 'abnormal', abnormal_images_train[i])))
                else:
                    images_train.append((os.path.join(path, name, 'normal', normal_images_train[random.randint(0, len(normal_images_train)-1)]),
                                         os.path.join(path, name, 'abnormal', abnormal_images_train[i])))

    else:
        train_dataloader = ImageDataset

        # Generate the training split
        images_train = [os.path.join(path, name, 'normal', x) for x in normal_images_train]

    test_dataloader = ImageDataset
    normal_images_test = [os.path.join(path, name, 'normal', x) for x in normal_images_test]
    abnormal_images_test = [os.path.join(path, name, 'abnormal', x) for x in abnormal_images_test]

    if model_config["always_RGB"] or dataset_config["RGB"]:
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

        train_dataset = train_dataloader(
            files=images_train,
            transform=train_tf,
            color_mode=color_mode
        )
        test_normal_dataset = test_dataloader(
            files=normal_images_test,
            transform=test_tf,
            color_mode=color_mode
        )
        test_abnormal_dataset = test_dataloader(
            files=abnormal_images_test,
            transform=test_tf,
            color_mode=color_mode
        )

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

        train_dataset_ori = train_dataloader(
            files=images_train,
            transform=transform_ori,
            color_mode=color_mode
        )
        train_dataset_ori_rot90 = train_dataloader(
            files=images_train,
            transform=transform_ori_rot90,
            color_mode=color_mode
        )
        train_dataset_ori_rot180 = train_dataloader(
            files=images_train,
            transform=transform_ori_rot180,
            color_mode=color_mode
        )
        train_dataset_ori_rot270 = train_dataloader(
            files=images_train,
            transform=transform_ori_rot270,
            color_mode=color_mode
        )
        train_dataset_hflip = train_dataloader(
            files=images_train,
            transform=transform_hflip,
            color_mode=color_mode
        )
        train_dataset_hflip_rot90 = train_dataloader(
            files=images_train,
            transform=transform_hflip_rot90,
            color_mode=color_mode
        )
        train_dataset_hflip_rot180 = train_dataloader(
            files=images_train,
            transform=transform_hflip_rot180,
            color_mode=color_mode
        )
        train_dataset_hflip_rot270 = train_dataloader(
            files=images_train,
            transform=transform_hflip_rot270,
            color_mode=color_mode
        )
        train_dataset_vflip = train_dataloader(
            files=images_train,
            transform=transform_vflip,
            color_mode=color_mode
        )
        train_dataset = torch.utils.data.ConcatDataset([
            train_dataset_ori,
            train_dataset_ori_rot90,
            train_dataset_ori_rot180,
            train_dataset_ori_rot270,
            train_dataset_hflip,
            train_dataset_hflip_rot90,
            train_dataset_hflip_rot180,
            train_dataset_hflip_rot270,
            train_dataset_vflip
        ])
        print(len(train_dataset))
        test_normal_dataset = test_dataloader(
            files=normal_images_test,
            transform=transform_ori_test,
            color_mode=color_mode
        )
        test_abnormal_dataset = test_dataloader(
            files=abnormal_images_test,
            transform=transform_ori_test,
            color_mode=color_mode
        )

    train_dataset = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=dataset_config["batch_size"],
        shuffle=True
    )
    test_normal_dataset = torch.utils.data.DataLoader(
        test_normal_dataset,
        batch_size=dataset_config["batch_size"],
        shuffle=False
    )
    test_abnormal_dataset = torch.utils.data.DataLoader(
        test_abnormal_dataset,
        batch_size=dataset_config["batch_size"],
        shuffle=False
    )

    return (train_dataset, test_normal_dataset, test_abnormal_dataset)