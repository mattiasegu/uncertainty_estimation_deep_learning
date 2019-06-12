#!/usr/bin/env python
# coding: utf-8

"""
This script assembles the training dataset
from the folder relative to different bags
"""

from __future__ import print_function, division
import os
import glob
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
from opts import parser
from img_utils import load_img

FLAGS = parser.parse_args()

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class SteeringAnglesDataset(Dataset):
    """Steering Angles dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # Transforms
        if transform == None:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor()
            ])
        self.transform = transform
        # Read the csv file
        self.steering_df = pd.read_csv(csv_file[0])
        self.root_dir = root_dir

    def __len__(self):
        return len(self.steering_df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir,
                                self.steering_df.iloc[idx, 0])
        target_size = (FLAGS.img_width,FLAGS.img_height)
        crop_size = (FLAGS.crop_img_width,FLAGS.crop_img_height)
        image = load_img(img_path, grayscale=True, target_size=target_size, crop_size=crop_size)
        
        steering = self.steering_df.iloc[idx, 1]
        steering = np.array(steering)
        steering = steering.astype('float')
        
        # Transform image to tensor
        image = self.transform(image) # apply transforms to images 
        image = image/255. # rescale between [0,1]
        image = image.type(torch.FloatTensor)
        steering = torch.tensor(steering, dtype=torch.float)
        
        return (image,steering)


class Concat(Dataset):

    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.offsets = np.cumsum(self.lengths)
        self.length = np.sum(self.lengths)

    def __getitem__(self, index):
        for i, offset in enumerate(self.offsets):
            if index < offset:
                if i > 0:
                    index -= self.offsets[i-1]
                return self.datasets[i][index]
        raise IndexError(f'{index} exceeds {self.length}')

    def __len__(self):
        return self.length


def create_dataset(folder=None,mode=None):
    # Path to the data extracted from the Udacity dataset
    # folder = "training"  or "testing" or "validation"
    assert folder, "You should provide the dataset folder"
    experiments = glob.glob(folder + "/*")
    
    rotation_range = 0.2
    width_shift_range = 0.2
    height_shift_range = 0.2
    translation_range = [width_shift_range, height_shift_range]

    train_tf= transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees=rotation_range, translate=translation_range),
            transforms.ToTensor()
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    datasets = []
    
    for exp in experiments:
        csv_file = glob.glob(exp + "/sync_steering.csv")
        root_dir = exp
        # Create custom dataset for the current subfolder in folder
        dataset = SteeringAnglesDataset(csv_file=csv_file, root_dir=root_dir)
        datasets.append(dataset)
        if mode == 'augmentation':
            # Create custom dataset for the current subfolder in folder using data augmentation
            dataset = SteeringAnglesDataset(csv_file=csv_file, root_dir=root_dir, transform=train_tf)
            datasets.append(dataset)
    
    # Concatenate datasets in datasets list to build final custom training dataset
    your_custom_dataset = ConcatDataset(datasets)
    
    return your_custom_dataset

