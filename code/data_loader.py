from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import torch
from torch.utils import data
from utils import load_image
import numpy as np


class MelanomaDataLoader(data.Dataset):
  """
  A simple dataloader for melanoma dataset
  """
  def __init__(self,
               root_folder,
               folder=None,
               num_classes=100,
               split="train",
               transforms=None):
    assert split in ["train", "val", "test"]
    # root folder, split
    self.root_folder = root_folder
    self.split = split
    self.transforms = transforms
    self.n_classes = num_classes

    # load all labels
    if folder is None:
      folder = os.path.join(root_folder, "ISIC2018_Task1_Training_GroundTruth")
    if not os.path.exists(folder):
      raise ValueError(
        'Label folder {:s} does not exist!'.format(folder))
    
    masks = []
    if split == "train":
      for itr in range(1, 1568):
        filename = "ISIC_Mask_" + i + ".png"
        mask = np.ascontiguousarray(load_image(os.path.join(folder,filename)))
        if mask is not None:
          masks.append(mask)

    # load input images
    if folder is None:
      folder = os.path.join(root_folder, "ISIC2018_Task1-2_Training_Input")
    if not os.path.exists(folder):
      raise ValueError(
        'Input folder {:s} does not exist!'.format(folder))
    
    images = []
    if split == "train":
      #60% of data
      for itr in range(1, 1568):
        filename = "ISIC_Input_" + i + ".jpg"
        img = np.ascontiguousarray(load_image(os.path.join(folder,filename)))
        if img is not None:
          images.append(img)

    self.images = images
    self.masks = masks

  def __len__(self):
    return len(self.images)

  def __getitem__(self, index):
    # load img and label
    img, mask = self.images[index], self.masks[index]

    # apply data augmentation
    if self.transforms is not None:
      img  = self.transforms(img)
    return img, mask
