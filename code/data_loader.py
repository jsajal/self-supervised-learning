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
    
    if split == "train":
      start, end = 1, 2395 #1,1200; 1200,2400 #count = 1596
    elif split == "val":
      start, end = 2395, 2595 #count = 500
    elif split == "test":
      start, end = 121, 151 #2095, 2595 #count = 500

    masks_filename = []
    for itr in range(start, end):
      filename = "ISIC_Mask_" + str(itr) + ".png"
      mask = os.path.join(folder,filename)
      if mask is not None:
        masks_filename.append(mask)

    # load input images
    folder = os.path.join(root_folder, "ISIC2018_Task1-2_Training_Input")
    if not os.path.exists(folder):
      raise ValueError(
        'Input folder {:s} does not exist!'.format(folder))
    
    images_filename = []
    for itr in range(start, end):
      filename = "ISIC_Input_" + str(itr) + ".jpg"
      img = os.path.join(folder,filename)
      if img is not None:
        images_filename.append(img)

    self.images_filename = images_filename
    self.masks_filename = masks_filename

  def __len__(self):
    return len(self.images_filename)

  def __getitem__(self, index):
    # load img and label
    img_filename, mask_filename = self.images_filename[index], self.masks_filename[index]
    img = np.ascontiguousarray(load_image(img_filename, 1))
    mask = np.ascontiguousarray(load_image(mask_filename, 0))

    # apply data augmentation
    if self.transforms is not None:
      img  = self.transforms(img)
      mask  = self.transforms(mask)
      mask[mask > 0.5] = 1
      mask[mask <= 0.5] = 0
    return img, mask
