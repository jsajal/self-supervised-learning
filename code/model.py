from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.nn.functional import fold, unfold
from torchvision.utils import make_grid
import math
from utils import resize_image
from collections import OrderedDict


#################################################################################
# Autoencoder implemented as UNet
#################################################################################

class SelfSupervised(nn.Module):
  # a simple UNet for self supervision task
  def __init__(self, conv_op=nn.Conv2d):
    super(SelfSupervised, self).__init__()
    #input : [-1, 3, 512, 512]
    self.down1 = nn.Sequential(
      # conv1 block:
      conv_op(3, 64, kernel_size=3, stride=1, padding=1), # [-1, 64, 512, 512]
      nn.ReLU(inplace=True),
      conv_op(64, 64, kernel_size=3, stride=1, padding=1), # [-1, 64, 512, 512]
      nn.ReLU(inplace=True),
    )
    self.down2 = nn.Sequential(
      # max pooling 1/2
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # [-1, 64, 256, 256]
      # conv2 block
      conv_op(64, 128, kernel_size=3, stride=1, padding=1), # [-1, 128, 256, 256]
      nn.ReLU(inplace=True),
      conv_op(128, 128, kernel_size=3, stride=1, padding=1), # [-1, 128, 256, 256]
      nn.ReLU(inplace=True),
    )
    self.down3 = nn.Sequential(
      # max pooling 1/2
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # [-1, 128, 128, 128]
      # conv3 block:
      conv_op(128, 256, kernel_size=3, stride=1, padding=1), # [-1, 256, 128, 128]
      nn.ReLU(inplace=True),
      conv_op(256, 256, kernel_size=3, stride=1, padding=1), # [-1, 256, 128, 128]
      nn.ReLU(inplace=True),
    )
    self.down4 = nn.Sequential(
      # max pooling 1/2
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # [-1, 256, 64, 64]
      # conv4 block:
      conv_op(256, 512, kernel_size=3, stride=1, padding=1), # [-1, 512, 64, 64]
      nn.ReLU(inplace=True),
      conv_op(512, 512, kernel_size=3, stride=1, padding=1), # [-1, 512, 64, 64]
      nn.ReLU(inplace=True),
    )
    self.down5 = nn.Sequential(
      # max pooling 1/2
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # [-1, 512, 32, 32]
      # conv5 block:
      conv_op(512, 1024, kernel_size=3, stride=1, padding=1), # [-1, 1024, 32, 32]
      nn.ReLU(inplace=True),
      conv_op(1024, 1024, kernel_size=3, stride=1, padding=1), # [-1, 1024, 32, 32]
      nn.ReLU(inplace=True),
    )
    
    self.up1 = nn.Sequential(
      #upSample
      nn.Upsample(scale_factor=2, mode='bilinear'), # [-1, 1024, 64, 64]
      conv_op(1024, 512, kernel_size=1, stride=1, padding=0), # [-1, 512, 64, 64]
    )
    self.up1conv = nn.Sequential(
      #conv block
      conv_op(1024, 512, kernel_size=3, stride=1, padding=1), # [-1, 512, 64, 64]
      nn.ReLU(inplace=True),
      conv_op(512, 512, kernel_size=3, stride=1, padding=1), # [-1, 512, 64, 64]
      nn.ReLU(inplace=True), 
    )
    self.up2 = nn.Sequential(
      #upSample
      nn.Upsample(scale_factor=2, mode='bilinear'), # [-1, 512, 128, 128]
      conv_op(512, 256, kernel_size=1, stride=1, padding=0), # [-1, 256, 128, 128]
    )
    self.up2conv = nn.Sequential(
      #conv block
      conv_op(512, 256, kernel_size=3, stride=1, padding=1), # [-1, 256, 128, 128]
      nn.ReLU(inplace=True),
      conv_op(256, 256, kernel_size=3, stride=1, padding=1), # [-1, 256, 128, 128]
      nn.ReLU(inplace=True), 
    )
    self.up3 = nn.Sequential(
      #upSample
      nn.Upsample(scale_factor=2, mode='bilinear'), # [-1, 256, 256, 256]
      conv_op(256, 128, kernel_size=1, stride=1, padding=0), # [-1, 128, 256, 256]
    )
    self.up3conv = nn.Sequential(
      #conv block
      conv_op(256, 128, kernel_size=3, stride=1, padding=1), # [-1, 128, 256, 256]
      nn.ReLU(inplace=True),
      conv_op(128, 128, kernel_size=3, stride=1, padding=1), # [-1, 128, 256, 256]
      nn.ReLU(inplace=True), 
    )
    self.up4 = nn.Sequential(
      #upSample
      nn.Upsample(scale_factor=2, mode='bilinear'), # [-1, 128, 512, 512]
      conv_op(128, 64, kernel_size=1, stride=1, padding=0), # [-1, 64, 512, 512]
    )
    self.up4conv = nn.Sequential(
      #conv block
      conv_op(128, 64, kernel_size=3, stride=1, padding=1), # [-1, 64, 512, 512]
      nn.ReLU(inplace=True),
      conv_op(64, 64, kernel_size=3, stride=1, padding=1), # [-1, 64, 512, 512]
      nn.ReLU(inplace=True), 
    )
    self.output = nn.Sequential(
      #conv block
      conv_op(64, 3, kernel_size=3, stride=1, padding=1), # [-1, 3, 512, 512]
      nn.ReLU(inplace=True),
    )

  def forward(self, x):

    x1 = self.down1(x)
    x2 = self.down2(x1)
    x3 = self.down3(x2)
    x4 = self.down4(x3)
    x5 = self.down5(x4)
    x6 = self.up1(x5)
    x6 = torch.cat((x4, x6), dim=1)
    x6 = self.up1conv(x6)
    x7 = self.up2(x6)
    x7 = torch.cat((x3, x7), dim=1)
    x7 = self.up2conv(x7)
    x8 = self.up3(x7)
    x8 = torch.cat((x2, x8), dim=1)
    x8 = self.up3conv(x8)
    x9 = self.up4(x8)
    x9 = torch.cat((x1, x9), dim=1)
    x9 = self.up4conv(x9)

    xOutput = self.output(x9)
    return xOutput

preTrain_model = SelfSupervised



#################################################################################
# Image Segmentation network implemented as UNet
#################################################################################

class ImageSegmentation(nn.Module):
  # a simple UNet for self supervision task
  def __init__(self, initial_param, conv_op=nn.Conv2d):
    super(ImageSegmentation, self).__init__()
    #input : [-1, 3, 512, 512]
    self.down1 = nn.Sequential(OrderedDict([
      # conv1 block:
      ('down1_1',conv_op(3, 64, kernel_size=3, stride=1, padding=1)), # [-1, 64, 512, 512]
      ('relu1',nn.ReLU(inplace=True)),
      ('down1_2',conv_op(64, 64, kernel_size=3, stride=1, padding=1)), # [-1, 64, 512, 512]
      ('relu2',nn.ReLU(inplace=True)),
    ]))
    self.down1._modules["down1_1"].weight = torch.nn.Parameter(initial_param['down1.0.weight'])
    self.down1._modules["down1_1"].bias  = torch.nn.Parameter(initial_param['down1.0.bias'])
    self.down1._modules["down1_2"].weight = torch.nn.Parameter(initial_param['down1.2.weight'])
    self.down1._modules["down1_2"].bias  = torch.nn.Parameter(initial_param['down1.2.bias'])

    self.down2 = nn.Sequential(OrderedDict([
      # max pooling 1/2
      ('mp1',nn.MaxPool2d(kernel_size=2, stride=2, padding=0)), # [-1, 64, 256, 256]
      # conv2 block
      ('down2_1',conv_op(64, 128, kernel_size=3, stride=1, padding=1)), # [-1, 128, 256, 256]
      ('relu1',nn.ReLU(inplace=True)),
      ('down2_2',conv_op(128, 128, kernel_size=3, stride=1, padding=1)), # [-1, 128, 256, 256]
      ('relu2',nn.ReLU(inplace=True)),
    ]))
    self.down2._modules["down2_1"].weight = torch.nn.Parameter(initial_param['down2.1.weight'])
    self.down2._modules["down2_1"].bias  = torch.nn.Parameter(initial_param['down2.1.bias'])
    self.down2._modules["down2_2"].weight = torch.nn.Parameter(initial_param['down2.3.weight'])
    self.down2._modules["down2_2"].bias  = torch.nn.Parameter(initial_param['down2.3.bias'])

    self.down3 = nn.Sequential(OrderedDict([
      # max pooling 1/2
      ('mp2',nn.MaxPool2d(kernel_size=2, stride=2, padding=0)), # [-1, 128, 128, 128]
      # conv3 block:
      ('down3_1',conv_op(128, 256, kernel_size=3, stride=1, padding=1)), # [-1, 256, 128, 128]
      ('relu1',nn.ReLU(inplace=True)),
      ('down3_2',conv_op(256, 256, kernel_size=3, stride=1, padding=1)), # [-1, 256, 128, 128]
      ('relu2',nn.ReLU(inplace=True)),
    ]))
    self.down3._modules["down3_1"].weight = torch.nn.Parameter(initial_param['down3.1.weight'])
    self.down3._modules["down3_1"].bias  = torch.nn.Parameter(initial_param['down3.1.bias'])
    self.down3._modules["down3_2"].weight = torch.nn.Parameter(initial_param['down3.3.weight'])
    self.down3._modules["down3_2"].bias  = torch.nn.Parameter(initial_param['down3.3.bias'])

    self.down4 = nn.Sequential(OrderedDict([
      # max pooling 1/2
      ('mp3',nn.MaxPool2d(kernel_size=2, stride=2, padding=0)), # [-1, 256, 64, 64]
      # conv4 block:
      ('down4_1',conv_op(256, 512, kernel_size=3, stride=1, padding=1)), # [-1, 512, 64, 64]
      ('relu1',nn.ReLU(inplace=True)),
      ('down4_2',conv_op(512, 512, kernel_size=3, stride=1, padding=1)), # [-1, 512, 64, 64]
      ('relu2',nn.ReLU(inplace=True)),
    ]))
    self.down4._modules["down4_1"].weight = torch.nn.Parameter(initial_param['down4.1.weight'])
    self.down4._modules["down4_1"].bias  = torch.nn.Parameter(initial_param['down4.1.bias'])
    self.down4._modules["down4_2"].weight = torch.nn.Parameter(initial_param['down4.3.weight'])
    self.down4._modules["down4_2"].bias  = torch.nn.Parameter(initial_param['down4.3.bias'])

    self.down5 = nn.Sequential(OrderedDict([
      # max pooling 1/2
      ('mp4',nn.MaxPool2d(kernel_size=2, stride=2, padding=0)), # [-1, 512, 32, 32]
      # conv5 block:
      ('down5_1',conv_op(512, 1024, kernel_size=3, stride=1, padding=1)), # [-1, 1024, 32, 32]
      ('relu1',nn.ReLU(inplace=True)),
      ('down5_2',conv_op(1024, 1024, kernel_size=3, stride=1, padding=1)), # [-1, 1024, 32, 32]
      ('relu2',nn.ReLU(inplace=True)),
    ]))
    self.down5._modules["down5_1"].weight = torch.nn.Parameter(initial_param['down5.1.weight'])
    self.down5._modules["down5_1"].bias  = torch.nn.Parameter(initial_param['down5.1.bias'])
    self.down5._modules["down5_2"].weight = torch.nn.Parameter(initial_param['down5.3.weight'])
    self.down5._modules["down5_2"].bias  = torch.nn.Parameter(initial_param['down5.3.bias'])
    
    self.up1 = nn.Sequential(
      #upSample
      nn.ConvTranspose2d(1024, 512, 2, 2), # [-1, 512, 64, 64]
    )
    self.up1conv = nn.Sequential(
      #conv block
      conv_op(1024, 512, kernel_size=3, stride=1, padding=1), # [-1, 512, 64, 64]
      nn.ReLU(inplace=True),
      conv_op(512, 512, kernel_size=3, stride=1, padding=1), # [-1, 512, 64, 64]
      nn.ReLU(inplace=True), 
    )
    self.up2 = nn.Sequential(
      #upSample
      nn.ConvTranspose2d(512, 256, 2, 2), # [-1, 256, 128, 128]
    )
    self.up2conv = nn.Sequential(
      #conv block
      conv_op(512, 256, kernel_size=3, stride=1, padding=1), # [-1, 256, 128, 128]
      nn.ReLU(inplace=True),
      conv_op(256, 256, kernel_size=3, stride=1, padding=1), # [-1, 256, 128, 128]
      nn.ReLU(inplace=True), 
    )
    self.up3 = nn.Sequential(
      #upSample
      nn.ConvTranspose2d(256, 128, 2, 2), # [-1, 128, 256, 256]
    )
    self.up3conv = nn.Sequential(
      #conv block
      conv_op(256, 128, kernel_size=3, stride=1, padding=1), # [-1, 128, 256, 256]
      nn.ReLU(inplace=True),
      conv_op(128, 128, kernel_size=3, stride=1, padding=1), # [-1, 128, 256, 256]
      nn.ReLU(inplace=True), 
    )
    self.up4 = nn.Sequential(
      #upSample
      nn.ConvTranspose2d(128, 64, 2, 2), # [-1, 64, 512, 512]
    )
    self.up4conv = nn.Sequential(
      #conv block
      conv_op(128, 64, kernel_size=3, stride=1, padding=1), # [-1, 64, 512, 512]
      nn.ReLU(inplace=True),
      conv_op(64, 64, kernel_size=3, stride=1, padding=1), # [-1, 64, 512, 512]
      nn.ReLU(inplace=True), 
    )
    self.output = nn.Sequential(
      #conv block
      conv_op(64, 1, kernel_size=1, stride=1, padding=0), # [-1, 1, 512, 512]
      nn.Sigmoid(),
    )

  def forward(self, x):

    x1 = self.down1(x)
    x2 = self.down2(x1)
    x3 = self.down3(x2)
    x4 = self.down4(x3)
    x5 = self.down5(x4)
    x6 = self.up1(x5)
    x6 = torch.cat((x4, x6), dim=1)
    x6 = self.up1conv(x6)
    x7 = self.up2(x6)
    x7 = torch.cat((x3, x7), dim=1)
    x7 = self.up2conv(x7)
    x8 = self.up3(x7)
    x8 = torch.cat((x2, x8), dim=1)
    x8 = self.up3conv(x8)
    x9 = self.up4(x8)
    x9 = torch.cat((x1, x9), dim=1)
    x9 = self.up4conv(x9)

    xOutput = self.output(x9)
    return xOutput

imgSeg_model = ImageSegmentation



#################################################################################
# UNet
#################################################################################

class UNet(nn.Module):
  # a simple UNet for self supervision task
  def __init__(self, conv_op=nn.Conv2d):
    super(UNet, self).__init__()
    #input : [-1, 3, 512, 512]
    self.down1 = nn.Sequential(
      # conv1 block:
      conv_op(3, 64, kernel_size=3, stride=1, padding=1), # [-1, 64, 512, 512]
      nn.ReLU(inplace=True),
      conv_op(64, 64, kernel_size=3, stride=1, padding=1), # [-1, 64, 512, 512]
      nn.ReLU(inplace=True),
    )
    self.down2 = nn.Sequential(
      # max pooling 1/2
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # [-1, 64, 256, 256]
      # conv2 block
      conv_op(64, 128, kernel_size=3, stride=1, padding=1), # [-1, 128, 256, 256]
      nn.ReLU(inplace=True),
      conv_op(128, 128, kernel_size=3, stride=1, padding=1), # [-1, 128, 256, 256]
      nn.ReLU(inplace=True),
    )
    self.down3 = nn.Sequential(
      # max pooling 1/2
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # [-1, 128, 128, 128]
      # conv3 block:
      conv_op(128, 256, kernel_size=3, stride=1, padding=1), # [-1, 256, 128, 128]
      nn.ReLU(inplace=True),
      conv_op(256, 256, kernel_size=3, stride=1, padding=1), # [-1, 256, 128, 128]
      nn.ReLU(inplace=True),
    )
    self.down4 = nn.Sequential(
      # max pooling 1/2
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # [-1, 256, 64, 64]
      # conv4 block:
      conv_op(256, 512, kernel_size=3, stride=1, padding=1), # [-1, 512, 64, 64]
      nn.ReLU(inplace=True),
      conv_op(512, 512, kernel_size=3, stride=1, padding=1), # [-1, 512, 64, 64]
      nn.ReLU(inplace=True),
    )
    self.down5 = nn.Sequential(
      # max pooling 1/2
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # [-1, 512, 32, 32]
      # conv5 block:
      conv_op(512, 1024, kernel_size=3, stride=1, padding=1), # [-1, 1024, 32, 32]
      nn.ReLU(inplace=True),
      conv_op(1024, 1024, kernel_size=3, stride=1, padding=1), # [-1, 1024, 32, 32]
      nn.ReLU(inplace=True),
    )
    
    self.up1 = nn.Sequential(
      #upSample
      nn.ConvTranspose2d(1024, 512, 2, 2), # [-1, 512, 64, 64]
    )
    self.up1conv = nn.Sequential(
      #conv block
      conv_op(1024, 512, kernel_size=3, stride=1, padding=1), # [-1, 512, 64, 64]
      nn.ReLU(inplace=True),
      conv_op(512, 512, kernel_size=3, stride=1, padding=1), # [-1, 512, 64, 64]
      nn.ReLU(inplace=True), 
    )
    self.up2 = nn.Sequential(
      #upSample
      nn.ConvTranspose2d(512, 256, 2, 2), # [-1, 256, 128, 128]
    )
    self.up2conv = nn.Sequential(
      #conv block
      conv_op(512, 256, kernel_size=3, stride=1, padding=1), # [-1, 256, 128, 128]
      nn.ReLU(inplace=True),
      conv_op(256, 256, kernel_size=3, stride=1, padding=1), # [-1, 256, 128, 128]
      nn.ReLU(inplace=True), 
    )
    self.up3 = nn.Sequential(
      #upSample
      nn.ConvTranspose2d(256, 128, 2, 2), # [-1, 128, 256, 256]
    )
    self.up3conv = nn.Sequential(
      #conv block
      conv_op(256, 128, kernel_size=3, stride=1, padding=1), # [-1, 128, 256, 256]
      nn.ReLU(inplace=True),
      conv_op(128, 128, kernel_size=3, stride=1, padding=1), # [-1, 128, 256, 256]
      nn.ReLU(inplace=True), 
    )
    self.up4 = nn.Sequential(
      #upSample
      nn.ConvTranspose2d(128, 64, 2, 2), # [-1, 64, 512, 512]
    )
    self.up4conv = nn.Sequential(
      #conv block
      conv_op(128, 64, kernel_size=3, stride=1, padding=1), # [-1, 64, 512, 512]
      nn.ReLU(inplace=True),
      conv_op(64, 64, kernel_size=3, stride=1, padding=1), # [-1, 64, 512, 512]
      nn.ReLU(inplace=True), 
    )
    self.output = nn.Sequential(
      #conv block
      conv_op(64, 1, kernel_size=1, stride=1, padding=0), # [-1, 1, 512, 512]
      nn.Sigmoid(),
    )

  def forward(self, x):

    x1 = self.down1(x)
    x2 = self.down2(x1)
    x3 = self.down3(x2)
    x4 = self.down4(x3)
    x5 = self.down5(x4)
    x6 = self.up1(x5)
    x6 = torch.cat((x4, x6), dim=1)
    x6 = self.up1conv(x6)
    x7 = self.up2(x6)
    x7 = torch.cat((x3, x7), dim=1)
    x7 = self.up2conv(x7)
    x8 = self.up3(x7)
    x8 = torch.cat((x2, x8), dim=1)
    x8 = self.up3conv(x8)
    x9 = self.up4(x8)
    x9 = torch.cat((x1, x9), dim=1)
    x9 = self.up4conv(x9)

    xOutput = self.output(x9)
    return xOutput

uNet_model = UNet