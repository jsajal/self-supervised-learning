from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# code modified from pytorch imagenet example

# python imports
import argparse
import os
import time
import math

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

# for visualization
from tensorboardX import SummaryWriter

from model import default_model

from data_loader import MelanomaDataLoader

# the arg parser
parser = argparse.ArgumentParser(description='PyTorch Image Classification')
parser.add_argument('data_folder', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup-epochs', default=0, type=int,
                    help='number of epochs for warmup')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-a', '--attack', dest='attack', action='store_true',
                    help='Attack with adersarial samples on validation set')
parser.add_argument('-v', '--vis', dest='vis', action='store_true',
                    help='Visualize the attention map')
parser.add_argument('--use-custom-conv', action='store_true',
                    help='Use custom convolution')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU ID to use.')

# tensorboard writer
writer = SummaryWriter('../logs')

# main function for training and testing
def main(args):
  # parse args
  best_acc1 = 0.0

  if args.gpu >= 0:
    print("Use GPU: {}".format(args.gpu))
  else:
    print('You are using CPU for computing!',
          'Yet we assume you are using a GPU.',
          'You will NOT be able to switch between CPU and GPU training!')

  train_dataset = MelanomaDataLoader(args.data_folder,
  	                                         split="train")
  val_dataset = MelanomaDataLoader(args.data_folder,
  	                                         split="val")
  test_dataset = MelanomaDataLoader(args.data_folder,
  	                                         split="test")

  train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False)
  val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=100, shuffle=False,
    num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False)
  test_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=100, shuffle=False,
    num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False)
