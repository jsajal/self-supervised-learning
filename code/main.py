from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# code modified from pytorch imagenet example
import imp
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


from data_loader import MelanomaDataLoader

import data_transform as transforms

from utils import AverageMeter

from inception import default_model
import numpy as np
parser = argparse.ArgumentParser(description='PyTorch Image Classification')
parser.add_argument('data_folder', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--mode', default='preTrain', type=str,
                    help='determine mode of training , either self-supervised (preTrain) or fully supervised (supTrain)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 1)')
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
writer = SummaryWriter('../logs')


def rotate_img(img, rot):
  if rot == 0:  # 0 degrees rotation
    return img
  elif rot == 90:

      return torch.from_numpy(np.flipud(np.transpose(img, (0, 1, 3, 2))).copy())

  elif rot == 180:  # 90 degrees rotation
    return torch.from_numpy(np.fliplr(np.flipud(img)).copy())
  elif rot == 270:
    return torch.from_numpy(np.transpose(np.flipud(img).copy(), (0, 1, 3, 2)))
  #   return torch.flip(torch.t(img),(1,2))
  else:
    raise ValueError('rotation should be 0, 90, 180, or 270 degrees')


# main function for training and testing
def main(args):
  # parse args
  best_acc1 = 1000000#0.0

  if args.gpu >= 0:
    print("Use GPU: {}".format(args.gpu))
  else:
    print('You are using CPU for computing!',
          'Yet we assume you are using a GPU.',
          'You will NOT be able to switch between CPU and GPU training!')


  print('Loading training, validation and test dataset......')

  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  # train transforms
  print('Loading training, validation and test dataset......')
  train_transforms = []
  train_transforms.append(transforms.Scale((256, 256)))
  train_transforms.append(transforms.ToTensor())
  train_transforms.append(normalize)

  train_transforms = transforms.Compose(train_transforms)
  # val transforms
  val_transforms = []
  val_transforms.append(transforms.Scale((256, 256)))
  val_transforms.append(transforms.ToTensor())
  val_transforms.append(normalize)
  val_transforms = transforms.Compose(val_transforms)



  # test transforms
  # test_transforms=[]
  # test_transforms.append(transforms.Scale((224, 224), interpolations=None))
  # test_transforms.append(transforms.ToTensor())
  # transforms.Normalize(mean=mean_pix, std=std_pix)
  # test_transforms = transforms.Compose(test_transforms)

  train_dataset = MelanomaDataLoader(args.data_folder,
  	                                         split="train", img_transforms=train_transforms)
  val_dataset = MelanomaDataLoader(args.data_folder,
  	                                         split="val", img_transforms=val_transforms)
  # test_dataset = MelanomaDataLoader(args.data_folder,
  # 	                                         split="test", transforms=test_transforms)



  #forrotnet



  train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.num_workers, pin_memory=True, sampler=None, drop_last=True)

  val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, shuffle=False,
    num_workers=args.num_workers, pin_memory=True, sampler=None, drop_last=False)
  # dataloader = DataLoader(train_dataset, batch_size=args.batch_size, unsupervised=True)
  # #testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, unsupervised=True)
  # valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, unsupervised=True)
  # test_loader = torch.utils.data.DataLoader(
  #   val_dataset, batch_size=1, shuffle=False,
  #   num_workers=args.num_workers, pin_memory=True, sampler=None, drop_last=False)



  if args.mode == "preTrain": # Train in self supervised mode
    model = default_model()
    #preTrain_model()
    criterion = nn.CrossEntropyLoss()
  # else: # Train in fully supervised mode
  #   model = preTrain_model()
  #   criterion = nn.CrossEntropyLoss()
  model_arch = "Inception"

  # put everthing to gpu
  if args.gpu >= 0:
    model.cuda(args.gpu)
    criterion.cuda(args.gpu)

  # setup the optimizer
  optimizer = torch.optim.SGD(model.parameters(), args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay)

  # enable cudnn benchmark
  cudnn.enabled = True
  cudnn.benchmark = True

  # start the training
  print("Training the model ...")
  for epoch in range(args.start_epoch, args.epochs):
    # train for one epoch
    train(train_loader, model, criterion, optimizer, epoch, "train", args)

    # evaluate on validation set
    #acc1 = validate(val_loader, model, epoch, args)
    val_loss = validate(val_loader, model, criterion, epoch, args)

    # remember best acc@1 and save checkpoint
    is_best = val_loss < best_acc1
    best_acc1 = min(val_loss, best_acc1)
    save_checkpoint({
      'epoch': epoch + 1,
      'model_arch': model_arch,
      'state_dict': model.state_dict(),
      'best_acc1': best_acc1,
      'optimizer' : optimizer.state_dict(),
    }, is_best)


def save_checkpoint(state, is_best,
                    file_folder="../models/", filename='checkpoint_preTrain.pth.tar'):
  """save checkpoint"""
  if not os.path.exists(file_folder):
    os.mkdir(file_folder)
  torch.save(state, os.path.join(file_folder, filename))
  if is_best:
    # skip the optimization state
    state.pop('optimizer', None)
    torch.save(state, os.path.join(file_folder, 'model_best_preTrain.pth.tar'))

def train(train_loader, model, criterion, optimizer, epoch, stage, args):
  """Training the model"""
  assert stage in ["train"]
  # adjust the learning rate
  num_iters = len(train_loader)
  #print("len",num_iters)
  lr = 0.0

  # set up meters
  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  acc = AverageMeter()

  # switch to train mode
  model.train()

  end = time.time()
  for i, (input, _) in enumerate(train_loader):
    # adjust the learning rate
    # cosine learning rate decay

    lr = 0.5 * args.lr * (1 + math.cos(
    	(epoch * num_iters + i) / float(args.epochs * num_iters) * math.pi))
    for param_group in optimizer.param_groups:
    	param_group['lr'] = lr
    	param_group['weight_decay'] = args.weight_decay

    # measure data loading time
    data_time.update(time.time() - end)


    rotated_imgs=[rotate_img(input, 0),
            rotate_img(input, 90),
            rotate_img(input, 180),
            rotate_img(input, 270)]


    rotation_labels = torch.LongTensor([0, 1, 2, 3])
    input=torch.cat(rotated_imgs, dim=0)
    target= rotation_labels
    if args.gpu >= 0:
    	input = input.cuda(args.gpu, non_blocking=True)
    	target = target.cuda(args.gpu, non_blocking=True)
 #   print("target", target[0])
  #  print("input", input[1])
    # compute output
    #print("target",target)
    output = model(input)
    loss = criterion(output, target)

    # measure accuracy and record loss
    #acc1, acc5 = accuracy(output, target, topk=(1, 5))
    losses.update(loss.item(), input.size(0))
    #top1.update(acc1[0], input.size(0))
    #top5.update(acc5[0], input.size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    # printing
    if i % args.print_freq == 0:
      print('Epoch: [{0}][{1}/{2}]\t'
        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
         epoch + 1, i, len(train_loader), batch_time=batch_time,
         data_time=data_time, loss=losses))
      # log loss / lr
      if stage == "train":
        writer.add_scalar('data/training_loss',
          losses.val, epoch * num_iters + i)
        writer.add_scalar('data/learning_rate',
        lr, epoch * num_iters + i)

  print("*************************************************************************")


  # print the learning rate
  print("[Stage {:s}]: Epoch {:d} finished with lr={:f}".format(
            stage, epoch + 1, lr))
  # log top-1/5 acc
  #writer.add_scalars('data/top1_accuracy',
  #  {"train" : top1.avg}, epoch + 1)
  #writer.add_scalars('data/top5_accuracy',
  #  {"train" : top5.avg}, epoch + 1)

def validate(val_loader, model, criterion, epoch, args, attacker=None, visualizer=None):
  """Test the model on the validation set"""
  batch_time = AverageMeter()
  losses = AverageMeter()

  # switch to evaluate mode (autograd will still track the graph!)
  model.eval()

  # disable/enable gradients
  grad_flag = (attacker is not None) or (visualizer is not None)
  with torch.set_grad_enabled(grad_flag):
    end = time.time()
    # loop over validation set
    for i, (input, target) in enumerate(val_loader):
      #self supervised mode
      # if args.mode == "preTrain":
      # 	target = input
      rotated_imgs = [rotate_img(input, 0),
                      rotate_img[input,90],
                      rotate_img(input, 180),
                      rotate_img(input, 270)]

      rotation_labels = torch.LongTensor([0, 1, 2, 3])
      input = torch.cat(rotated_imgs, dim=0)
      target = rotation_labels
      if args.gpu >= 0:
        input = input.cuda(args.gpu, non_blocking=False)
        target = target.cuda(args.gpu, non_blocking=False)

      # forward the model
      output = model(input)
      loss = criterion(output, target)

      # measure accuracy and record loss
      #acc = accuracy(output, target)
      losses.update(loss, input.size(0))

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      # printing
      if i % args.print_freq == 0:
        print('Test: [{0}/{1}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'loss {losses.val:.3f} ({losses.avg:.3f})\t'.format(
           i, len(val_loader), batch_time=batch_time,
           losses=losses))

        # visualize the results
        #if args.vis and args.evaluate:
        #  vis_output = visualizer.explain(model, input)
        #  vis_output = default_visfunction(input, vis_output=vis_output)
        #  writer.add_image("Image/Image_Atten", vis_output, i)

  print('******Loss on val/test set = {losses.avg:.3f}'.format(
            losses=losses))

  if (not args.evaluate):
    # log top-1/5 acc
    writer.add_scalars('data/val_preTrainloss',
      {"val_preTrainloss" : losses.avg}, epoch + 1)

  return losses.avg


if __name__ == '__main__':

  args = parser.parse_args()
  main(args)
