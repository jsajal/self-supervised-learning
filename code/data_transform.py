from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math
import random
import numpy as np
import torch

import cv2
import numbers
import collections

from utils import resize_image

# default list of interpolations
_DEFAULT_INTERPOLATIONS = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC]

#################################################################################
# Solution for HW 1
#################################################################################

class Compose(object):
  """Composes several transforms together.

  Args:
      transforms (list of ``Transform`` objects): list of transforms to compose.

  Example:
      >>> Compose([
      >>>     Scale(320),
      >>>     RandomSizedCrop(224),
      >>> ])
  """
  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, img):
    for t in self.transforms:
      img = t(img)
    return img

  def __repr__(self):
    repr_str = ""
    for t in self.transforms:
      repr_str += t.__repr__() + '\n'
    return repr_str

class RandomHorizontalFlip(object):
  """Horizontally flip the given numpy array randomly
     (with a probability of 0.5).
  """
  def __call__(self, img):
    """
    Args:
        img (numpy array): Image to be flipped.

    Returns:
        numpy array: Randomly flipped image
    """
    if random.random() < 0.5:
      img = cv2.flip(img, 1)
    return img

  def __repr__(self):
    return "Random Horizontal Flip"

#################################################################################
# You will need to fill in the missing code in these classes
#################################################################################
class Scale(object):
  """Rescale the input numpy array to the given size.

  Args:
      size (sequence or int): Desired output size. If size is a sequence like
          (w, h), output size will be matched to this. If size is an int,
          smaller edge of the image will be matched to this number.
          i.e, if height > width, then image will be rescaled to
          (size, size * height / width)

      interpolations (list of int, optional): Desired interpolation.
      Default is ``CV2.INTER_NEAREST|CV2.INTER_LANCZOS|CV2.INTER_LINEAR|CV2.INTER_CUBIC``
      Pass None during testing: always use CV2.INTER_LINEAR
  """
  def __init__(self, size, interpolations=_DEFAULT_INTERPOLATIONS):
    assert (isinstance(size, int)
            or (isinstance(size, collections.Iterable)
                and len(size) == 2)
           )
    self.size = size
    # use bilinear if interpolation is not specified
    if interpolations is None:
      interpolations = [cv2.INTER_LINEAR]
    assert isinstance(interpolations, collections.Iterable)
    self.interpolations = interpolations

  def __call__(self, img):
    """
    Args:
        img (numpy array): Image to be scaled.

    Returns:
        numpy array: Rescaled image
    """
    # sample interpolation method
    interpolation = random.sample(self.interpolations, 1)[0]

    # scale the image
    if isinstance(self.size, int):
      h, w = img.shape[0], img.shape[1]
      if (w <= h and w == self.size) or (h <= w and h == self.size):
        return img
      if w < h:
        ow = int(self.size)
        oh = int(round(self.size * h / w))
        img = resize_image(img, (ow, oh), interpolation=interpolation)
      else:
        oh = int(self.size)
        ow = int(round(self.size * w / h))
        img = resize_image(img, (ow, oh), interpolation=interpolation)
      return img
    else:
      #################################################################################
      # Solution
      #################################################################################
      img = resize_image(img, self.size, interpolation=interpolation)
      return img

  def __repr__(self):
    if isinstance(self.size, int):
      return "Scale [Shortest side {:d}]".format(self.size)
    else:
      target_size = self.size
      return "Scale [Exact Size ({:d}, {:d})]".format(target_size[0], target_size[1])


#################################################################################
# Additional helper functions
#################################################################################
class ToTensor(object):
  """Convert a ``numpy.ndarray`` image pair to tensor.

  Converts a numpy.ndarray (H x W x C) image in the range
  [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
  """
  def __call__(self, img):
    assert isinstance(img, np.ndarray)
    # make ndarray normal
    img = img.copy()
    # convert image to tensor
    if img.ndim == 2:
      img = img[:, :, None]

    tensor_img = torch.from_numpy(img.transpose((2, 0, 1)))
    # backward compatibility
    if isinstance(tensor_img, torch.ByteTensor):
      return tensor_img.float().div(255)
    else:
      return tensor_img

  def __repr__(self):
    return "To Tensor()"

class Normalize(object):
  """Normalize an tensor image with mean and standard deviation.

  Given mean: (R, G, B) and std: (R, G, B),
  will normalize each channel of the torch.*Tensor, i.e.
  channel = (channel - mean) / std

  Args:
      mean (sequence): Sequence of means for R, G, B channels respecitvely.
      std (sequence): Sequence of standard deviations for R, G, B channels
        respecitvely.
  """
  def __init__(self, mean, std, scale=1.0):
    self.mean = mean
    self.std = std
    self.scale = scale

  def __call__(self, tensor_img):
    # multiply scale -> subtract mean (per channel) -> divide by std (per channel)
    tensor_img.mul_(self.scale)
    for t, m, s in zip(tensor_img, self.mean, self.std):
      t.sub_(m).div_(s)
    return tensor_img

  def __repr__(self):
    return "Normalize" + '(mean={0}, std={1})'.format(self.mean, self.std)

class Denormalize(object):
  """De-normalize an tensor image with mean and standard deviation.

  Given mean: (R, G, B) and std: (R, G, B),
  will normalize each channel of the torch.*Tensor, i.e.
  channel = channel * std + mean

  Args:
      mean (sequence): Sequence of means for R, G, B channels respecitvely.
      std (sequence): Sequence of standard deviations for R, G, B channels
        respecitvely.
  """
  def __init__(self, mean, std, scale=1.0):
    self.mean = mean
    self.std = std
    self.scale = scale

  def __call__(self, tensor_img):
    # multiply by std (per channel) -> add mean (per channel) -> divide by scale
    for t, m, s in zip(tensor_img, self.mean, self.std):
      t.mul_(s).add_(m)
    tensor_img.div_(self.scale)
    return tensor_img

  def __repr__(self):
    return "De-normalize" + '(mean={0}, std={1})'.format(self.mean, self.std)
