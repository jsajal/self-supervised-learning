from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
from PIL import Image
import numpy as np

def load_image(path):
  # load an image
  img = cv2.imread(path)
  img = img[:, :, ::-1]  # BGR -> RGB
  return img

def save_image(path, img):
  img = img.copy()[:,:,::-1]
  return cv2.imwrite(path, img)