from __future__ import division

from models import *
from utils import *
from datasets import *
from torchvision import transforms
import os
import sys


import torch
from torchvision import datasets
from torch.autograd import Variable
import numpy as np



class YoloDetector:
    def __init__(self, model, conf_thres = 0.8, nms_thres = 0.4):
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.model = model
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.transform = transforms.Compose([transforms.ToTensor()])
        
        
    def get_detections(self, img):
        # img = np.asarray(img)
        image_tensor = Variable(img.type(self.Tensor))
        # image_tensor = self.transform(img)[None, ...]
        with torch.no_grad():
            detections = self.model(image_tensor)
            detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)
        return detections



