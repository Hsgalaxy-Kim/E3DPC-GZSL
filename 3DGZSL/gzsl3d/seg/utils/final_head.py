import torch
import torch.nn as nn
import torch.nn.functional as F
from gzsl3d.kpconv.models.blocks import *

class Final_Head(nn.Module):
    def __init__(self, type='scannet', num_classes = 21, unseen_list = [0,5,7,8,11]):
        super(Final_Head, self).__init__()
        self.num_classes = num_classes
        self.unseen_list = unseen_list
        if type == 'scannet':
            self.fcout_gen = nn.Conv1d(64, num_classes, 1)
        if type == 's3dis':
            self.fcout_gen = nn.Linear(128, num_classes)
        if type == 'semantickitti':
            self.fcout_gen = nn.Sequential(
                UnaryBlock(128, 128, False, 0),
                UnaryBlock(128, num_classes, False, 0)
            )

    def forward(self, x):
        out1 = self.fcout_gen(x)
        
        return out1