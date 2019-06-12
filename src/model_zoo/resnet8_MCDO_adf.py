from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch
import logging
import numpy as np
from contrib import adf
from opts import parser

FLAGS = parser.parse_args()


def keep_variance(x, min_variance):
    return x + min_variance


def finitialize_msra(modules, small=False):
    logging.info("Initializing MSRA")
    for layer in modules:
        if isinstance(layer, adf.Conv2d) or isinstance(layer, adf.Linear):  # convolution: bias=0, weight=msra
            nn.init.kaiming_normal_(layer.weight)
            if small:
                layer.weight.data.mul_(0.001)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)


def finitialize_xavier(modules, small=False):
    logging.info("Initializing Xavier")
    for layer in modules:
        if isinstance(layer, adf.Conv2d) or isinstance(layer, adf.Linear):  # convolution: bias=0, weight=msra
            nn.init.xavier_normal_(layer.weight)
            if small:
                layer.weight.data.mul_(0.001)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

class Resnet8_MCDO_adf(nn.Module):

    def __init__(self, img_channels, output_dim, noise_variance=1e-3, min_variance=1e-3, initialize_msra=False):
        super(Resnet8_MCDO_adf, self).__init__()
        
        p = FLAGS.dropout
        self._keep_variance_fn = lambda x: keep_variance(x, min_variance=min_variance)
        self._noise_variance = noise_variance
        self.layer1 = adf.Sequential(
            adf.Conv2d(
                in_channels=img_channels, out_channels=32, kernel_size=5, padding=5//2,
                stride=2, bias=True, keep_variance_fn=self._keep_variance_fn),
            adf.Dropout(p, keep_variance_fn=self._keep_variance_fn),
            adf.MaxPool2d(keep_variance_fn=self._keep_variance_fn))
        
        self.residual_block_1a = adf.Sequential(
            adf.BatchNorm2d(32),
            adf.ReLU(),
            adf.Conv2d(
                in_channels=32, out_channels=32, kernel_size=3, padding=3//2,
                stride=2, bias=True, keep_variance_fn=self._keep_variance_fn),
            adf.Dropout(p, keep_variance_fn=self._keep_variance_fn),
            adf.BatchNorm2d(32),
            adf.ReLU(),
            adf.Conv2d(
                in_channels=32, out_channels=32, kernel_size=3, padding=3//2,
                bias=True, keep_variance_fn=self._keep_variance_fn),
            adf.Dropout(p, keep_variance_fn=self._keep_variance_fn))
        
        self.parallel_conv_1 = adf.Sequential(
            adf.Conv2d(
                in_channels=32, out_channels=32, kernel_size=1, padding=1//2,
                stride=2, bias=True, keep_variance_fn=self._keep_variance_fn),
            adf.Dropout(p, keep_variance_fn=self._keep_variance_fn))
        
        self.residual_block_2a = adf.Sequential(
            adf.BatchNorm2d(32),
            adf.ReLU(),
            adf.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, padding=3//2,
                stride=2, bias=True, keep_variance_fn=self._keep_variance_fn),
            adf.Dropout(p, keep_variance_fn=self._keep_variance_fn),
            adf.BatchNorm2d(64),
            adf.ReLU(),
            adf.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, padding=3//2,
                bias=True, keep_variance_fn=self._keep_variance_fn),
            adf.Dropout(p, keep_variance_fn=self._keep_variance_fn))
        
        self.parallel_conv_2 = adf.Sequential(
            adf.Conv2d(
                in_channels=32, out_channels=64, kernel_size=1, padding=1//2,
                stride=2, bias=True, keep_variance_fn=self._keep_variance_fn),
            adf.Dropout(p, keep_variance_fn=self._keep_variance_fn))
        
        self.residual_block_3a = adf.Sequential(
            adf.BatchNorm2d(64),
            adf.ReLU(),
            adf.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, padding=3//2,
                stride=2, bias=True, keep_variance_fn=self._keep_variance_fn),
            adf.Dropout(p, keep_variance_fn=self._keep_variance_fn),
            adf.BatchNorm2d(128),
            adf.ReLU(),
            adf.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, padding=3//2,
                bias=True, keep_variance_fn=self._keep_variance_fn),
            adf.Dropout(p, keep_variance_fn=self._keep_variance_fn))
        
        self.parallel_conv_3 = adf.Sequential(
            adf.Conv2d(
                in_channels=64, out_channels=128, kernel_size=1, padding=1//2,
                stride=2, bias=True, keep_variance_fn=self._keep_variance_fn),
            adf.Dropout(p, keep_variance_fn=self._keep_variance_fn))
        
        self.output_dim = output_dim

        self.last_block = adf.Sequential(
            adf.ReLU(),
            adf.Linear(6272,self.output_dim))
        
        # Initialize layers exactly as in Keras
        for layer in self.modules():
            if isinstance(layer, adf.Conv2d) or isinstance(layer, adf.Linear):  # convolution: bias=0, weight=msra
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, adf.BatchNorm2d):  
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)
                

                
    def forward(self, x):
        
        inputs_mean = x
        inputs_variance = torch.zeros_like(inputs_mean) + self._noise_variance
        x = inputs_mean, inputs_variance
        x1 = self.layer1(*x)
        # First residual block
        x2 = self.residual_block_1a(*x1)
        x1 = self.parallel_conv_1(*x1)
        x3_mean = x1[0].add(x2[0])
        x3_var = x1[1].add(x2[1])
        x3 = x3_mean, x3_var
        # Second residual block
        x4 = self.residual_block_2a(*x3)
        x3 = self.parallel_conv_2(*x3)
        x5_mean = x3[0].add(x4[0])
        x5_var = x3[1].add(x4[1])
        x5 = x5_mean, x5_var
        # Third residual block
        x6 = self.residual_block_3a(*x5)
        x5 = self.parallel_conv_3(*x5)
        x7_mean = x5[0].add(x6[0])
        x7_var = x5[1].add(x6[1])
        x7 = x7_mean, x7_var
        
        out_mean = x7[0].view(x7[0].size(0), -1) # Flatten
        out_var = x7[1].view(x7[1].size(0), -1)
        out = out_mean, out_var
        out = self.last_block(*out)
        
        return out

