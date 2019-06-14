#!/usr/bin/env python
# coding: utf-8

import torch.nn as nn
import numpy as np
from opts import parser

FLAGS = parser.parse_args()

def init_kernel(m):
    if isinstance(m, nn.Conv2d): 
        # Initialize kernels of Conv2d layers as kaiming normal
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        # Initialize biases of Conv2d layers at 0
        nn.init.zeros_(m.bias)
       
class resnet8(nn.Module):
    """
    Define model architecture.
    
    # Arguments
       img_channels: Number of channels in target image
       img_width: Target image widht.
       img_height: Target image height.
       output_dim: Dimension of model output.
       
    """

    def __init__(self, img_channels, output_dim):
        super(resnet8, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=img_channels,out_channels=32, 
                      kernel_size=[5,5], stride=[2,2], padding=[5//2,5//2]),
            nn.MaxPool2d(kernel_size=2))
        
        self.residual_block_1a = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=32, kernel_size=[3,3], 
                      stride=[2,2], padding=[3//2,3//2]), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=32, kernel_size=[3,3], 
                      padding=[3//2,3//2]))
        
        self.parallel_conv_1 = nn.Conv2d(in_channels=32,out_channels=32, 
                                         kernel_size=[1,1], stride=[2,2], 
                                         padding=[1//2,1//2])
        
        self.residual_block_2a = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=64, kernel_size=[3,3], 
                      stride=[2,2], padding=[3//2,3//2]), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=64, kernel_size=[3,3], 
                      padding=[3//2,3//2]))
        
        

        self.parallel_conv_2 = nn.Conv2d(in_channels=32,out_channels=64, 
                                         kernel_size=[1,1], stride=[2,2], 
                                         padding=[1//2,1//2])
        
        self.residual_block_3a = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=128, kernel_size=[3,3], 
                      stride=[2,2], padding=[3//2,3//2]), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=128, kernel_size=[3,3], 
                      padding=[3//2,3//2]))
        
        

        self.parallel_conv_3 = nn.Conv2d(in_channels=64,out_channels=128, 
                                         kernel_size=[1,1], stride=[2,2], 
                                         padding=[1//2,1//2])
        
        self.output_dim = output_dim

        self.last_block = nn.Sequential(
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Linear(6272,self.output_dim))
        
        # Initialize layers exactly as in Keras
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)    
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize kernels of Conv2d layers as kaiming normal
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        self.residual_block_1a.apply(init_kernel)
        self.residual_block_2a.apply(init_kernel)
        self.residual_block_3a.apply(init_kernel)

                
    def forward(self, x):
        x1 = self.layer1(x)
        # First residual block
        x2 = self.residual_block_1a(x1)
        x1 = self.parallel_conv_1(x1)
        x3 = x1.add(x2)
        # Second residual block
        x4 = self.residual_block_2a(x3)
        x3 = self.parallel_conv_2(x3)
        x5 = x3.add(x4)
        # Third residual block
        x6 = self.residual_block_3a(x5)
        x5 = self.parallel_conv_3(x5)
        x7 = x5.add(x6)
        
        out = x7.view(x7.size(0), -1) # Flatten
        out = self.last_block(out)
        
        return out


class resnet8_MCDO(nn.Module):
    """
    Define model architecture.
    
    # Arguments
       img_channels: Number of channels in target image
       img_width: Target image widht.
       img_height: Target image height.
       output_dim: Dimension of model output.

    Dropout is here applied after every convolutional layer, 
    not only after inner-product ones. Dropout will be enabled at test
    time. As mentioned by Gal, place Dropout after conv layers and before 
    MaxPool.
       
    """

    def __init__(self, img_channels, in_height, in_width, output_dim):
        super(resnet8_MCDO, self).__init__()
        

        p = FLAGS.dropout

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=img_channels,out_channels=32, 
                      kernel_size=[5,5], stride=[2,2], padding=[5//2,5//2]),
            nn.Dropout2d(p=p),
            nn.MaxPool2d(kernel_size=2))
        
        self.residual_block_1a = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=32, kernel_size=[3,3], 
                      stride=[2,2], padding=[3//2,3//2]),
            nn.Dropout2d(p=p),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=32, kernel_size=[3,3], 
                      padding=[3//2,3//2]),
            nn.Dropout2d(p=p))
        
        self.parallel_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=32, kernel_size=[1,1], 
                      stride=[2,2], padding=[1//2,1//2]),
            nn.Dropout2d(p=p))
        
        self.residual_block_2a = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=64, kernel_size=[3,3], 
                      stride=[2,2], padding=[3//2,3//2]), 
            nn.Dropout2d(p=p),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=64, kernel_size=[3,3], 
                      padding=[3//2,3//2]),
            nn.Dropout2d(p=p))
        
        self.parallel_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64, kernel_size=[1,1], 
                      stride=[2,2], padding=[1//2,1//2]),
            nn.Dropout2d(p=p))
        
        self.residual_block_3a = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=128, kernel_size=[3,3], 
                      stride=[2,2], padding=[3//2,3//2]), 
            nn.Dropout2d(p=p),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=128, kernel_size=[3,3], 
                      padding=[3//2,3//2]),
            nn.Dropout2d(p=p))
        
        self.parallel_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128, kernel_size=[1,1], 
                      stride=[2,2], padding=[1//2,1//2]),
            nn.Dropout2d(p=p))
        
        self.output_dim = output_dim

        self.last_block = nn.Sequential(
            nn.ReLU(),
            nn.Linear(6272,self.output_dim))
        
        # Initialize layers exactly as in Keras
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)    
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize kernels of Conv2d layers as kaiming normal
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        self.residual_block_1a.apply(init_kernel)
        self.residual_block_2a.apply(init_kernel)
        self.residual_block_3a.apply(init_kernel)

    def forward(self, x):
        x1 = self.layer1(x)
        # First residual block
        x2 = self.residual_block_1a(x1)
        x1 = self.parallel_conv_1(x1)
        x3 = x1.add(x2)
        # Second residual block
        x4 = self.residual_block_2a(x3)
        x3 = self.parallel_conv_2(x3)
        x5 = x3.add(x4)
        # Third residual block
        x6 = self.residual_block_3a(x5)
        x5 = self.parallel_conv_3(x5)
        x7 = x5.add(x6)
        
        out = x7.view(x7.size(0), -1) # Flatten
        # We model the network to learn also log var
        out = self.last_block(out)
        
        return out


class resnet8_MCDO_ale(nn.Module):
    """
    Define model architecture.
    
    # Arguments
       img_channels: Number of channels in target image
       img_width: Target image widht.
       img_height: Target image height.
       output_dim: Dimension of model output.

    Dropout is here applied after every convolutional layer, 
    not only after inner-product ones. Dropout will be enabled at test
    time. As mentioned by Gal, place Dropout after conv layers and before 
    MaxPool.
       
    """

    def __init__(self, img_channels, in_height, in_width, output_dim):
        super(resnet8_MCDO_ale, self).__init__()
        

        p = FLAGS.dropout

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=img_channels,out_channels=32, 
                      kernel_size=[5,5], stride=[2,2], padding=[5//2,5//2]),
            nn.Dropout2d(p=p),
            nn.MaxPool2d(kernel_size=2))
        
        self.residual_block_1a = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=32, kernel_size=[3,3], 
                      stride=[2,2], padding=[3//2,3//2]),
            nn.Dropout2d(p=p),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=32, kernel_size=[3,3], 
                      padding=[3//2,3//2]),
            nn.Dropout2d(p=p))
        
        self.parallel_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=32, kernel_size=[1,1], 
                      stride=[2,2], padding=[1//2,1//2]),
            nn.Dropout2d(p=p))
        
        self.residual_block_2a = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=64, kernel_size=[3,3], 
                      stride=[2,2], padding=[3//2,3//2]), 
            nn.Dropout2d(p=p),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=64, kernel_size=[3,3], 
                      padding=[3//2,3//2]),
            nn.Dropout2d(p=p))
        
        self.parallel_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64, kernel_size=[1,1], 
                      stride=[2,2], padding=[1//2,1//2]),
            nn.Dropout2d(p=p))
        
        self.residual_block_3a = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=128, kernel_size=[3,3], 
                      stride=[2,2], padding=[3//2,3//2]), 
            nn.Dropout2d(p=p),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=128, kernel_size=[3,3], 
                      padding=[3//2,3//2]),
            nn.Dropout2d(p=p))
        
        self.parallel_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128, kernel_size=[1,1], 
                      stride=[2,2], padding=[1//2,1//2]),
            nn.Dropout2d(p=p))
        
        self.output_dim = output_dim

        self.last_block_mean = nn.Sequential(
            nn.ReLU(),
            nn.Linear(6272,self.output_dim))
        
        self.last_block_var = nn.Sequential(
            nn.ReLU(),
            nn.Linear(6272,self.output_dim))
        
        # Initialize layers exactly as in Keras
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)    
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize kernels of Conv2d layers as kaiming normal
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        self.residual_block_1a.apply(init_kernel)
        self.residual_block_2a.apply(init_kernel)
        self.residual_block_3a.apply(init_kernel)

                
    def forward(self, x):
        x1 = self.layer1(x)
        # First residual block
        x2 = self.residual_block_1a(x1)
        x1 = self.parallel_conv_1(x1)
        x3 = x1.add(x2)
        # Second residual block
        x4 = self.residual_block_2a(x3)
        x3 = self.parallel_conv_2(x3)
        x5 = x3.add(x4)
        # Third residual block
        x6 = self.residual_block_3a(x5)
        x5 = self.parallel_conv_3(x5)
        x7 = x5.add(x6)
        
        out = x7.view(x7.size(0), -1) # Flatten
        # We model the network to learn also log var
        out_mean = self.last_block_mean(out)
        out_log_var = self.last_block_var(out)
        
        out = {'mean': out_mean,
               'log_var': out_log_var}

        return out

