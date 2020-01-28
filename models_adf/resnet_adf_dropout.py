'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import logging
from contrib import adf


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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, p=0.2, keep_variance_fn=None):
        super(BasicBlock, self).__init__()
        
        self.keep_variance_fn = keep_variance_fn
        
        self.conv1 = adf.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, keep_variance_fn=self.keep_variance_fn)
        self.bn1 = adf.BatchNorm2d(planes, keep_variance_fn=self.keep_variance_fn)
        self.conv2 = adf.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, keep_variance_fn=self.keep_variance_fn)
        self.bn2 = adf.BatchNorm2d(planes, keep_variance_fn=self.keep_variance_fn)
        self.ReLU = adf.ReLU(keep_variance_fn=self.keep_variance_fn)

        self.shortcut = adf.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = adf.Sequential(
                adf.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, keep_variance_fn=self.keep_variance_fn),
                adf.BatchNorm2d(self.expansion*planes, keep_variance_fn=self.keep_variance_fn)
            )
            
        self.dropout = adf.Dropout(p=p, keep_variance_fn=self.keep_variance_fn)

    def forward(self, inputs_mean, inputs_variance):
        x = inputs_mean, inputs_variance
        
        out = self.dropout(*self.ReLU(*self.bn1(*self.conv1(*x))))
        out_mean, out_var = self.bn2(*self.conv2(*out))
        shortcut_mean, shortcut_var = self.shortcut(*x)
        out_mean, out_var = out_mean + shortcut_mean, out_var + shortcut_var
        out = out_mean, out_var 
        out = self.dropout(*self.ReLU(*out))
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, p=0.2, keep_variance_fn=None):
        super(Bottleneck, self).__init__()
        
        self.keep_variance_fn = keep_variance_fn
        
        self.conv1 = adf.Conv2d(in_planes, planes, kernel_size=1, bias=False, keep_variance_fn=self.keep_variance_fn)
        self.bn1 = adf.BatchNorm2d(planes, keep_variance_fn=self.keep_variance_fn)
        self.conv2 = adf.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, keep_variance_fn=self.keep_variance_fn)
        self.bn2 = adf.BatchNorm2d(planes, keep_variance_fn=self.keep_variance_fn)
        self.conv3 = adf.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False, keep_variance_fn=self.keep_variance_fn)
        self.bn3 = adf.BatchNorm2d(self.expansion*planes, keep_variance_fn=self.keep_variance_fn)
        self.ReLU = adf.ReLU(keep_variance_fn=self.keep_variance_fn)

        self.shortcut = adf.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = adf.Sequential(
                adf.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, keep_variance_fn=self.keep_variance_fn),
                adf.BatchNorm2d(self.expansion*planes, keep_variance_fn=self.keep_variance_fn)
            )
            
        self.dropout = adf.Dropout(p=p, keep_variance_fn=self.keep_variance_fn)

    def forward(self, inputs_mean, inputs_variance):
        x = inputs_mean, inputs_variance
        
        out = self.dropout(*self.ReLU(*self.bn1(*self.conv1(*x))))
        out = self.dropout(self.ReLU(*self.bn2(*self.conv2(*out))))
        out = self.bn3(*self.conv3(*out))
        out += self.shortcut(*x)
        out = self.dropout(*self.ReLU(*out))
        return out


class ResNetADFDropout(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, p=0.2, noise_variance=1e-3, min_variance=1e-3, initialize_msra=False):
        super(ResNetADFDropout, self).__init__()

        self.keep_variance_fn = lambda x: keep_variance(x, min_variance=min_variance)
        self._noise_variance = noise_variance

        self.in_planes = 64

        self.conv1 = adf.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False, keep_variance_fn=self.keep_variance_fn)
        self.bn1 = adf.BatchNorm2d(64, keep_variance_fn=self.keep_variance_fn)
        self.ReLU = adf.ReLU(keep_variance_fn=self.keep_variance_fn)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, p=p, keep_variance_fn=self.keep_variance_fn)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, p=p, keep_variance_fn=self.keep_variance_fn)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, p=p, keep_variance_fn=self.keep_variance_fn)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, p=p, keep_variance_fn=self.keep_variance_fn)
        self.linear = adf.Linear(512*block.expansion, num_classes, keep_variance_fn=self.keep_variance_fn)
        self.AvgPool2d = adf.AvgPool2d(keep_variance_fn=self.keep_variance_fn)
        
        self.dropout = adf.Dropout(p=p, keep_variance_fn=self.keep_variance_fn)

    def _make_layer(self, block, planes, num_blocks, stride, p=0.2, keep_variance_fn=None):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, p=p, keep_variance_fn=self.keep_variance_fn))
            self.in_planes = planes * block.expansion
        return adf.Sequential(*layers)

    def forward(self, x):
 
        inputs_mean = x
        inputs_variance = torch.zeros_like(inputs_mean) + self._noise_variance
        x = inputs_mean, inputs_variance

        out = self.dropout(*self.ReLU(*self.bn1(*self.conv1(*x))))
        out = self.layer1(*out)
        out = self.layer2(*out)
        out = self.layer3(*out)
        out = self.layer4(*out)
        out = self.AvgPool2d(*out, 4)
        out_mean = out[0].view(out[0].size(0), -1) # Flatten
        out_var = out[1].view(out[1].size(0), -1)
        out = out_mean, out_var
        out = self.linear(*out)
        return out


def ResNet18ADFDropout(p=0.2, noise_variance=1e-3, min_variance=1e-3):
    return ResNetADFDropout(BasicBlock, 
                            [2,2,2,2], 
                            num_classes=10, 
                            p=p,
                            noise_variance=noise_variance, 
                            min_variance=min_variance, 
                            initialize_msra=False)

def ResNet34ADFDropout(p=0.2, noise_variance=1e-3, min_variance=1e-3):
    return ResNetADFDropout(BasicBlock, 
                            [3,4,6,3],  
                            num_classes=10, 
                            p=p,
                            noise_variance=noise_variance, 
                            min_variance=min_variance, 
                            initialize_msra=False)

def ResNet50ADFDropout(p=0.2, noise_variance=1e-3, min_variance=1e-3):
    return ResNetADFDropout(Bottleneck, 
                            [3,4,6,3],  
                            num_classes=10, 
                            p=p,
                            noise_variance=noise_variance, 
                            min_variance=min_variance, 
                            initialize_msra=False)

def ResNet101ADFDropout(p=0.2, noise_variance=1e-3, min_variance=1e-3):
    return ResNetADFDropout(Bottleneck, 
                            [3,4,23,3],  
                            num_classes=10, 
                            p=p,
                            noise_variance=noise_variance, 
                            min_variance=min_variance, 
                            initialize_msra=False)

def ResNet152ADFDropout(p=0.2, noise_variance=1e-3, min_variance=1e-3):
    return ResNetADFDropout(Bottleneck, 
                            [3,8,36,3], 
                            num_classes=10, 
                            p=p,
                            noise_variance=noise_variance, 
                            min_variance=min_variance, 
                            initialize_msra=False)


def test():
    net = ResNet18ADFDropout()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
