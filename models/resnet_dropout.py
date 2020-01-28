'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, p=0.2):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        
        self.dropout = nn.Dropout2d(p)

    def forward(self, x):
        out = self.dropout(F.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.dropout(F.relu(out))
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, p=0.2):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        
        self.dropout = nn.Dropout2d(p)

    def forward(self, x):
        out = self.dropout(F.relu(self.bn1(self.conv1(x))))
        out = self.dropout(F.relu(self.bn2(self.conv2(out))))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.dropout(F.relu(out))
        return out


class ResNetDropout(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, p=0.2):
        super(ResNetDropout, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, p=p)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, p=p)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, p=p)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, p=p)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        
        self.dropout = nn.Dropout2d(p)

    def _make_layer(self, block, planes, num_blocks, stride, p=0.2):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, p=p))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.dropout(F.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18Dropout(p=0.2):
    return ResNetDropout(BasicBlock, [2,2,2,2], p=p)

def ResNet34Dropout(p=0.2):
    return ResNetDropout(BasicBlock, [3,4,6,3], p=p)

def ResNet50Dropout(p=0.2):
    return ResNetDropout(Bottleneck, [3,4,6,3], p=p)

def ResNet101Dropout(p=0.2):
    return ResNetDropout(Bottleneck, [3,4,23,3], p=p)

def ResNet152Dropout(p=0.2):
    return ResNetDropout(Bottleneck, [3,8,36,3], p=p)


def test():
    net = ResNet18Dropout()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
