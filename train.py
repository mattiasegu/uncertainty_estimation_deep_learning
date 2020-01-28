'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import argparse
import os

from contrib import adf
from models.resnet import ResNet18
from models.resnet_dropout import ResNet18Dropout
from utils import progress_bar

# Model flags
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--p', default=0.2, type=float, help='dropout rate')
parser.add_argument('--noise_variance', default=1e-3, type=float, 
                    help='noise variance')
parser.add_argument('--min_variance', default=1e-3, type=float, 
                    help='min variance')
# Training flags
parser.add_argument('--model_name', default='resnet18', type=str,  
                    help='model to train')
parser.add_argument('--resume', '-r', action='store_true', default=False, 
                    help='resume from checkpoint')
parser.add_argument('--show_bar', '-b', action='store_true', default=True, 
                    help='show bar or not')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--num_epochs', default=350, type=int, 
                    help='number of training epochs')
parser.add_argument('--batch_size', default=128, type=int, 
                    help='size of training batch')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data...')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', 
                                        train=True, 
                                        download=True, 
                                        transform=transform_train)

trainloader = torch.utils.data.DataLoader(trainset, 
                                          batch_size=args.batch_size, 
                                          shuffle=True, 
                                          num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', 
                                       train=False, 
                                       download=True, 
                                       transform=transform_test)

testloader = torch.utils.data.DataLoader(testset, 
                                         batch_size=100, 
                                         shuffle=False, 
                                         num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 
           'ship', 'truck')

# Model
print('==> Building model...')

def model_loader():
    model = {'resnet18': ResNet18,
             'resnet18_dropout': ResNet18Dropout,
             'resnet18_adf': ResNet18ADF,
             'resnet18_dropout_adf': ResNet18ADFDropout,
             }
    
    params = {'resnet18': [],
             'resnet18_dropout': [args.p],
             'resnet18_heteroscedastic': [args.p],
             'resnet18_adf': [args.noise_variance, args.min_variance],
             'resnet18_dropout_adf': [args.p, args.noise_variance, args.min_variance],
             }
    
    return model[args.model_name.lower()](*params[args.model_name.lower()])

net = model_loader().to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'

    model_to_load = args.model_name.lower()
    ckpt_path = './checkpoint/ckpt_{}.pth'.format(model_to_load)
    checkpoint = torch.load(ckpt_path)
    print('Loaded checkpoint at location {}'.format(ckpt_path))

    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

def one_hot_pred_from_label(y_pred, labels):
    y_true = torch.zeros_like(y_pred)
    ones = torch.ones_like(y_pred)
    indexes = [l for l in labels]
    y_true[torch.arange(labels.size(0)), indexes] = ones[torch.arange(labels.size(0)), indexes]
    
    return y_true

def keep_variance(x, min_variance):
        return x + min_variance


# Heteroscedastic loss
class SoftmaxHeteroscedasticLoss(torch.nn.Module):    
    def __init__(self):
        super(SoftmaxHeteroscedasticLoss, self).__init__()
        keep_variance_fn = lambda x: keep_variance(x, min_variance=args.min_variance)
        self.adf_softmax = adf.Softmax(dim=1, keep_variance_fn=keep_variance_fn)
        
    def forward(self, outputs, targets, eps=1e-5):
        mean, var = self.adf_softmax(*outputs)
        targets = one_hot_pred_from_label(mean, targets)
        
        precision = 1/(var + eps)
        return torch.mean(0.5*precision * (targets-mean)**2 + 0.5*torch.log(var+eps))

if args.model_name.lower().endswith('adf'):
    criterion = SoftmaxHeteroscedasticLoss()
else:
    criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 150], gamma=0.1, last_epoch=-1)

# Training
def train(epoch, net):
    print('\nEpoch: {} ==> lr: {}'.format(epoch, scheduler.get_last_lr()))
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        if args.model_name.lower().endswith('adf'):
            outputs_mean, outputs_var = outputs
            loss = criterion(outputs, targets)
            outputs_mean, _ = outputs
        else:
            outputs_mean = outputs
            loss = criterion(outputs_mean, targets)

        # print(loss)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs_mean.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if args.show_bar:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch, net):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs) 
            if args.model_name.lower().endswith('adf'):
                outputs_mean, outputs_var = outputs
                loss = criterion(outputs, targets)
                outputs_mean, _ = outputs
            else:
                outputs_mean = outputs
                loss = criterion(outputs_mean, targets)

            test_loss += loss.item()
            _, predicted = outputs_mean.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if args.show_bar:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('\nSaving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
            
        torch.save(state, './checkpoint/ckpt_{}.pth'.format(args.model_name))
        best_acc = acc

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('\nSaving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
            
        torch.save(state, './checkpoint/ckpt_{}.pth'.format(args.model_name))
        best_acc = acc
        
print('==> Training parameters:')
print('        start_epoch = {}'.format(start_epoch+1))
print('        best_acc    = {}'.format(best_acc))
print('        lr @epoch=0 = {}'.format(args.lr))
print('==> Starting training...')

for epoch in range(0, args.num_epochs):
    if epoch>start_epoch:
        train(epoch, net)
        test(epoch, net)
    scheduler.step()
