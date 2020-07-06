'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import argparse
import os
import time

from contrib import adf

from models.resnet import ResNet18
from models.resnet_dropout import ResNet18Dropout
from models_adf.resnet_adf import ResNet18ADF
from models_adf.resnet_adf_dropout import ResNet18ADFDropout
from utils import progress_bar

# Model flags
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--p', default=0.2, type=float, help='dropout rate')
parser.add_argument('--num_samples', default=10, type=int, help='number of samples to collect with Monte Carlo dropout')
parser.add_argument('--noise_variance', default=1e-4, type=float, 
                    help='noise variance')
parser.add_argument('--min_variance', default=1e-4, type=float, 
                    help='min variance')
parser.add_argument('--tau', default=1e-4, type=float, 
                    help='constant data variance for Monte Carlo dropout.')

# Testing flags
parser.add_argument('--load_model_name', default='resnet18_dropout', type=str,  
                    help='model to load')
parser.add_argument('--test_model_name', default='resnet18_dropout_adf', type=str,  
                    help='model to test')
parser.add_argument('--resume', '-r', action='store_true', default=True, 
                    help='resume from checkpoint')
parser.add_argument('--show_bar', '-b', action='store_true', default=True, 
                    help='show bar or not')
parser.add_argument('--verbose', '-v', action='store_true', default=True, 
                    help='regulate output verbosity')
parser.add_argument('--use_mcdo', '-m', action='store_true', default=False,  
                    help='use Monte Carlo dropout to compute predictions and'
                    'model uncertainty estimates.')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
if args.verbose: print('==> Preparing data...')
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
                                          batch_size=128,
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
if args.verbose: print('==> Building model...')

def model_loader():
    model = {'resnet18': ResNet18,
             'resnet18_dropout': ResNet18Dropout,
             'resnet18_adf': ResNet18ADF,
             'resnet18_dropout_adf': ResNet18ADFDropout,
             }
    
    params = {'resnet18': [],
             'resnet18_dropout': [args.p],
             'resnet18_adf': [args.noise_variance, args.min_variance],
             'resnet18_dropout_adf': [args.p, args.noise_variance, args.min_variance],
             }
    
    return model[args.test_model_name.lower()](*params[args.test_model_name.lower()])

net = model_loader().to(device)
criterion = nn.CrossEntropyLoss()

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    if args.verbose: print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    
    model_to_load = args.load_model_name.lower()
    # if model_to_load.endswith('adf'):
    #     model_to_load = model_to_load[0:-4]
    ckpt_path = './checkpoint/ckpt_{}.pth'.format(model_to_load)
    checkpoint = torch.load(ckpt_path,map_location=torch.device(device))
    if args.verbose: print('Loaded checkpoint at location {}'.format(ckpt_path))
    
    #Preliminary bugfix for CPU execution
    state_dict = checkpoint['net']
    if device == 'cpu':
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove 'module.' of dataparallel
            new_state_dict[name]=v
        state_dict = new_state_dict
    net.load_state_dict(new_state_dict)

    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

def set_training_mode_for_dropout(net, training=True):
    """Set Dropout mode to train or eval."""

    for m in net.modules():
#        print(m.__class__.__name__)
        if m.__class__.__name__.startswith('Dropout'):
            if training==True:
                m.train()
            else:
                m.eval()
    return net        

def one_hot_pred_from_label(y_pred, labels):
    y_true = torch.zeros_like(y_pred)
    ones = torch.ones_like(y_pred)
    indexes = [l for l in labels]
    y_true[torch.arange(labels.size(0)), indexes] = ones[torch.arange(labels.size(0)), indexes]
    
    return y_true

def compute_log_likelihood(y_pred, y_true, sigma):
    dist = torch.distributions.normal.Normal(loc=y_pred, scale=sigma)
    log_likelihood = dist.log_prob(y_true)
    log_likelihood = torch.mean(log_likelihood, dim=1)
    return log_likelihood

def compute_brier_score(y_pred, y_true):
    """Brier score implementation follows 
    https://papers.nips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles.pdf.
    The lower the Brier score is for a set of predictions, the better the predictions are calibrated."""        
        
    brier_score = torch.mean((y_true-y_pred)**2, 1)
    return brier_score

def compute_preds(net, inputs, use_adf=False, use_mcdo=False):
    
    model_variance = None
    data_variance = None
    
    def keep_variance(x, min_variance):
        return x + min_variance

    keep_variance_fn = lambda x: keep_variance(x, min_variance=args.min_variance)
    softmax = nn.Softmax(dim=1)
    adf_softmax = adf.Softmax(dim=1, keep_variance_fn=keep_variance_fn)
    
    net.eval()
    if use_mcdo:
        net = set_training_mode_for_dropout(net, True)
        outputs = [net(inputs) for i in range(args.num_samples)]
        
        if use_adf:
            outputs = [adf_softmax(*outs) for outs in outputs]
            outputs_mean = [mean for (mean, var) in outputs]
            data_variance = [var for (mean, var) in outputs]
            data_variance = torch.stack(data_variance)
            data_variance = torch.mean(data_variance, dim=0)
        else:
            outputs_mean = [softmax(outs) for outs in outputs]
            
        outputs_mean = torch.stack(outputs_mean)
        model_variance = torch.var(outputs_mean, dim=0)
        # Compute MCDO prediction
        outputs_mean = torch.mean(outputs_mean, dim=0)
    else:
        outputs = net(inputs)
        if adf:
            outputs_mean, data_variance = adf_softmax(*outputs)
        else:
            outputs_mean = outputs
        
    net = set_training_mode_for_dropout(net, False)
    
    return outputs_mean, data_variance, model_variance


def evaluate(net, use_adf=False, use_mcdo=False):
    net.eval()
    test_loss = 0
    correct = 0
    brier_score = 0
    neg_log_likelihood = 0
    total = 0
    outputs_variance = None
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs_mean, data_variance, model_variance = compute_preds(net, inputs, use_adf, use_mcdo)
            if data_variance is not None and model_variance is not None:
                outputs_variance = data_variance + model_variance
            elif data_variance is not None:
                outputs_variance = data_variance
            elif model_variance is not None:
                outputs_variance = model_variance + args.tau
            
            one_hot_targets = one_hot_pred_from_label(outputs_mean, targets)
            
            # Compute negative log-likelihood (if variance estimate available)
            if outputs_variance is not None:
                batch_log_likelihood = compute_log_likelihood(outputs_mean, one_hot_targets, outputs_variance)
                batch_neg_log_likelihood = -batch_log_likelihood
                # Sum along batch dimension
                neg_log_likelihood += torch.sum(batch_neg_log_likelihood, 0).cpu().numpy().item()
            
            # Compute brier score
            batch_brier_score = compute_brier_score(outputs_mean, one_hot_targets)
            # Sum along batch dimension
            brier_score += torch.sum(batch_brier_score, 0).cpu().numpy().item()
            
            # Compute loss
            loss = criterion(outputs_mean, targets)
            test_loss += loss.item()
            
            # Compute predictions and numer of correct predictions
            _, predicted = outputs_mean.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if args.show_bar and args.verbose:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    accuracy = 100.*correct/total
    brier_score = brier_score/total
    neg_log_likelihood = neg_log_likelihood/total
    return accuracy, brier_score, neg_log_likelihood


# Testing adf model
print('==> Loaded model statistics:')
print('        test_model_name = {}'.format(args.test_model_name))
print('        load_model_name = {}'.format(args.load_model_name))
print('        @epoch = {}'.format(start_epoch))
print('        best_acc    = {}'.format(best_acc))
print('==> Selected parameters:')
print('        use_mcdo       = {}'.format(args.use_mcdo))
print('        num_samples    = {}'.format(args.num_samples))
print('        p              = {}'.format(args.p))
print('        min_variance   = {}'.format(args.min_variance))
print('        noise_variance = {}'.format(args.noise_variance))
print('        tau = {}'.format(args.tau))
print('==> Starting evaluation...')

eval_time = time.time()

accuracy, brier_score, neg_log_likelihood = evaluate(
        net,
        use_adf=args.test_model_name.lower().endswith('adf'), 
        use_mcdo=args.use_mcdo)

eval_time = time.time() - eval_time

print('Accuracy                = {}'.format(accuracy))
print('Brier Score             = {}'.format(brier_score))
print('Negative log-likelihood = {}'.format(neg_log_likelihood))
print('Time                    = {}'.format(eval_time))
    
