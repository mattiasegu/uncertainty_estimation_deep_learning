#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import os
import matplotlib
matplotlib.use('Agg') # Change default visualization resource 
                      # (do not display plots on screen)
import matplotlib.pyplot as plt
from opts import parser

FLAGS = parser.parse_args()

def compute_l2_reg(model,model_name,FLAGS):
    # Function that sets weight_decay only for weights and not biases and only 
    # for conv layers inside residual layers
    lambda_ = FLAGS.weight_decay
    params_dict = dict(model.named_parameters())
    l2_reg=[]  
    if model_name == 'resnet8':
        for key, value in params_dict.items():
            if ((key[-8:] == '2.weight' or key[-8:] == '5.weight') and key[0:8]=='residual'):
                l2_reg += [lambda_*torch.norm(value.view(value.size(0),-1),2)]
    else:
        for key, value in params_dict.items():
            if ((key[-8:] == '2.weight' or key[-8:] == '6.weight') and key[0:8]=='residual'):
                l2_reg += [lambda_*torch.norm(value.view(value.size(0),-1),2)]
    #l2_reg = sum(l2_reg)/FLAGS.batch_size
    l2_reg = sum(l2_reg)
    return l2_reg



class het_loss(torch.nn.Module):
    
    def __init__(self):
        super(het_loss,self).__init__()
        
    def forward(self, mean, log_var, targets, epoch, FLAGS):
        precision = torch.exp(-log_var)
        return torch.mean(0.5*precision * (targets-mean)**2 + 0.5*log_var)


#def het_loss(mean, log_var, targets, epoch, FLAGS):
#    precision = torch.exp(-log_var)
#    return torch.mean(precision * (targets-mean)**2 + log_var)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        ckpt_path = os.path.join(FLAGS.experiment_rootdir,'checkpoint.pt')
        torch.save(model.state_dict(), ckpt_path)
        self.val_loss_min = val_loss


def visualize_loss(train_loss,valid_loss,model_name):
    # Visualize the loss as the network trained
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(train_loss)+1),train_loss,label='Training Loss')
    plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')
    
    # Find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
    
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 0.5) # consistent scale
    plt.xlim(0, len(train_loss)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig_name = model_name + '_loss_plot.png'
    fig_path = os.path.join(FLAGS.experiment_rootdir, fig_name)
    fig.savefig(fig_path, bbox_inches='tight')
