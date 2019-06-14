#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

parser = argparse.ArgumentParser()

# Input
parser.add_argument('--img_width', type=int, default = 320,
                    help='Target Image Width')
parser.add_argument('--img_height', type=int, default = 240,
                    help='Target Image Height')
parser.add_argument('--crop_img_width', type=int, default=200, 
                    help='Cropped image widht')
parser.add_argument('--crop_img_height', type=int, default=200, 
                    help='Cropped image height')
parser.add_argument('--img_mode', type=str, default="grayscale", 
                    help='Load mode for images, either rgb or grayscale')

# Training
parser.add_argument('--batch_size', type = int, default=32, 
                    help='Batch size in training and evaluation')
parser.add_argument('--num_epochs', type = int, default=100, 
                    help='Number of epochs for training')
parser.add_argument('--patience', type = int, default=20, 
                     help='How many epochs to wait since validation loss' 
                          'has stopped to decrease')
parser.add_argument('--learning_rate', type = float, default=0.001, 
                    help='Initial learning rate for training')
parser.add_argument('--dropout', type = float, default=0.2, 
                    help='Dropout rate for training MC Dropout NN')
parser.add_argument('--weight_decay', type = float, default=1e-4, 
                    help='Weight decay for NN weights, for biases is 0')
parser.add_argument('--decay', type = float, default=1e-5, 
                    help='Learning rate decay')
parser.add_argument('--model_to_train', type=str, default='resnet8_MCDO', 
                    help='Select model to train: resnet8, resnet8_MCDO or all')

# Files
parser.add_argument('--experiment_rootdir', type=str, default="../exp", 
                    help='Folder containing logs, model weights and results')
parser.add_argument('--experiment_rootdir_adf', type=str, default="../exp_adf", 
                    help='Folder containing logs, model weights and results')
parser.add_argument('--experiment_rootdir_het', type=str, default="../exp_het", 
                    help='Folder containing logs, model weights and results')
parser.add_argument('--experiment_rootdir_comp', type=str, default="../exp_comp", 
                    help='Folder containing logs, model weights and results')
parser.add_argument('--experiment_rootdir_video', type=str, default="../exp_video", 
                    help='Folder containing video')
parser.add_argument('--experiment_rootdir_comp_adf', type=str, default="../exp_comp_adf", 
                    help='Folder containing logs, model weights and results')
parser.add_argument('--train_dir', type=str, default="../data/training", 
                    help='Folder containing training experiments')
parser.add_argument('--val_dir', type=str, default="../data/validation", 
                    help='Folder containing validation experiments')
parser.add_argument('--test_dir', type=str, default="../data/testing", 
                    help='Folder containing testing experiments')

# Testing
parser.add_argument('--T', type=int, default=50, 
                         help='Number of tests for MC Dropout')
#parser.add_argument('--tau', type=float, default=0.1, 
#                         help='Noise on input data')
parser.add_argument('--noise_var', type=float, default=1e-3, 
                         help='Noise Variance on input data')
parser.add_argument('--min_var', type=float, default=1e-3, 
                         help='Noise Variance on every layer activation')
parser.add_argument('--model_to_test', type=str, default='resnet8_MCDO', 
                         help='Select model to test: resnet8, resnet8_MCDO')
parser.add_argument('--is_MCDO', type=lambda x: (str(x).lower() == 'true'),
                    default=True, 
                    help='Whether to use the model trained for MC Dropout' 
                         'sampling or not (True or False)')

# Adversarial Attacks
parser.add_argument('--gen_adv_key', type=str, default='high_var', 
                         help='Select key for adversarial example generation: '
                              'low_var, high_var')
parser.add_argument('--epsilon', type=float, default=0.01, 
                         help='Epsilon for adversarial examples generation')

