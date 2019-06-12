#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import os
import cv2

from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F

from opts import parser
from wrapper_datasets import create_dataset
from pytorchtools import EarlyStopping, visualize_loss, compute_l2_reg, mse_loss

from model_zoo.models_resnet8 import resnet8, resnet8_MCDO


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
def train_model(model, num_epochs, learning_rate, train_loader, valid_loader, 
                patience, model_name):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # To track the training loss as the model trains
    train_losses = []
    # To track the validation loss as the model trains
    valid_losses = []
    # To track the average training loss per epoch as the model trains
    avg_train_losses = []
    # To track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    
    # Initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    # Training loop
    decay = FLAGS.decay # Default 1e-5
    fcn = lambda step: 1./(1. + decay*step)
    scheduler = LambdaLR(optimizer, lr_lambda=fcn)
    
    for epoch in range(1, num_epochs+1):
        ###################
        # TRAIN the model #
        ###################
        model.train() # prep model for training
        for batch, (images, targets) in enumerate(train_loader, 1):
            # Uncomment the following section to visualize images and 
            # augmentation effects on training samples
            '''
            for i in range(len(images)):
                img_name = 'Steering = {}'.format(targets[i].cpu().numpy())
                cv2.imshow(img_name,images[i].cpu().numpy())
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            '''
            # Load images and targets to device
            images = images.to(device)
#            targets = targets.view(-1,1)
            targets = targets.to(device)
            # Clear gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(images)
            outputs = outputs.view(-1)
            # Calculate loss
            l2_reg = compute_l2_reg(model,model_name, FLAGS)
            loss = mse_loss(outputs,targets,epoch-1, FLAGS) + l2_reg
            # Backward pass
            loss.backward()
            # Update weights
            optimizer.step()
            # Decay Learning Rate     
            scheduler.step()
            # Record training loss
            train_losses.append(loss.item())
            
        ######################    
        # VALIDATE the model #
        ######################
        model.eval() # prep model for evaluation
        for images, targets in valid_loader:
            images = images.to(device)
            targets = targets.to(device)
            # Forward pass:
            outputs = model(images)
            outputs = outputs.view(-1)
            # Calculate loss
            loss = F.mse_loss(outputs, targets)
            # Record validation loss
            valid_losses.append(loss.item())

        # Print training/validation statistics 
        # Calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        epoch_len = len(str(num_epochs))
        
        print_msg = (f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        print(print_msg)
        
        # Clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        # Early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break    
    
    # Load the last checkpoint with the best model 
    # (returned by early_stopping call)
    ckpt_path = os.path.join(FLAGS.experiment_rootdir,'checkpoint.pt')
    model.load_state_dict(torch.load(ckpt_path))
    print('Training completed and model saved.')

    return  model, avg_train_losses, avg_valid_losses


def main(FLAGS):

    # Create the experiment rootdir if not already there
    if not os.path.exists(FLAGS.experiment_rootdir):
        os.makedirs(FLAGS.experiment_rootdir)

    # Cropped image dimensions
    crop_img_width, crop_img_height = FLAGS.crop_img_width, FLAGS.crop_img_height

    # Image mode
    if FLAGS.img_mode=='rgb':
        img_channels = 3
    elif FLAGS.img_mode == 'grayscale':
        img_channels = 1
    else:
        raise IOError("Unidentified image mode: use 'grayscale' or 'rgb'")
    
    # Output dimension
    output_dim = 1


    # Train only if cuda is available
    if device.type == 'cuda':
        # Hyper-parameters initialization
        num_epochs = FLAGS.num_epochs # Default 100
        batch_size = FLAGS.batch_size # Default 32
        learning_rate = FLAGS.learning_rate # Default 0.001

        # Generate training data with real-time augmentation
        train_steer_dataset = create_dataset(FLAGS.train_dir,mode='augmentation')
    
        train_loader = torch.utils.data.DataLoader(dataset=train_steer_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        # Generate validation data with real-time augmentation
        valid_steer_dataset = create_dataset(FLAGS.val_dir)
    
        valid_loader = torch.utils.data.DataLoader(dataset=valid_steer_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)
        
        # early stopping patience; how long to wait after last time validation loss improved.
        patience = FLAGS.patience
        
        print("Training model: ", FLAGS.model_to_train)
        # Select model to train
        if FLAGS.model_to_train == 'all':
            # Train standard resnet8 
            model_name = 'resnet8'
            model = resnet8(img_channels, output_dim).to(device)
            model, train_loss, valid_loss = train_model(model, num_epochs, learning_rate, train_loader, valid_loader, patience, model_name)
            visualize_loss(train_loss,valid_loss,model_name) # Store visualization of loss trends
            # Storing trained model
            ckpt_name = os.path.join(FLAGS.experiment_rootdir , model_name + '.pt')
            torch.save(model.state_dict(), ckpt_name)
            # Train MCDO resnet8
            model_name='resnet8_MCDO'
            model_MCDO = resnet8_MCDO(img_channels, crop_img_height, crop_img_width, output_dim).to(device)
            model_MCDO, train_loss_MCDO, valid_loss_MCDO = train_model(model_MCDO, num_epochs, learning_rate, train_loader, valid_loader, patience, model_name)
            visualize_loss(train_loss_MCDO,valid_loss_MCDO,model_name) # Store visualization of loss trends
            # Storing trained model
            ckpt_name = os.path.join(FLAGS.experiment_rootdir , model_name + '.pt')
            torch.save(model_MCDO.state_dict(), ckpt_name)
        elif FLAGS.model_to_train == 'resnet8':
            # Train standard resnet8
            model_name = 'resnet8'
            model = resnet8(img_channels, crop_img_height, crop_img_width, output_dim).to(device)
            model, train_loss, valid_loss = train_model(model, num_epochs, learning_rate, train_loader, valid_loader, patience, model_name)
            visualize_loss(train_loss,valid_loss,model_name) # Store visualization of loss trends
            # Storing trained model
            ckpt_name = os.path.join(FLAGS.experiment_rootdir , model_name + '.pt')
            torch.save(model.state_dict(), ckpt_name)
        elif FLAGS.model_to_train == 'resnet8_MCDO':
            # Train MCDO resnet8
            model_name='resnet8_MCDO'
            model_MCDO = resnet8_MCDO(img_channels, crop_img_height, crop_img_width, output_dim).to(device)
            model_MCDO, train_loss_MCDO, valid_loss_MCDO = train_model(model_MCDO, num_epochs, learning_rate, train_loader, valid_loader, patience, model_name)
            visualize_loss(train_loss_MCDO,valid_loss_MCDO,model_name) # Store visualization of loss trends
            # Storing trained model
            ckpt_name = os.path.join(FLAGS.experiment_rootdir , model_name + '.pt')
            torch.save(model_MCDO.state_dict(), ckpt_name)
        else:
            raise IOError("Model to train must be 'resnet8' or 'resnet8_MCDO' or 'all'.")
    else:
        raise IOError('Cuda is not available.')
        

if __name__ == "__main__":
    FLAGS = parser.parse_args()
    main(FLAGS)


