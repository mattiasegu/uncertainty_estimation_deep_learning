#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import numpy as np
import os
from wrapper_datasets import create_dataset
from model_zoo.models_resnet8 import resnet8_MCDO
from model_zoo.resnet8_MCDO_adf import Resnet8_MCDO_adf
import eval_utils
from opts import parser
import utils
import matplotlib
# matplotlib.use('Agg') # Change default visualization resource 
#                      # (do not display plots on screen)
import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(FLAGS):
    
    if not os.path.exists(FLAGS.experiment_rootdir_comp_adf):
        os.makedirs(FLAGS.experiment_rootdir_comp_adf)
        
    # Train only if cuda is available
    if device.type == 'cuda':
        # Create the experiment rootdir adf if not already there
        if not os.path.exists(FLAGS.experiment_rootdir_adf):
            os.makedirs(FLAGS.experiment_rootdir_adf)
        # Hyperparameters
        batch_size = FLAGS.batch_size # Default 32
        
        # Loading testing dataset
        test_steer_dataset = create_dataset(FLAGS.test_dir)
        test_loader = torch.utils.data.DataLoader(dataset=test_steer_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)
        
        targets = []
        for image, target in test_steer_dataset:
            targets.append(np.asscalar(target.cpu().numpy()))
        
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
        # Load standard model
        model = resnet8_MCDO(img_channels, crop_img_height, crop_img_width, output_dim).to(device)
        model_ckpt = os.path.join(FLAGS.experiment_rootdir,'resnet8_MCDO.pt')
        model.load_state_dict(torch.load(model_ckpt))
        # Load ADF model    	    
        model_adf = Resnet8_MCDO_adf(img_channels, output_dim, 
                                     FLAGS.noise_var, FLAGS.min_var).to(device)
        
        model_adf.load_state_dict(torch.load(model_ckpt))
    
        # Compute epistemic variance
        FLAGS.is_MCDO = True
        print("Computing epistemic variances")
        # Get predictions and ground truth
        _, pred_steerings_mean_MCDO, real_steerings, epistemic_variances = \
        utils.compute_predictions_and_gt(model, test_loader, device, FLAGS)
        
        # Compute total variance
        print("Computing total variances with ADF")
        # Get predictions and ground truth
        _, pred_steerings_mean_adf_MCDO, aleatoric_variances_adf, real_steerings, total_variances_adf = \
        utils.compute_predictions_and_gt_adf(model_adf, test_loader, device, FLAGS)
        
        # Compute log-likelihoods        
        ll_epi = utils.log_likelihood(pred_steerings_mean_MCDO, targets, np.sqrt(epistemic_variances))
        ll_ale_adf = utils.log_likelihood(pred_steerings_mean_adf_MCDO, targets, np.sqrt(aleatoric_variances_adf))
        ll_tot_adf = utils.log_likelihood(pred_steerings_mean_adf_MCDO, targets, np.sqrt(total_variances_adf))
        
        print("Log-likelihood considering         EPISTEMIC uncertainty is: {}".format(ll_epi))
        print("Log-likelihood considering     ALEATORIC_adf uncertainty is: {}\n".format(ll_ale_adf))
        print("Log-likelihood considering         TOTAL_adf uncertainty is: {}\n".format(ll_tot_adf))

    else:
        raise IOError('Cuda is not available.')

if __name__ == "__main__":
    FLAGS = parser.parse_args()
    main(FLAGS)


