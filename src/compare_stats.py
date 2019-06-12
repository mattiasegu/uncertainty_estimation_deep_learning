#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import os
import cv2
import numpy as np
from wrapper_datasets import create_dataset
from model_zoo.models_resnet8 import resnet8_MCDO
from model_zoo.resnet8_MCDO_adf import Resnet8_MCDO_adf
from opts import parser
import utils
from eval_utils import evaluate_regression_stats, compute_min_max_variances
import matplotlib
#matplotlib.use('Agg') # Change default visualization resource 
#                      # (do not display plots on screen)
import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_stats(metric_std,metric_ls,metric_name):
    # Visualize RMSE and EVA
    fig = plt.figure(figsize=(10,8))
    label_std = 'Standard ' + metric_name
    label_MCDO = 'MCDO ' + metric_name
    plt.plot(range(1,len(metric_ls)+1),metric_ls,label=label_MCDO)
    # Plot horizontal line for metric value without MCDO
    plt.axhline(metric_std, linestyle='--', color='r',label=label_std)
    plt.xscale("log")
    plt.xlabel('MC Samples T')
    plt.ylabel(metric_name)
    if metric_name=='EVA':
        plt.ylim(-1, 1) # consistent scale
    else: plt.ylim(0, 0.2) # consistent scale
    plt.xlim(0, len(metric_ls)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.axis('tight')
    plt.show()
    fig_name = metric_name + '_T' + str(len(metric_ls)) + '.png'
    fig_path = os.path.join(FLAGS.experiment_rootdir_comp, fig_name)
    fig.savefig(fig_path, bbox_inches='tight')

def plot_variances(epistemic_var_ls,total_var_ls):
    # Visualize epistemic and total variances on the same plot
    fig = plt.figure(figsize=(10,8))
    label_epi = 'Epistemic Variance'
    label_tot = 'Total Variance'
    plt.plot(range(1,len(epistemic_var_ls)+1),epistemic_var_ls,label=label_epi)
    plt.plot(range(1,len(total_var_ls)+1),total_var_ls,label=label_tot)
    plt.xscale("log")
    plt.xlabel('MC Samples T')
    plt.ylabel('Variances')
    plt.ylim(0, 1) # consistent scale
    plt.xlim(0, len(epistemic_var_ls)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.axis('tight')
    plt.show()
    fig_name = 'compare_var_T' + str(len(epistemic_var_ls)) + '.png'
    fig_path = os.path.join(FLAGS.experiment_rootdir_comp, fig_name)
    fig.savefig(fig_path, bbox_inches='tight')


def show_lowest_highest(test_steer_dataset, variances, min_variances, max_variances, mode=None):        
    '''
    Function that shows images with lowest and highest variances, 
    where min_variances and max_variances are the corresponding indexes
    '''
    for i in range(len(min_variances)):  
        # Show i-th low variance image
        test_steer_min = test_steer_dataset[min_variances[i]]
        img_min = test_steer_min[0]
        steer_min = test_steer_min[1]
        var_min = variances[min_variances[i]]
        
        plt.figure()
        img_name1 = 'Low '+ mode +' Variance'
        img_text1 = 'steer = {:3.5f}\nvar = {:3.5f}'.format(steer_min,var_min)
        plt.subplot(121)
        plt.imshow(img_min.cpu().numpy()[0], cmap='gray', vmin=0, vmax=1)
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.title(img_name1)
        plt.text(50, 170, img_text1, style='italic', bbox={'facecolor':'white', 'alpha':0.5, 'pad':5})
        
        
        # Show i-th high variance image
        test_steer_max = test_steer_dataset[max_variances[i]]
        img_max = test_steer_max[0]
        steer_max = test_steer_max[1]
        var_max = variances[max_variances[i]]
        
        img_name2 = 'High '+ mode +' Variance'
        img_text2 = 'steer = {:3.5f}\nvar = {:3.5f}'.format(steer_max,var_max)
        plt.subplot(122)
        plt.imshow(img_max.cpu().numpy()[0], cmap='gray', vmin=0, vmax=1)
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.title(img_name2)
        plt.text(50, 170, img_text2, style='italic', bbox={'facecolor':'white', 'alpha':0.5, 'pad':5})
        plt.show()



def main(FLAGS):
    
    if not os.path.exists(FLAGS.experiment_rootdir_comp):
        os.makedirs(FLAGS.experiment_rootdir_comp)
        
    # Train only if cuda is available
    if device.type == 'cuda':
        # Create the experiment rootdir adf if not already there
        if not os.path.exists(FLAGS.experiment_rootdir):
            os.makedirs(FLAGS.experiment_rootdir_adf)
        # Hyperparameters
        batch_size = FLAGS.batch_size # Default 32
        
        # Loading testing dataset
        test_steer_dataset = create_dataset(FLAGS.test_dir)
        test_loader = torch.utils.data.DataLoader(dataset=test_steer_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)
        
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
        model = resnet8_MCDO(img_channels, crop_img_height, crop_img_width, output_dim).to(device)
        model_ckpt = os.path.join(FLAGS.experiment_rootdir,'resnet8_MCDO.pt')
        model.load_state_dict(torch.load(model_ckpt))
        
        model_adf = Resnet8_MCDO_adf(img_channels, output_dim, 
                                     FLAGS.noise_var, FLAGS.min_var).to(device)
        model_adf.load_state_dict(torch.load(model_ckpt))

        # Ensure that MCDO is NOT enabled
        FLAGS.is_MCDO = False
        T_FLAG = FLAGS.T
        
        # Compute stats without MCDO
        FLAGS.T = 0
        # Get predictions and ground truth
        print("Computing standard predictions\n...")
        MC_samples, pred_steerings_mean, real_steerings, _ = \
        utils.compute_predictions_and_gt(model, test_loader, device, FLAGS)
        
        # Evaluate predictions: EVA, residuals
        print("Evaluation of standard predictions")
        evas_std, rmse_std = evaluate_regression_stats(pred_steerings_mean, real_steerings)
        
        # Compute stats with ADF
        FLAGS.is_MCDO = True
        FLAGS.T = T_FLAG
        # Get predictions and ground truth
        print("Computing adf predictions\n...")
        MC_samples, _, ale_variances, _, tot_variances = \
        utils.compute_predictions_and_gt_adf(model_adf, test_loader, device, FLAGS)
            
        MC_samples_means = MC_samples['mean']
        MC_samples_vars = MC_samples['var']
                
        evas_ls = []
        rmse_ls = []
        epistemic_var_ls = []
        total_var_ls = []
        # At T-th iteration, take the mean of only the first T samples
        for T in range(1,T_FLAG+1):
            pred_steerings_cur = np.mean(MC_samples_means[0:T,:], axis=0)
            # Evaluate predictions: EVA, residuals
            print("Evaluation of predictions for {} MC samples".format(T))
            evas, rmse = evaluate_regression_stats(pred_steerings_cur, real_steerings)
            # Compute epistemic and total variances and mean over them
            epistemic_var = np.mean(np.var(MC_samples_means[0:T,:], axis=0),axis=0)
            aleatoric_var = np.mean(np.mean(MC_samples_vars[0:T,:], axis=0),axis=0)
            total_var = epistemic_var + aleatoric_var

            evas_ls.append(evas)
            rmse_ls.append(rmse)
            epistemic_var_ls.append(epistemic_var)
            total_var_ls.append(total_var)

        plot_variances(epistemic_var_ls,total_var_ls)        
        plot_stats(evas_std,evas_ls,'EVA')
        plot_stats(rmse_std,rmse_ls,'RMSE')
        print("Saved plots for EVA, RMSE and Variances comparison in folder " + FLAGS.experiment_rootdir_comp)
        
        # Compute highest and lowest variances indexes
        epi_variances = tot_variances - ale_variances
        max_epi_variances, min_epi_variances = compute_min_max_variances(epi_variances)
        max_ale_variances, min_ale_variances = compute_min_max_variances(ale_variances)
        max_tot_variances, min_tot_variances = compute_min_max_variances(tot_variances)
            
        print("\nSamples with highest epistemic uncertainty: ", max_epi_variances )
        print("\nSamples with lowest epistemic uncertainty: ", min_epi_variances )
        print("\nSamples with highest aleatoric uncertainty: ", max_ale_variances )
        print("\nSamples with lowest aleatoric uncertainty: ", min_ale_variances )
        print("\nSamples with highest total uncertainty: ", max_tot_variances )
        print("\nSamples with lowest total uncertainty: ", min_tot_variances )
        
        # Show qualitative results        
        show_lowest_highest(test_steer_dataset, epi_variances, min_epi_variances, max_epi_variances, mode='Epistemic')
        show_lowest_highest(test_steer_dataset, ale_variances, min_ale_variances, max_ale_variances, mode='Aleatoric')
        show_lowest_highest(test_steer_dataset, tot_variances, min_tot_variances, max_tot_variances, mode='Total')
        
        
    else:
        raise IOError('Cuda is not available.')

if __name__ == "__main__":
    FLAGS = parser.parse_args()
    main(FLAGS)


