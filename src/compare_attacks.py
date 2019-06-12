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
from eval_utils import FGSM, evaluate_regression_stats, compute_min_max_variances
import matplotlib
# matplotlib.use('Agg') # Change default visualization resource 
#                       # (do not display plots on screen)
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

def attack(model_adf, test_steer_dataset, indexes):
    # Generate adversarial examples for all inputs corresponding to indexes
    new_images = []
    targets = []
    for i in range(len(indexes)):
        idx = indexes[i]
        test_sample = test_steer_dataset[idx]
        img_sample = test_sample[0].to(device)
        steer_sample = test_sample[1].to(device)
        attack_ = FGSM(epsilon=FLAGS.epsilon)
        adv_sample, _ = attack_(model_adf, img_sample, steer_sample, 'adf')
        new_images.append(adv_sample)
        targets.append(steer_sample)
    new_images = [t.cpu().numpy() for t in new_images]
    targets = [t.cpu().numpy() for t in targets]
    targets = [np.asscalar(t) for t in targets]
    new_images = torch.FloatTensor(new_images)
    targets = torch.FloatTensor(targets)
    # Evaluate adversarial images and corresponding variances
    predictions, epi_var, ale_var, tot_var = utils.evaluate_adversarial_variance(
            model_adf, new_images, targets, device, FLAGS)
    return new_images, predictions, epi_var, ale_var, tot_var

def compare_adv_var(adv_inputs, adv_preds, adv_vars, test_steer_dataset, std_preds, variances, indexes, mode):
    """
    Subplot with image and variance - adversarial image and adversarial variance
    """
    for i in range(len(adv_inputs)):  
        # Show i-th image and adversarial image
        test_steer = test_steer_dataset[indexes[i]]
        img = test_steer[0]
        gt_steer = test_steer[1]
        std_steer = std_preds[indexes[i]]
        var = variances[indexes[i]]
        
        adv_img = adv_inputs[i]
        adv_var = adv_vars[i]
        adv_steer = adv_preds[i]
        
        plt.figure()
        img_name1 = 'Image'
        img_text1 = 'pred = {:3.5f}\nvar = {:3.5f}'.format(std_steer, var)
        plt.subplot(121)
        plt.imshow(img.cpu().numpy()[0], cmap='gray', vmin=0, vmax=1)
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.title(img_name1)
        plt.text(50, 170, img_text1, style='italic', bbox={'facecolor':'white', 'alpha':0.5, 'pad':5})
        
        img_name2 = 'Adversarial Image'
        img_text2 = 'pred = {:3.5f}\nvar = {:3.5f}'.format(adv_steer, adv_var)
        plt.subplot(122)
        plt.imshow(adv_img.cpu().numpy()[0], cmap='gray', vmin=0, vmax=1)
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.title(img_name2)
        plt.text(50, 170, img_text2, style='italic', bbox={'facecolor':'white', 'alpha':0.5, 'pad':5})
        
        plt.suptitle(mode + " Uncertainty Evaluation\nGT steer = {:3.5f} rad".format(gt_steer))

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

        # Load trained model weights on ADF model
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
        _, _, ale_variances, _, tot_variances = \
        utils.compute_predictions_and_gt_adf(model_adf, test_loader, device, FLAGS)
            
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
        
        # Qualitative evaluation of uncertainty with adversarial examples
        if FLAGS.gen_adv_key == 'high_var':
            indexes_epi = max_epi_variances
            indexes_ale = max_ale_variances
            indexes_tot = max_tot_variances
        elif FLAGS.gen_adv_key == 'low_var':
            indexes_epi = min_epi_variances
            indexes_ale = min_ale_variances
            indexes_tot = min_tot_variances

        # Attack standard model and ADF model
        adv_inputs, adv_preds, epi_adv_var, ale_adv_var, tot_adv_var = \
            attack(model_adf, test_steer_dataset, indexes_epi)
        # Compare epistemic variances before and after attacks
        compare_adv_var(adv_inputs, adv_preds, epi_adv_var, test_steer_dataset, 
                        pred_steerings_mean, epi_variances, indexes_epi, "Epistemic")

        # Attack standard model and ADF model
        adv_inputs, adv_preds, epi_adv_var, ale_adv_var, tot_adv_var = \
            attack(model_adf, test_steer_dataset, indexes_ale)
        # Compare aleatoric variances before and after attacks
        compare_adv_var(adv_inputs, adv_preds, ale_adv_var, test_steer_dataset, 
                        pred_steerings_mean, ale_variances, indexes_ale, "Aleatoric")

        # Attack standard model and ADF model
        adv_inputs, adv_preds, epi_adv_var, ale_adv_var, tot_adv_var = \
            attack(model_adf, test_steer_dataset, indexes_tot)
        # Compare total variances before and after attacks
        compare_adv_var(adv_inputs, adv_preds, tot_adv_var, test_steer_dataset, 
                        pred_steerings_mean, tot_variances, indexes_tot, "Total")
        
    else:
        raise IOError('Cuda is not available.')

if __name__ == "__main__":
    FLAGS = parser.parse_args()
    main(FLAGS)


