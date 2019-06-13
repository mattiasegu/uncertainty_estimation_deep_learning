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
import progressbar


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(FLAGS):
    
    if not os.path.exists(FLAGS.experiment_rootdir_video):
        os.makedirs(FLAGS.experiment_rootdir_video)
        
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
        
        # Compute stats with ADF
        FLAGS.is_MCDO = True
        FLAGS.T = T_FLAG
        # Get predictions and ground truth
        print("Computing adf predictions\n...")
        
        _, predictions_mean, ale_variances, _, tot_variances = \
        utils.compute_predictions_and_gt_adf(model_adf, test_loader, device, FLAGS)
        
        """Preds from here will be referred to each video in test folder, they should be ordered"""
        frames_dir = os.path.join(FLAGS.test_dir,"HMB_3", "center")            
        ls_frames = sorted(os.listdir(frames_dir))

        writer = None
        
        out_video = os.path.join(FLAGS.experiment_rootdir_video, "outpy_3.avi")
        print("\nCreating video\n...")
        bar = progressbar.ProgressBar()
        center = (320,480)
        radius = 140
        axes = (radius,radius)
        angle=270
        
        for i, frame_name in bar(enumerate(ls_frames)):
            frame = cv2.imread(os.path.join(frames_dir,frame_name))
            cv2.circle(frame,center, radius, (255,0,0), 6)
            """Here print the circle shaded section, befor arrow"""
            std_dev = np.sqrt(tot_variances[i])
#            std_dev = tot_variances[i]
            startAngle=int(-np.rad2deg(predictions_mean[i]-3*std_dev))
            endAngle=int(-np.rad2deg(predictions_mean[i]+3*std_dev))
            cv2.ellipse(frame, center, axes, angle, startAngle, endAngle, (0,255,255), -2) #yellow
#            cv2.ellipse(frame, center, axes, angle, startAngle, endAngle, (235,244,68), -2) #light blue
            startAngle=int(-np.rad2deg(predictions_mean[i]-2*std_dev))
            endAngle=int(-np.rad2deg(predictions_mean[i]+2*std_dev))
            cv2.ellipse(frame, center, axes, angle, startAngle, endAngle, (28,184,255), -2) #orange
#            cv2.ellipse(frame, center, axes, angle, startAngle, endAngle, (167,252,68), -2) #marine blue
            startAngle=int(-np.rad2deg(predictions_mean[i]-std_dev))
            endAngle=int(-np.rad2deg(predictions_mean[i]+std_dev))
            cv2.ellipse(frame, center, axes, angle, startAngle, endAngle, (0,0,255), -2) #red
#            cv2.ellipse(frame, center, axes, angle, startAngle, endAngle, (255,0,0), -2) #blue
            pt2_x_off = np.sin(predictions_mean[i])
            pt2_y_off = np.cos(predictions_mean[i])
            pt2_x = np.round(center[0]-radius*pt2_x_off)
            pt2_y = np.round(center[1]-radius*pt2_y_off)
            pt2 = (int(pt2_x), int(pt2_y))
            cv2.arrowedLine(frame, center, pt2, (0,0,255), 3)
            
            
            
            if writer is None:
                # initialize our video writer
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                
                writer = cv2.VideoWriter(out_video, fourcc, 30,
                    (frame.shape[1], frame.shape[0]), True)

            # write the output frame to disk
            writer.write(frame)
        writer.release()


        """ A cool thing to do would be to extend the video to adversarial attacks, maybe showing in parallel the same video with and without adversarial attacks
        - normal video
        - attacks in range (I can try with unrealistic but in range so it is visible)
        - attacks out of range
        """
    else:
        raise IOError('Cuda is not available.')

if __name__ == "__main__":
    FLAGS = parser.parse_args()
    main(FLAGS)


