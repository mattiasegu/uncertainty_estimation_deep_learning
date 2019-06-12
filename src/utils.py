import numpy as np
import json
import torch
from scipy.misc import logsumexp


def evaluate_adversarial_variance(model_adf, images, targets, device, FLAGS):

    model_adf.eval()
    # Set Dropout to be applied also in eval mode
    if FLAGS.is_MCDO:
        for m in model_adf.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
                
    with torch.no_grad():
        images = images.to(device)
        targets = targets.cpu().numpy()
        all_MC_samples_mean = []
        all_outputs_mean = []
        all_outputs_var = []
        all_targets = []
        if FLAGS.is_MCDO: 
            # Perform T forward passes (collect T MC samples)
            MC_samples = [model_adf(images) for _ in range(FLAGS.T)]
            MC_means = np.array([t[0].view(-1).cpu().numpy() for t in MC_samples])
            MC_vars = np.array([t[1].view(-1).cpu().numpy() for t in MC_samples])
            MC_pred_mean = np.mean(MC_means, axis=0)
            MC_pred_var = np.mean(MC_vars, axis=0)
            
            all_MC_samples_mean.append(MC_means)
            all_outputs_mean.append(MC_pred_mean)
            all_outputs_var.append(MC_pred_var)
            all_targets.append(targets)
            # outputs_variance is NOT computed here because taking samples
            # with MCDO already includes aleatoric variance inside total_variance
        else:
            # Forward pass
            outputs = model_adf(images)
            outputs_mean = outputs[0].view(-1).cpu().numpy()
            outputs_var = outputs[1].view(-1).cpu().numpy()
            # Append results
            all_outputs_mean.append(outputs_mean)
            all_outputs_var.append(outputs_var)
            all_targets.append(targets)

          
    predictions_mean = np.concatenate(all_outputs_mean, axis=0)
    aleatoric_variances = np.concatenate(all_outputs_var, axis=0)
    ground_truth = np.concatenate(all_targets, axis=0) 
    
    MC_samples = None
    epistemic_variances=None
    total_variances=None
    if FLAGS.is_MCDO:
        MC_samples = np.concatenate(all_MC_samples_mean, axis=1)
        # Compute epistemic uncertainty
        epistemic_variances = np.var(MC_samples, axis=0)
        total_variances = epistemic_variances + aleatoric_variances
    else:
        aleatoric_variances = np.concatenate(all_outputs_var, axis=0)        
    
    return predictions_mean, epistemic_variances, aleatoric_variances, total_variances



def compute_predictions_and_gt(model, data_loader, device, FLAGS):
    
    model.eval()
    # Set Dropout to be applied also in eval mode
    if FLAGS.is_MCDO:
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
    
    with torch.no_grad():
        all_MC_samples = []
        all_outputs = []
        all_targets = []
        for i, (images, targets) in enumerate(data_loader):
            images = images.to(device)
            targets = targets.cpu().numpy()
            all_targets.append(targets)
            if FLAGS.is_MCDO: 
                # Perform T forward passes (collect T MC samples)
                MC_samples = np.array([model(images).view(-1).cpu().numpy() for _ in range(FLAGS.T)])
                MC_pred = np.mean(MC_samples, axis=0)
                all_MC_samples.append(MC_samples)
                all_outputs.append(MC_pred)
            else:
                outputs = model(images)
                all_outputs.append(outputs.view(-1).cpu().numpy())
    
    MC_samples = None
    epistemic_variance=None
    if FLAGS.is_MCDO:
        MC_samples = np.concatenate(all_MC_samples, axis=1)
        # Compute epistemic uncertainty
        epistemic_variance = np.var(MC_samples, axis=0)
        
    predictions = np.concatenate(all_outputs, axis=0)        
    ground_truth = np.concatenate(all_targets, axis=0) 
    
    return MC_samples, predictions, ground_truth, epistemic_variance

def compute_predictions_and_gt_adf(model, data_loader, device, FLAGS):
    
    model.eval()
    # Set Dropout to be applied also in eval mode
    if FLAGS.is_MCDO:
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
                
    with torch.no_grad():
        all_MC_samples_mean = []
        all_MC_samples_var = []
        all_outputs_mean = []
        all_outputs_var = []
        all_targets = []
        for i, (images, targets) in enumerate(data_loader):
            images = images.to(device)
            targets = targets.cpu().numpy()
            if FLAGS.is_MCDO: 
                # Perform T forward passes (collect T MC samples)
                MC_samples = [model(images) for _ in range(FLAGS.T)]
                MC_means = np.array([t[0].view(-1).cpu().numpy() for t in MC_samples])
                MC_vars = np.array([t[1].view(-1).cpu().numpy() for t in MC_samples])
                MC_pred_mean = np.mean(MC_means, axis=0)
                MC_pred_var = np.mean(MC_vars, axis=0)
                
                all_MC_samples_mean.append(MC_means)
                all_MC_samples_var.append(MC_vars)
                all_outputs_mean.append(MC_pred_mean)
                all_outputs_var.append(MC_pred_var)
                all_targets.append(targets)
            else:
                # Forward pass
                outputs = model(images)
                outputs_mean = outputs[0].view(-1).cpu().numpy()
                outputs_var = outputs[1].view(-1).cpu().numpy()
                # Append results
                all_outputs_mean.append(outputs_mean)
                all_outputs_var.append(outputs_var)
                all_targets.append(targets)

          
    predictions_mean = np.concatenate(all_outputs_mean, axis=0)
    aleatoric_variances = np.concatenate(all_outputs_var, axis=0)
    ground_truth = np.concatenate(all_targets, axis=0) 
    
    MC_samples = None
    total_variances=None
    if FLAGS.is_MCDO:
        MC_samples_mean = np.concatenate(all_MC_samples_mean, axis=1)
        MC_samples_var = np.concatenate(all_MC_samples_var, axis=1)
        MC_samples = {'mean': MC_samples_mean,
                      'var': MC_samples_var}
        # Compute epistemic uncertainty
        epistemic_variances = np.var(MC_samples_mean, axis=0)
        total_variances = epistemic_variances + aleatoric_variances
    else:
        aleatoric_variances = np.concatenate(all_outputs_var, axis=0)        
    
    return MC_samples, predictions_mean, aleatoric_variances, ground_truth, total_variances

def compute_predictions_and_gt_het(model, data_loader, device, FLAGS):
    
    model.eval()
    # Set Dropout to be applied also in eval mode
    if FLAGS.is_MCDO:
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
                
    with torch.no_grad():
        all_MC_samples_mean = []
        all_outputs_mean = []
        all_outputs_var = []
        all_targets = []
        for i, (images, targets) in enumerate(data_loader):
            images = images.to(device)
            targets = targets.cpu().numpy()
            if FLAGS.is_MCDO: 
                # Perform T forward passes (collect T MC samples)
                MC_samples = [model(images) for _ in range(FLAGS.T)]
                MC_means = np.array([t['mean'].view(-1).cpu().numpy() for t in MC_samples])
                MC_vars = np.array([t['log_var'].view(-1).cpu().numpy() for t in MC_samples])
                MC_pred_mean = np.mean(MC_means, axis=0)
                MC_pred_var = np.mean(MC_vars, axis=0)
                
                all_MC_samples_mean.append(MC_means)
                all_outputs_mean.append(MC_pred_mean)
                all_outputs_var.append(MC_pred_var)
                all_targets.append(targets)
                # outputs_variance is NOT computed here because taking samples
                # with MCDO already includes aleatoric variance inside total_variance
            else:
                # Forward pass
                outputs = model(images)
                outputs_mean = outputs['mean'].view(-1).cpu().numpy()
                outputs_var = outputs['log_var'].view(-1).cpu().numpy()
                # Append results
                all_outputs_mean.append(outputs_mean)
                all_outputs_var.append(outputs_var)
                all_targets.append(targets)

          
    predictions_mean = np.concatenate(all_outputs_mean, axis=0)
    aleatoric_variances = np.concatenate(all_outputs_var, axis=0)
    aleatoric_variances = np.exp(aleatoric_variances)
    ground_truth = np.concatenate(all_targets, axis=0) 
    
    MC_samples = None
    total_variances=None
    if FLAGS.is_MCDO:
        MC_samples = np.concatenate(all_MC_samples_mean, axis=1)
        # Compute epistemic uncertainty
        epistemic_variances = np.var(MC_samples, axis=0)
        total_variances = epistemic_variances + aleatoric_variances       
    
    return MC_samples, predictions_mean, aleatoric_variances, ground_truth, total_variances

def log_likelihood(y_pred, y_true, sigma):
    y_true = torch.Tensor(y_true)
    y_pred= torch.Tensor(y_pred)
    sigma = torch.Tensor(sigma)
    
    dist = torch.distributions.normal.Normal(loc=y_pred, scale=sigma)
    ll = torch.mean(dist.log_prob(y_true))
    ll = np.asscalar(ll.numpy())
    return ll

def write_to_file(dictionary, fname):
    """
    Writes everything is in a dictionary in json model.
    """
    with open(fname, "w") as f:
        json.dump(dictionary,f)
        print("Written file {}".format(fname))
