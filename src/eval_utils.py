#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import utils

# Functions to evaluate steering prediction

def explained_variance_1d(ypred,y):
    """
    Var[ypred - y] / var[y].
    https://www.quora.com/What-is-the-meaning-proportion-of-variance-explained-in-linear-regression
    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary


def compute_explained_variance(predictions, real_values):
    """
    Computes the explained variance of prediction for each
    steering and the average of them
    """
    assert np.all(predictions.shape == real_values.shape)
    ex_variance = explained_variance_1d(predictions, real_values)
    print("EVA = {}".format(ex_variance))
    return ex_variance


def compute_sq_residuals(predictions, real_values):
    assert np.all(predictions.shape == real_values.shape)
    sq_res = np.square(predictions - real_values)
    sr = np.mean(sq_res, axis = -1)
    print("MSE = {}".format(sr))
    return sq_res


def compute_rmse(predictions, real_values):
    assert np.all(predictions.shape == real_values.shape)
    mse = np.mean(np.square(predictions - real_values))
    rmse = np.sqrt(mse)
    print("RMSE = {}".format(rmse))
    return rmse


def compute_highest_regression_errors(predictions, real_values, n_errors=20):
    """
    Compute the indexes with highest error
    """
    assert np.all(predictions.shape == real_values.shape)
    sq_res = np.square(predictions - real_values)
    highest_errors = sq_res.argsort()[-n_errors:][::-1]
    return highest_errors

def compute_min_max_variances(variances, n_variances=10):
    """
    Compute the indexes with highest and lowest variances
    """
    max_variances = variances.argsort()[-n_variances:][::-1]
    min_variances = variances.argsort()[:n_variances]
    return max_variances, min_variances
    
    
def random_regression_baseline(real_values):
    mean = np.mean(real_values)
    std = np.std(real_values)
    return np.random.normal(loc=mean, scale=abs(std), size=real_values.shape)


def constant_baseline(real_values):
    mean = np.mean(real_values)
    return mean * np.ones_like(real_values)


def evaluate_regression_stats(predictions, real_values):
    evas = compute_explained_variance(predictions, real_values)
    rmse = compute_rmse(predictions, real_values)
    
    return evas, rmse

def evaluate_regression(predictions, real_values, fname):
    evas = compute_explained_variance(predictions, real_values)
    rmse = compute_rmse(predictions, real_values)
    highest_errors = compute_highest_regression_errors(predictions, real_values,
            n_errors=20)
    
    dictionary = {"evas": evas.tolist(), "rmse": rmse.tolist(),
                  "highest_errors": highest_errors.tolist()}
    utils.write_to_file(dictionary, fname)

class FGSM(nn.Module):
    """
    Class to generate adversarial examples with FGSM method
    """
    def __init__(self, epsilon=0.1):
        super(FGSM, self).__init__()
        self._epsilon = epsilon

    def __call__(self, model, input1, target1, mode=None):
        input1v = input1.clone()
        input1v.requires_grad = True
#        example_dict["input1"] = input1v

        model.zero_grad()
        if mode=='adf':
            mean1, var1 = model(input1v.unsqueeze(0))
            loss = F.mse_loss(mean1.view(-1), target1.view(-1))
        else: 
            output1 = model(input1v.unsqueeze(0))
            loss = F.mse_loss(output1.view(-1), target1.view(-1))

        loss.backward()

        new_input = input1 + torch.sign(input1v.grad)*self._epsilon
        new_input = torch.clamp(new_input, 0.0, 1.0)

        return new_input, target1
