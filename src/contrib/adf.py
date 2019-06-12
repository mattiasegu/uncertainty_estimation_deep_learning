from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import operator
from collections import OrderedDict
from itertools import islice
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as tf
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.conv import _ConvTransposeMixin
from torch.nn.modules.utils import _pair

from contrib.interpolation import resize2D_as
from contrib.math import normpdf, normcdf


class ReLU(nn.Module):
    def __init__(self, keep_variance_fn=None):
        super(ReLU, self).__init__()
        self._keep_variance_fn = keep_variance_fn

    def forward(self, features_mean, features_variance):
        features_stddev = torch.sqrt(features_variance)
        div = features_mean / features_stddev
        pdf = normpdf(div)
        cdf = normcdf(div)
        outputs_mean = features_mean * cdf + features_stddev * pdf
        outputs_variance = (features_mean ** 2 + features_variance) * cdf \
                           + features_mean * features_stddev * pdf - outputs_mean ** 2
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return outputs_mean, outputs_variance


class LeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01, keep_variance_fn=None):
        super(LeakyReLU, self).__init__()
        self._keep_variance_fn = keep_variance_fn
        self._negative_slope = negative_slope

    def forward(self, features_mean, features_variance):
        features_stddev = torch.sqrt(features_variance)
        div = features_mean / features_stddev
        pdf = normpdf(div)
        cdf = normcdf(div)
        negative_cdf = 1.0 - cdf
        mu_cdf = features_mean * cdf
        stddev_pdf = features_stddev * pdf
        squared_mean_variance = features_mean ** 2 + features_variance
        mean_stddev_pdf = features_mean * stddev_pdf
        mean_r = mu_cdf + stddev_pdf
        variance_r = squared_mean_variance * cdf + mean_stddev_pdf - mean_r ** 2
        mean_n = - features_mean * negative_cdf + stddev_pdf
        variance_n = squared_mean_variance * negative_cdf - mean_stddev_pdf - mean_n ** 2
        covxy = - mean_r * mean_n
        outputs_mean = mean_r - self._negative_slope * mean_n
        outputs_variance = variance_r \
                           + self._negative_slope * self._negative_slope * variance_n \
                           - 2.0 * self._negative_slope * covxy
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return outputs_mean, outputs_variance

class Dropout(nn.Module):
    def __init__(self, p: float = 0.5, keep_variance_fn=None):
        super(Dropout, self).__init__()
        self._keep_variance_fn = keep_variance_fn
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p

    def forward(self, inputs_mean, inputs_variance):
        if self.training:
            pKeep = 1-self.p
            
            device = torch.device('cuda' if inputs_mean.is_cuda else 'cpu')

            binary_mask = torch.rand(inputs_mean.size(1)) < pKeep
            z = (torch.ones(inputs_mean.size(1))*binary_mask.float()).to(device)

            z *= (1.0/(1-self.p))
            
            
            outputs_mean= inputs_mean
            for i in range(outputs_mean.size(1)):
                outputs_mean[:,i,:,:]=outputs_mean[:,i,:,:]*z[i]
            
            outputs_variance = inputs_variance
            for i in range(outputs_variance.size(1)):
                outputs_variance[:,i,:,:]=outputs_variance[:,i,:,:]*z[i]**2
            
            if self._keep_variance_fn is not None:
                outputs_variance = self._keep_variance_fn(outputs_variance)
            return outputs_mean, outputs_variance
        
        outputs_variance = inputs_variance
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return inputs_mean, outputs_variance


class MaxPool2d(nn.Module):
    def __init__(self, keep_variance_fn=None):
        super(MaxPool2d, self).__init__()
        self._keep_variance_fn = keep_variance_fn

    def _max_pool_internal(self, mu_a, mu_b, var_a, var_b):
        stddev = torch.sqrt(var_a + var_b)
        ab = mu_a - mu_b
        alpha = ab / stddev
        pdf = normpdf(alpha)
        cdf = normcdf(alpha)
        z_mu = stddev * pdf + ab * cdf + mu_b
        z_var = ((mu_a + mu_b) * stddev * pdf +
                 (mu_a ** 2 + var_a) * cdf +
                 (mu_b ** 2 + var_b) * (1.0 - cdf) - z_mu ** 2)
        if self._keep_variance_fn is not None:
            z_var = self._keep_variance_fn(z_var)
        return z_mu, z_var

    def _max_pool_1x2(self, inputs_mean, inputs_variance):
        mu_a = inputs_mean[:, :, :, 0::2]
        mu_b = inputs_mean[:, :, :, 1::2]
        var_a = inputs_variance[:, :, :, 0::2]
        var_b = inputs_variance[:, :, :, 1::2]
        outputs_mean, outputs_variance = self._max_pool_internal(
            mu_a, mu_b, var_a, var_b)
        return outputs_mean, outputs_variance

    def _max_pool_2x1(self, inputs_mean, inputs_variance):
        mu_a = inputs_mean[:, :, 0::2, :]
        mu_b = inputs_mean[:, :, 1::2, :]
        var_a = inputs_variance[:, :, 0::2, :]
        var_b = inputs_variance[:, :, 1::2, :]
        outputs_mean, outputs_variance = self._max_pool_internal(
            mu_a, mu_b, var_a, var_b)
        return outputs_mean, outputs_variance

    def forward(self, inputs_mean, inputs_variance):
        z_mean, z_variance = self._max_pool_1x2(inputs_mean, inputs_variance)
        outputs_mean, outputs_variance = self._max_pool_2x1(z_mean, z_variance)
        return outputs_mean, outputs_variance


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, keep_variance_fn=None):
        super(Linear, self).__init__()
        self._keep_variance_fn = keep_variance_fn
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, inputs_mean, inputs_variance):
        outputs_mean = tf.linear(inputs_mean, self.weight, self.bias)
        outputs_variance = tf.linear(inputs_variance, self.weight**2, None)
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return outputs_mean, outputs_variance

class BatchNorm2d(nn.Module):
    _version = 2
    __constants__ = ['track_running_stats', 'momentum', 'eps', 'weight', 'bias',
                     'running_mean', 'running_var', 'num_batches_tracked']

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, keep_variance_fn=None):
        super(BatchNorm2d, self).__init__()
        self._keep_variance_fn = keep_variance_fn
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.uniform_(self.weight)
            nn.init.zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, inputs_mean, inputs_variance):

        # exponential_average_factor is self.momentum set to
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        outputs_mean = tf.batch_norm(
            inputs_mean, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        outputs_variance = inputs_variance
        for i in range(outputs_variance.size(1)):
            outputs_variance[:,i,:,:]=outputs_variance[:,i,:,:]*self.weight[i]**2
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return outputs_mean, outputs_variance


class Conv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 keep_variance_fn=None):
        self._keep_variance_fn = keep_variance_fn
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, inputs_mean, inputs_variance):
        outputs_mean = tf.conv2d(
            inputs_mean, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        outputs_variance = tf.conv2d(
            inputs_variance, self.weight ** 2, None, self.stride, self.padding, self.dilation, self.groups)
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return outputs_mean, outputs_variance


class ConvTranspose2d(_ConvTransposeMixin, _ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1,
                 keep_variance_fn=None):
        self._keep_variance_fn = keep_variance_fn
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super(ConvTranspose2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias)

    def forward(self, inputs_mean, inputs_variance, output_size=None):
        output_padding = self._output_padding(inputs_mean, output_size)
        outputs_mean = tf.conv_transpose2d(
            inputs_mean, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)
        outputs_variance = tf.conv_transpose2d(
            inputs_variance, self.weight ** 2, None, self.stride, self.padding,
            output_padding, self.groups, self.dilation)
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return outputs_mean, outputs_variance


def concatenate_as(tensor_list, tensor_as, dim, mode="bilinear"):
    means = [resize2D_as(x[0], tensor_as[0], mode=mode) for x in tensor_list]
    variances = [resize2D_as(x[1], tensor_as[0], mode=mode) for x in tensor_list]
    means = torch.cat(means, dim=dim)
    variances = torch.cat(variances, dim=dim)
    return means, variances


class Sequential(nn.Module):
    def __init__(self, *args):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return Sequential(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, inputs, inputs_variance):
        for module in self._modules.values():
            inputs, inputs_variance = module(inputs, inputs_variance)

        return inputs, inputs_variance
