# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere University
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

A Hopkins statistic-based loss function for PyTorch which allows to set a desired
target value for the Hopkins statistic (i.e. to modify or enforce the feature space
towards a pre-defined topology).

"""

from torch.nn import Module
from torch import flatten, rand, randperm, cov, sqrt
from torch.linalg import vector_norm, inv
import torch.nn.functional as F
import sys


def mahalanobis_distance(a, b, inv_cov_matrix):
    diff = a - b
    md = sqrt((diff @ inv_cov_matrix * diff).sum(dim=-1))
    return md


class HopkinsLoss(Module):
    def __init__(self, percentage_random_samples=0.05, target_H=0.01, distance_metric='chebyshev', eps=1e-12):
        super().__init__()
        
        if distance_metric not in ['chebyshev', 'mahalanobis', 'cosine', 'euclidean', 'manhattan']:
            sys.exit(f'Distance metric {distance_metric} not implemented!')
        
        self.percentage_random_samples = percentage_random_samples
        self.target_H = target_H
        self.distance_metric = distance_metric
        self.eps = eps

    def forward(self, X):
        """
        A custom Hopkins statistic-based loss function for PyTorch. The Hopkins statistic is a
        measure of cluster tendency in a dataset. If the Hopkins statistic is:
            * between 0.01 and 0.3 --> the data is regularly spaced
            * around 0.5 --> the data is randomly spaced
            * between 0.7 and 0.99 --> the data has a high tendency to cluster
        
        This loss function has the ability to set a desired target value "target_H" for the Hopkins
        statistic. For example, if we set the input parameter "target_H" to 0.99, we want to modify our
        data to have a high clustering tendency.
        
        The computation of the Hopkins statistic, H, is as follows (adapted from "Validating Clusters
        using the Hopkins Statistic" by Banerjee & Davé (2004)):
            1. Let X be a set of n data points.
            2. Select a random sample of m << n data points sampled from X without replacement, denoted
               as X_tilde.
            3. Generate a set Y of m data points uniformly distributed at random.
            4. Define two distance measures:
                a) u_i: The minimum distance of y_i in Y from its nearest neighbor in X.
                b) w_i: The minimum distance of x_i_tilde in X_tilde from its nearest neighbor in X.
            5. H = sum(u_i) / (sum(u_i) + sum(w_i))
        
        _____________________________________________________________________________________________
        Input parameters:
            
            
        X: A Torch tensor for which we want to compute the Hopkins loss. The shape of the samples can
           be anything as long as batches are along the first dimension, so e.g. samples of shape
           (batch_size x num_features), (batch_size x time_dimension x num_features), or 
           (batch_size x height x width x num_channels) are all fine.
        
        Percentage_random_samples: The percentage of samples in a minibatch that we use as our sample
                                   size. Lawson and Jurs (1990) suggest to use a percentage of 5% in
                                   order to make the nearest neighbor distances independent for them
                                   to follow a beta distribution.
                
        target_H: The target value for the Hopkins statistic that we want to obtain.
        
        distance_metric: The distance metric that is used to compute the nearest neighbors.
                         Options: 'chebyshev', 'mahalanobis', 'cosine', 'euclidean', 'manhattan'
                 
        eps: A small value to avoid dividing by zero.
        _____________________________________________________________________________________________
        
        """
        
        device = X.device
        
        # We want to convert our data to shape (batch_size x num_features) if it isn't already
        if len(X.size()) > 2:
            X = flatten(X, start_dim=1)
        
        m = int(self.percentage_random_samples * X.size()[0])
        
        # We randomly sample m points from X without replacement
        X_tilde_indices = randperm(X.size()[0])[:m]
        X_tilde = X[X_tilde_indices]
        
        # We take a uniform random sample (of size m) in the feature space of the minibatch samples
        Y = (X.min(axis=0).values - X.max(axis=0).values) * rand(m, X.size()[1]).to(device) + X.max(axis=0).values
        
        # We first perform k-NN using X and Y for u_distances, and then we do the same for w_distances
        # using X and X_tilde. Note that, for the latter one, we take the second nearest neighbor in k-NN
        # since the nearest neighbors for X_tilde samples in X are the points themselves.
        if self.distance_metric == 'cosine':
            u_distances = 1 - F.cosine_similarity(X.unsqueeze(0), Y.unsqueeze(1), dim=2).topk(1, largest=False).values[:, 0]
            w_distances = 1 - F.cosine_similarity(X.unsqueeze(0), X_tilde.unsqueeze(1), dim=2).topk(2, largest=False).values[:, 1]
        elif self.distance_metric == 'euclidean':
            u_distances = vector_norm(X.unsqueeze(0) - Y.unsqueeze(1), dim=2).topk(1, largest=False).values[:, 0]
            w_distances = vector_norm(X.unsqueeze(0) - X_tilde.unsqueeze(1), dim=2).topk(2, largest=False).values[:, 1]
        elif self.distance_metric == 'manhattan':
            u_distances = vector_norm(X.unsqueeze(0) - Y.unsqueeze(1), ord=1, dim=2).topk(1, largest=False).values[:, 0]
            w_distances = vector_norm(X.unsqueeze(0) - X_tilde.unsqueeze(1), ord=1, dim=2).topk(2, largest=False).values[:, 1]
        elif self.distance_metric == 'chebyshev':
            u_distances = (X.unsqueeze(0) - Y.unsqueeze(1)).abs().max(dim=2).values.topk(1, largest=False).values[:, 0]
            w_distances = (X.unsqueeze(0) - X_tilde.unsqueeze(1)).abs().max(dim=2).values.topk(2, largest=False).values[:, 1]
        elif self.distance_metric == 'mahalanobis':
            cov_matrix = cov(X.T)
            inv_cov_matrix = inv(cov_matrix)
            u_distances = mahalanobis_distance(X.unsqueeze(0), Y.unsqueeze(1), inv_cov_matrix).topk(1, largest=False).values[:, 0]
            w_distances = mahalanobis_distance(X.unsqueeze(0), X_tilde.unsqueeze(1), inv_cov_matrix).topk(2, largest=False).values[:, 1]
        else:
            sys.exit(f'Distance metric {self.distance_metric} not implemented!')
        
        # Finally, we compute the Hopkins statistic H
        H = u_distances.sum(dim=0) / (u_distances.sum(dim=0) + w_distances.sum(dim=0) + self.eps)
        
        return (self.target_H - H).abs()


