# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere University
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

This file contains the MLP networks used in the Hopkins loss-based experiments.

"""

import sys
from torch.nn import Module, Linear, Dropout, GELU, LeakyReLU, ELU, ReLU






class MLP_network(Module):
    """
    A three-layer MLP network for the Hopkins loss-based experiments.
    
    """
    
    def __init__(self,
                 input_dim = 128,
                 hidden_dim = 128,
                 output_dim = 2,
                 non_linearity = 'gelu',
                 dropout=0.2):

        super().__init__()
        
        self.linear_layer_1 = Linear(in_features=input_dim, out_features=hidden_dim)
        self.linear_layer_2 = Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.linear_layer_3 = Linear(in_features=hidden_dim, out_features=output_dim)
        
        if non_linearity == 'elu':
            self.non_linearity = ELU()
        elif non_linearity == 'gelu':
            self.non_linearity = GELU()
        elif non_linearity == 'relu':
            self.non_linearity = ReLU()
        elif non_linearity == 'leakyrelu':
            self.non_linearity = LeakyReLU()
        else:
            sys.exit(f'{non_linearity} is not an option! Options: "elu", "gelu", "relu", or "leakyrelu"')
        
        self.dropout = Dropout(dropout)

    def forward(self, X):
        
        # X is of size [batch_size, input_dim]
        X_output_layer_1 = self.dropout(self.non_linearity(self.linear_layer_1(X)))
        X_output_layer_2 = self.dropout(self.non_linearity(self.linear_layer_2(X_output_layer_1)))
        X_output = self.linear_layer_3(X_output_layer_2)
        # X_output is of size [batch_size, output_dim]
        
        return X_output, X_output_layer_1, X_output_layer_2




class MLP_linear(Module):
    """
    An MLP network consisting of a single linear layer.
    
    """
    
    def __init__(self,
                 input_dim = 128,
                 output_dim = 2):

        super().__init__()
        
        self.linear_layer = Linear(in_features=input_dim, out_features=output_dim)

    def forward(self, X):
        
        # X is of size [batch_size, input_dim]
        X_output = self.linear_layer(X)
        # X_output is of size [batch_size, output_dim]
        
        return X_output





