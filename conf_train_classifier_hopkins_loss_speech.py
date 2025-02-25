#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere University
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

The configuration file for train_model_hopkins.py.

"""

experiment_number = 1

# Options: 'text', 'speech', 'images'
data_type = 'speech'

# The name of the time-series model that we want to use from the file hopkins_model.py for fine-tuning
classification_model = 'MLP_network'

# A flag for determining whether we want to print the contents of the configuration file into the
# logging file
print_conf_contents = True

# The directory where the experimental results are saved (the models and the logging output)
result_dir = f'hopkins_results_{data_type}_{experiment_number}'

# The name of the text file into which we log the output of the training process. Please note that this
# file (and its directory) will be saved under the directory result_dir.
name_of_log_textfile = f'hopkins_{data_type}_trainlog_{experiment_number}.txt'

# The name of the saved model. Please note that this file (and its directory) will be saved under the
# directory result_dir.
best_model_name = f'hopkins_{data_type}_best_classifier_experiment_{experiment_number}.pt'

# The number of times we repeat the experiment
num_experiment_repetitions = 100

# The name of the .npy file where the experiment results are saved. Please note that this
# file (and its directory) will be saved under the directory result_dir.
name_of_result_file = f'hopkins_{data_type}_results_{experiment_number}.npy'


"""The hyperparameters for our training process"""

# The maximum number of training epochs
max_epochs = 10000

# The initial learning rate of our model training
learning_rate = 1e-4

# The minibatch size for the model training
batch_size = 1024

# The patience counter for early stopping
patience = 100

# Dropout rate of the model
dropout = 0.2

# Select the model testing criterion
testing_criterion = 'accuracy' # Options: 'accuracy' / 'f1' / 'recall' / 'uar'

# The weight for the classification loss (Hopkins loss weight will be 1 - classification_loss_weight)
classification_loss_weight = 0.75

# Define which of the MLP outputs (first, second, or third layer output) we want to use
# as the input for the Hopkins loss
# Options: 'layer_1_output', 'layer_2_output', 'layer_3_output'
hopkins_loss_input = 'layer_2_output'

# Define our loss function that we want to use from torch.nn for classification
classification_loss_name = 'CrossEntropyLoss'

# The hyperparameters for the loss functions
classification_loss_params = {}
hopkins_loss_params = {'percentage_random_samples': 0.05,
                       'target_H': 0.99,
                       'distance_metric': 'chebyshev'}

# Define the optimization algorithm we want to use from torch.optim
optimization_algorithm = 'Adam'

# The hyperparameters for our optimization algorithm
optimization_algorithm_params = {'lr': learning_rate}

# A flag to determine if we want to use a learning rate scheduler
use_lr_scheduler = True

# Define our learning rate scheduler (from torch.optim.lr_scheduler)
lr_scheduler = 'ReduceLROnPlateau'
lr_scheduler_params = {'mode': 'min',
                       'factor': 0.5,
                       'patience': 30}

"""The hyperparameters for our model"""
classification_model_params = {'input_dim': 88,
                               'hidden_dim': 128,
                               'output_dim': 8,
                               'dropout': dropout}



"""The hyperparameters for our dataset and data loaders"""

# Select whether we want to shuffle our training data
shuffle_training_data = True

# The hyperparameters for our data loaders
params_train_dataset = {}
params_validation_dataset = {}
params_test_dataset = {}

# The hyperparameters for training and validation (arguments for torch.utils.data.DataLoader object)
params_train = {'batch_size': batch_size,
                'shuffle': shuffle_training_data,
                'drop_last': False}

# The hyperparameters for testing the performance of our trained model (arguments for
# torch.utils.data.DataLoader object)
params_test = {'batch_size': batch_size,
               'shuffle': False,
               'drop_last': False}
