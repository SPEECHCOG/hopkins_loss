# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere University
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

Code for training a classifier with or without using the Hopkins loss during the
training process.

"""

import numpy as np 
import os
import time
import sys

from importlib.machinery import SourceFileLoader
from py_conf_file_into_text import convert_py_conf_file_to_text
from sklearn.metrics import f1_score, recall_score, accuracy_score
from copy import deepcopy

import torch.nn.functional as F
from torch import cuda, no_grad, save, load, flatten, randperm, rand, cov, sqrt, stack
from torch.linalg import vector_norm, inv
from torch.utils.data import DataLoader
from torch.nn import Softmax


# Read the configuration file
if len(sys.argv) > 2:
    sys.exit('\nUsage: \n1) python train_classifier_hopkins_loss.py \nOR \n2) python train_classifier_hopkins_loss.py <configuration_file>')
if len(sys.argv) == 2:
    conf = SourceFileLoader('', sys.argv[1]).load_module()
    conf_file_name = sys.argv[1]
else:
    try:
        import conf_train_classifier_hopkins_loss_images as conf
        conf_file_name = 'conf_train_classifier_hopkins_loss_images.py'
    except ModuleNotFoundError:
        sys.exit('\nUsage: \n1) python train_classifier_hopkins_loss.py \nOR \n2) python train_classifier_hopkins_loss.py <configuration_file>\n\n' \
                 'By using the first option, you need to have a configuration file named "conf_train_classifier_hopkins_loss_images.py" in the same ' \
                 'directory as "train_classifier_hopkins_loss.py"')

# Import our classification model
classification_model = getattr(__import__('hopkins_model', fromlist=[conf.classification_model]), conf.classification_model)

if conf.data_type == 'text':
    dataset = getattr(__import__('hopkins_data_loader', fromlist=['imdb_dataset']), 'imdb_dataset')
elif conf.data_type == 'speech':
    dataset = getattr(__import__('hopkins_data_loader', fromlist=['ravdess_egemaps_dataset']), 'ravdess_egemaps_dataset')
elif conf.data_type == 'images':
    dataset = getattr(__import__('hopkins_data_loader', fromlist=['fashion_mnist_dataset']), 'fashion_mnist_dataset')
else:
    sys.exit('The configuration setting "data_type" should be either "text", "speech", or "images"!')

# Import our loss functions
classification_loss = getattr(__import__('torch.nn', fromlist=[conf.classification_loss_name]), conf.classification_loss_name)
hopkins_loss = getattr(__import__('hopkins_loss', fromlist=['HopkinsLoss']), 'HopkinsLoss')

# Import our optimization algorithm
optimization_algorithm = getattr(__import__('torch.optim', fromlist=[conf.optimization_algorithm]), conf.optimization_algorithm)



def uar_function(y_true, y_pred):
    """
    Calculate the unweighted average recall.
    
    """
    
    uar = np.mean(recall_score(y_true, y_pred, average=None))
    
    return uar



def compute_hopkins_statistic(X, percentage_random_samples=conf.hopkins_loss_params['percentage_random_samples'],
                              distance_metric=conf.hopkins_loss_params['distance_metric'], eps=1e-12):
    
    """
    A function to compute the Hopkins statistic.
    
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
    
    distance_metric: The distance metric that is used to compute the nearest neighbors.
                     Options: 'chebyshev', 'mahalanobis', 'cosine', 'euclidean', 'manhattan'
             
    eps: A small value to avoid dividing by zero.
    
    """
    
    device = X.device
    
    # We want to convert our data to shape (batch_size x num_features) if it isn't already
    if len(X.size()) > 2:
        X = flatten(X, start_dim=1)
    
    m = int(percentage_random_samples * X.size()[0])
    
    # We randomly sample m points from X without replacement
    X_tilde_indices = randperm(X.size()[0])[:m]
    X_tilde = X[X_tilde_indices]
    
    # We take a uniform random sample (of size m) in the feature space of the minibatch samples
    Y = (X.min(axis=0).values - X.max(axis=0).values) * rand(m, X.size()[1]).to(device) + X.max(axis=0).values
    
    # We first perform k-NN using X and Y for u_distances, and then we do the same for w_distances
    # using X and X_tilde. Note that, for the latter one, we take the second nearest neighbor in k-NN
    # since the nearest neighbors for X_tilde samples in X are the points themselves.
    if distance_metric == 'cosine':
        u_distances = 1 - F.cosine_similarity(X.unsqueeze(0), Y.unsqueeze(1), dim=2).topk(1, largest=False).values[:, 0]
        w_distances = 1 - F.cosine_similarity(X.unsqueeze(0), X_tilde.unsqueeze(1), dim=2).topk(2, largest=False).values[:, 1]
    elif distance_metric == 'euclidean':
        u_distances = vector_norm(X.unsqueeze(0) - Y.unsqueeze(1), dim=2).topk(1, largest=False).values[:, 0]
        w_distances = vector_norm(X.unsqueeze(0) - X_tilde.unsqueeze(1), dim=2).topk(2, largest=False).values[:, 1]
    elif distance_metric == 'mahalanobis':
        cov_matrix = cov(X.T)
        inv_cov_matrix = inv(cov_matrix)
        u_distances = mahalanobis_distance(X.unsqueeze(0), Y.unsqueeze(1), inv_cov_matrix).topk(1, largest=False).values[:, 0]
        w_distances = mahalanobis_distance(X.unsqueeze(0), X_tilde.unsqueeze(1), inv_cov_matrix).topk(2, largest=False).values[:, 1]
    elif distance_metric == 'manhattan':
        u_distances = vector_norm(X.unsqueeze(0) - Y.unsqueeze(1), ord=1, dim=2).topk(1, largest=False).values[:, 0]
        w_distances = vector_norm(X.unsqueeze(0) - X_tilde.unsqueeze(1), ord=1, dim=2).topk(2, largest=False).values[:, 1]
    elif distance_metric == 'chebyshev':
        u_distances = (X.unsqueeze(0) - Y.unsqueeze(1)).abs().max(dim=2).values.topk(1, largest=False).values[:, 0]
        w_distances = (X.unsqueeze(0) - X_tilde.unsqueeze(1)).abs().max(dim=2).values.topk(2, largest=False).values[:, 1]
    else:
        sys.exit(f'Distance metric {distance_metric} not implemented!')
    
    # Finally, we compute the Hopkins statistic H
    H = u_distances.sum(dim=0) / (u_distances.sum(dim=0) + w_distances.sum(dim=0) + eps)
    
    return H


def mahalanobis_distance(a, b, inv_cov_matrix):
    diff = a - b
    md = sqrt((diff @ inv_cov_matrix * diff).sum(dim=-1))
    return md




if __name__ == "__main__":
    
    # We make sure that we are able to write the logging file
    textfile_path, textfile_name = os.path.split(f'{conf.result_dir}/{conf.name_of_log_textfile}')
    if not os.path.exists(textfile_path):
        if textfile_path != '':
            os.makedirs(textfile_path)
    file = open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'w')
    file.close()
    
    # Read the text in the configuration file and add it to the logging file
    if conf.print_conf_contents:
        conf_file_lines = convert_py_conf_file_to_text(conf_file_name)
        with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
            f.write(f'The configuration settings in the file {conf_file_name}:\n\n')
            for line in conf_file_lines:
                f.write(f'{line}\n')
            f.write('\n########################################################################################\n\n\n\n')
    
    # Do some hyperparameter checking
    if not 0.0 <= conf.classification_loss_weight <= 1.0:
        sys.exit('The classification loss weight must be in the range [0.0, 1.0]!')
    
    if conf.hopkins_loss_input not in ['layer_1_output', 'layer_2_output', 'layer_3_output']:
        sys.exit(f'"{conf.hopkins_loss_input}" not implemented for hopkins_loss_input!')
    
    # Use CUDA if it is available, else use CPU
    device = 'cuda' if cuda.is_available() else 'cpu'
    with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
        f.write(f'Process on {device}\n\n')
    
    # Initialize the data loaders
    with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
        f.write('Initializing training set...\n')
    training_set = dataset(train_val_test='train', **conf.params_train_dataset)
    train_data_loader = DataLoader(training_set, **conf.params_train)
    with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
        f.write('Done!\n')
        f.write('Initializing validation set...\n')
    validation_set = dataset(train_val_test='validation', **conf.params_validation_dataset)
    validation_data_loader = DataLoader(validation_set, **conf.params_train)
    with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
        f.write('Done!\n')
        f.write('Initializing test set...\n')
    test_set = dataset(train_val_test='test', **conf.params_test_dataset)
    test_data_loader = DataLoader(test_set, **conf.params_test)
    with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
        f.write('Done!\n\n')
    
    hopkins_loss_weight = 1 - conf.classification_loss_weight
    
    experiment_results = []
    feats_train_H_random_init_means = []
    feats_validation_H_random_init_means = []
    feats_test_H_random_init_means = []
    feats_train_H_random_init_stds = []
    feats_validation_H_random_init_stds = []
    feats_test_H_random_init_stds = []
    feats_train_H_means = []
    feats_validation_H_means = []
    feats_test_H_means = []
    feats_train_H_stds = []
    feats_validation_H_stds = []
    feats_test_H_stds = []
    for repetition_number in range(conf.num_experiment_repetitions):
        with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
            f.write(f'############# Experiment repetition number {repetition_number + 1}/{conf.num_experiment_repetitions} #############\n\n')
        
        # Initialize our model, and pass the model to the available device
        Classifier = classification_model(**conf.classification_model_params).to(device)
        
        # Give the parameters of our model to an optimizer
        optimizer = optimization_algorithm(params=Classifier.parameters(), **conf.optimization_algorithm_params)
        
        # Get our learning rate for later use
        learning_rate = optimizer.param_groups[0]['lr']
        
        # Give the optimizer to the learning rate scheduler
        if conf.use_lr_scheduler:
            scheduler = getattr(__import__('torch.optim.lr_scheduler', fromlist=[conf.lr_scheduler]), conf.lr_scheduler)
            lr_scheduler = scheduler(optimizer, **conf.lr_scheduler_params)
        
        # Initialize our loss functions
        if conf.classification_loss_weight > 0:
            loss_classification = classification_loss(**conf.classification_loss_params)
        if hopkins_loss_weight > 0:
            loss_hopkins = hopkins_loss(**conf.hopkins_loss_params)
        
        # We compute the Hopkins statistic for a randomly initialized model
        Classifier.eval()
        feats_train_H_random_init = []
        feats_validation_H_random_init = []
        feats_test_H_random_init = []
        with no_grad():
            for train_data in train_data_loader:
                X, _ = [element.to(device) for element in train_data]
                Y_pred, Y_layer_1, Y_layer_2 = Classifier(X.float())
                if conf.hopkins_loss_input == 'layer_1_output':
                    feats_train_H_random_init.append(compute_hopkins_statistic(Y_layer_1))
                elif conf.hopkins_loss_input == 'layer_2_output':
                    feats_train_H_random_init.append(compute_hopkins_statistic(Y_layer_2))
                else:
                    feats_train_H_random_init.append(compute_hopkins_statistic(Y_pred))
            
            for validation_data in validation_data_loader:
                X, _ = [element.to(device) for element in validation_data]
                Y_pred, Y_layer_1, Y_layer_2 = Classifier(X.float())
                if conf.hopkins_loss_input == 'layer_1_output':
                    feats_validation_H_random_init.append(compute_hopkins_statistic(Y_layer_1))
                elif conf.hopkins_loss_input == 'layer_2_output':
                    feats_validation_H_random_init.append(compute_hopkins_statistic(Y_layer_2))
                else:
                    feats_validation_H_random_init.append(compute_hopkins_statistic(Y_pred))
            
            for test_data in test_data_loader:
                X, _ = [element.to(device) for element in test_data]
                Y_pred, Y_layer_1, Y_layer_2 = Classifier(X.float())
                if conf.hopkins_loss_input == 'layer_1_output':
                    feats_test_H_random_init.append(compute_hopkins_statistic(Y_layer_1))
                elif conf.hopkins_loss_input == 'layer_2_output':
                    feats_test_H_random_init.append(compute_hopkins_statistic(Y_layer_2))
                else:
                    feats_test_H_random_init.append(compute_hopkins_statistic(Y_pred))
        
        feats_train_H_random_init = stack(feats_train_H_random_init, dim=0).cpu().numpy()
        feats_validation_H_random_init = stack(feats_validation_H_random_init, dim=0).cpu().numpy()
        feats_test_H_random_init = stack(feats_test_H_random_init, dim=0).cpu().numpy()
        
        feats_train_H_random_init_means.append(feats_train_H_random_init.mean())
        feats_validation_H_random_init_means.append(feats_validation_H_random_init.mean())
        feats_test_H_random_init_means.append(feats_test_H_random_init.mean())
        feats_train_H_random_init_stds.append(feats_train_H_random_init.std())
        feats_validation_H_random_init_stds.append(feats_validation_H_random_init.std())
        feats_test_H_random_init_stds.append(feats_test_H_random_init.std())
        
        # Variables for early stopping
        lowest_validation_loss = 1e10
        best_validation_epoch = 0
        patience_counter = 0
        
        best_model = None
        
        # Flag for indicating if max epochs are reached
        max_epochs_reached = 1
        
        # Start training our model
        with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
            f.write('Starting training...\n')
        
        for epoch in range(1, conf.max_epochs + 1):
            
            start_time = time.time()
    
            # Lists containing the losses of each epoch
            epoch_loss_training = []
            epoch_loss_validation = []
            epoch_loss_classification_training = []
            epoch_loss_classification_validation = []
            epoch_loss_hopkins_training = []
            epoch_loss_hopkins_validation = []
    
            # Indicate that we are in training mode, so e.g. dropout will function
            Classifier.train()
            
            # Loop through every minibatch of our training data
            for train_data in train_data_loader:
                
                # Get the minibatches
                X, Y = [element.to(device) for element in train_data]
                
                # Zero the gradient of the optimizer
                optimizer.zero_grad()
                
                # Pass our data through the model
                Y_pred, Y_layer_1, Y_layer_2 = Classifier(X.float())
                
                # Compute the losses
                loss_h = 0.0
                loss_c = 0.0
                if conf.classification_loss_weight > 0:
                    loss_c = loss_classification(input=Y_pred, target=Y.long())
                if hopkins_loss_weight > 0:
                    if conf.hopkins_loss_input == 'layer_1_output':
                        loss_h = loss_hopkins(Y_layer_1)
                    elif conf.hopkins_loss_input == 'layer_2_output':
                        loss_h = loss_hopkins(Y_layer_2)
                    else:
                        loss_h = loss_hopkins(Y_pred)
                if conf.classification_loss_weight > 0 and hopkins_loss_weight > 0:
                    loss = conf.classification_loss_weight * loss_c + hopkins_loss_weight * loss_h
                else:
                    loss = loss_c + loss_h
                
                # Perform the backward pass
                loss.backward()
                
                # Update the weights
                optimizer.step()
                
                # Add the losses to their appropriate lists
                epoch_loss_training.append(loss.item())
                if conf.classification_loss_weight > 0:
                    epoch_loss_classification_training.append(loss_c.item())
                else:
                    epoch_loss_classification_training.append(0.0)
                if hopkins_loss_weight > 0:
                    epoch_loss_hopkins_training.append(loss_h.item())
                else:
                    epoch_loss_hopkins_training.append(0.0)
            
            
            # Indicate that we are in evaluation mode, so e.g. dropout will not function
            Classifier.eval()
    
            # Make PyTorch not calculate the gradients, so everything will be much faster.
            with no_grad():
                
                # Loop through every batch of our validation data and perform a similar process
                # as for the training data
                for validation_data in validation_data_loader:
                    X, Y = [element.to(device) for element in validation_data]
                    Y_pred, Y_layer_1, Y_layer_2 = Classifier(X.float())
                    loss_h = 0.0
                    loss_c = 0.0
                    if conf.classification_loss_weight > 0:
                        loss_c = loss_classification(input=Y_pred, target=Y.long())
                    if hopkins_loss_weight > 0:
                        if conf.hopkins_loss_input == 'layer_1_output':
                            loss_h = loss_hopkins(Y_layer_1)
                        elif conf.hopkins_loss_input == 'layer_2_output':
                            loss_h = loss_hopkins(Y_layer_2)
                        else:
                            loss_h = loss_hopkins(Y_pred)
                    if conf.classification_loss_weight > 0 and hopkins_loss_weight > 0:
                        loss = conf.classification_loss_weight * loss_c + hopkins_loss_weight * loss_h
                    else:
                        loss = loss_c + loss_h
                    epoch_loss_validation.append(loss.item())
                    if conf.classification_loss_weight > 0:
                        epoch_loss_classification_validation.append(loss_c.item())
                    else:
                        epoch_loss_classification_validation.append(0.0)
                    if hopkins_loss_weight > 0:
                        epoch_loss_hopkins_validation.append(loss_h.item())
                    else:
                        epoch_loss_hopkins_validation.append(0.0)
            
            # Calculate mean losses
            epoch_loss_training = np.array(epoch_loss_training).mean()
            epoch_loss_validation = np.array(epoch_loss_validation).mean()
            epoch_loss_classification_training = np.array(epoch_loss_classification_training).mean()
            epoch_loss_classification_validation = np.array(epoch_loss_classification_validation).mean()
            epoch_loss_hopkins_training = np.array(epoch_loss_hopkins_training).mean()
            epoch_loss_hopkins_validation = np.array(epoch_loss_hopkins_validation).mean()
            
            # Check early stopping conditions
            if epoch_loss_validation < lowest_validation_loss:
                lowest_validation_loss = epoch_loss_validation
                patience_counter = 0
                best_model = deepcopy(Classifier.state_dict())
                best_validation_epoch = epoch
                
                # We first make sure that we are able to write the files
                save_names = [f'{conf.result_dir}/{conf.best_model_name}']
                for model_save_name in save_names:
                    model_path, model_filename = os.path.split(model_save_name)
                    if not os.path.exists(model_path):
                        if model_path != '':
                            os.makedirs(model_path)
                
                save(best_model, f'{conf.result_dir}/{conf.best_model_name}')
            else:
                patience_counter += 1
            
            end_time = time.time()
            epoch_time = end_time - start_time
            
            with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                f.write(f'Epoch: {epoch:04d} | Mean train loss: {epoch_loss_training:6.4f} | '
                  f'Mean val loss: {epoch_loss_validation:6.4f} (lowest: {lowest_validation_loss:6.4f}) | '
                  f'Mean train loss (c): {epoch_loss_classification_training:6.4f} | '
                  f'Mean val loss (c): {epoch_loss_classification_validation:6.4f} | '
                  f'Mean train loss (H): {epoch_loss_hopkins_training:6.4f} | '
                  f'Mean val loss (H): {epoch_loss_hopkins_validation:6.4f} | '
                  f'Duration: {epoch_time:4.2f} seconds\n')
            
            # We check that do we need to update the learning rate based on the validation loss
            if conf.use_lr_scheduler:
                if conf.lr_scheduler == 'ReduceLROnPlateau':
                    lr_scheduler.step(epoch_loss_validation)
                else:
                    lr_scheduler.step()
                current_learning_rate = optimizer.param_groups[0]['lr']
                if current_learning_rate != learning_rate:
                    learning_rate = current_learning_rate
                    with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                        f.write(f'Updated learning rate after epoch {epoch} based on learning rate scheduler, now lr={learning_rate}\n')
            
            # If patience counter is fulfilled, stop the training
            if patience_counter >= conf.patience:
                max_epochs_reached = 0
                break
        
        if max_epochs_reached:
            with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                f.write('\nMax number of epochs reached, stopping training\n\n')
        else:
            with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                f.write('\nExiting due to early stopping\n\n')
        
        if best_model is None:
            with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                f.write('\nNo best model. The criteria for the lowest acceptable validation accuracy not satisfied!\n\n')
            sys.exit('No best model, exiting...')
        else:
            with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                f.write(f'\nBest epoch {best_validation_epoch} with validation loss {lowest_validation_loss}\n\n')
        
        
        with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
            f.write('\nStarting testing... => ')
            
        # Load the best version of the model
        try:
            Classifier.load_state_dict(load(f'{conf.result_dir}/{conf.best_model_name}', map_location=device))
        except (FileNotFoundError, RuntimeError):
            Classifier.load_state_dict(best_model)
                
        testing_loss = []
        epoch_true_Y_testing = np.array([])
        epoch_pred_Y_testing = np.array([])
        Classifier.eval()
        with no_grad():
            for test_data in test_data_loader:
                X, Y = [element.to(device) for element in test_data]
                Y_pred, Y_layer_1, Y_layer_2 = Classifier(X.float())
                loss_h = 0.0
                loss_c = 0.0
                if conf.classification_loss_weight > 0:
                    loss_c = loss_classification(input=Y_pred, target=Y.long())
                if hopkins_loss_weight > 0:
                    if conf.hopkins_loss_input == 'layer_1_output':
                        loss_h = loss_hopkins(Y_layer_1)
                    elif conf.hopkins_loss_input == 'layer_2_output':
                        loss_h = loss_hopkins(Y_layer_2)
                    else:
                        loss_h = loss_hopkins(Y_pred)
                if conf.classification_loss_weight > 0 and hopkins_loss_weight > 0:
                    loss = conf.classification_loss_weight * loss_c + hopkins_loss_weight * loss_h
                else:
                    loss = loss_c + loss_h
                smax = Softmax(dim=1)
                Y_pred_smax_np = smax(Y_pred).detach().cpu().numpy()
                predictions = np.argmax(Y_pred_smax_np, axis=1)
                epoch_true_Y_testing = np.concatenate((epoch_true_Y_testing, Y.detach().cpu().numpy()))
                epoch_pred_Y_testing = np.concatenate((epoch_pred_Y_testing, predictions))
                testing_loss.append(loss.item())
            testing_loss = np.array(testing_loss).mean()
            
        if conf.testing_criterion == 'f1':
            testing_accuracy = f1_score(epoch_true_Y_testing, epoch_pred_Y_testing, average='macro')
        elif conf.testing_criterion == 'recall':
            testing_accuracy = recall_score(epoch_true_Y_testing, epoch_pred_Y_testing, average='macro')
        elif conf.testing_criterion == 'uar':
            testing_accuracy = uar_function(epoch_true_Y_testing, epoch_pred_Y_testing)
        elif conf.testing_criterion == 'accuracy':
            testing_accuracy = accuracy_score(epoch_true_Y_testing, epoch_pred_Y_testing)
        else:
            sys.exit(f'The training criterion {conf.train_criterion} not implemented!')
        
        with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
            f.write(f'Testing loss: {testing_loss:7.4f}, testing accuracy: {testing_accuracy:7.4f} ({conf.testing_criterion})\n\n')
        
        experiment_results.append(testing_accuracy)
        np.save(f'{conf.result_dir}/{conf.name_of_result_file}', np.array(experiment_results))
        
        with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
            f.write(f'Saved the results of {repetition_number + 1}/{conf.num_experiment_repetitions} experiments to {conf.result_dir}/{conf.name_of_result_file}\n\n\n\n')
        
        # Compute the Hopkins statistic using the best classification model
        feats_train_H = []
        feats_validation_H = []
        feats_test_H = []
        with no_grad():
            for train_data in train_data_loader:
                X, _ = [element.to(device) for element in train_data]
                Y_pred, Y_layer_1, Y_layer_2 = Classifier(X.float())
                if conf.hopkins_loss_input == 'layer_1_output':
                    feats_train_H.append(compute_hopkins_statistic(Y_layer_1))
                elif conf.hopkins_loss_input == 'layer_2_output':
                    feats_train_H.append(compute_hopkins_statistic(Y_layer_2))
                else:
                    feats_train_H.append(compute_hopkins_statistic(Y_pred))
            
            for validation_data in validation_data_loader:
                X, _ = [element.to(device) for element in validation_data]
                Y_pred, Y_layer_1, Y_layer_2 = Classifier(X.float())
                if conf.hopkins_loss_input == 'layer_1_output':
                    feats_validation_H.append(compute_hopkins_statistic(Y_layer_1))
                elif conf.hopkins_loss_input == 'layer_2_output':
                    feats_validation_H.append(compute_hopkins_statistic(Y_layer_2))
                else:
                    feats_validation_H.append(compute_hopkins_statistic(Y_pred))
            
            for test_data in test_data_loader:
                X, _ = [element.to(device) for element in test_data]
                Y_pred, Y_layer_1, Y_layer_2 = Classifier(X.float())
                if conf.hopkins_loss_input == 'layer_1_output':
                    feats_test_H.append(compute_hopkins_statistic(Y_layer_1))
                elif conf.hopkins_loss_input == 'layer_2_output':
                    feats_test_H.append(compute_hopkins_statistic(Y_layer_2))
                else:
                    feats_test_H.append(compute_hopkins_statistic(Y_pred))
        
        feats_train_H = stack(feats_train_H, dim=0).cpu().numpy()
        feats_validation_H = stack(feats_validation_H, dim=0).cpu().numpy()
        feats_test_H = stack(feats_test_H, dim=0).cpu().numpy()
        
        feats_train_H_means.append(feats_train_H.mean())
        feats_validation_H_means.append(feats_validation_H.mean())
        feats_test_H_means.append(feats_test_H.mean())
        feats_train_H_stds.append(feats_train_H.std())
        feats_validation_H_stds.append(feats_validation_H.std())
        feats_test_H_stds.append(feats_test_H.std())
    
    feats_train_H_random_init_mean = np.array(feats_train_H_random_init_means).mean()
    feats_validation_H_random_init_mean = np.array(feats_validation_H_random_init_means).mean()
    feats_test_H_random_init_mean = np.array(feats_test_H_random_init_means).mean()
    feats_train_H_random_init_std = np.array(feats_train_H_random_init_stds).mean()
    feats_validation_H_random_init_std = np.array(feats_validation_H_random_init_stds).mean()
    feats_test_H_random_init_std = np.array(feats_test_H_random_init_stds).mean()
    feats_train_H_mean = np.array(feats_train_H_means).mean()
    feats_validation_H_mean = np.array(feats_validation_H_means).mean()
    feats_test_H_mean = np.array(feats_test_H_means).mean()
    feats_train_H_std = np.array(feats_train_H_stds).mean()
    feats_validation_H_std = np.array(feats_validation_H_stds).mean()
    feats_test_H_std = np.array(feats_test_H_stds).mean()
    
    with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
        f.write('######################################################################################################\n')
        f.write(f'The average results of {conf.num_experiment_repetitions} classification experiments (Hopkins statistic):\n\n')
        f.write(f'Training feats (random init), mean H (STD): {feats_train_H_random_init_mean} ({feats_train_H_random_init_std})\n')
        f.write(f'Validation feats (random init), mean H (STD): {feats_validation_H_random_init_mean} ({feats_validation_H_random_init_std})\n')
        f.write(f'Test feats (random init), mean H (STD): {feats_test_H_random_init_mean} ({feats_test_H_random_init_std})\n\n')
        
        f.write(f'Training feats, mean H (STD): {feats_train_H_mean} ({feats_train_H_std})\n')
        f.write(f'Validation feats, mean H (STD): {feats_validation_H_mean} ({feats_validation_H_std})\n')
        f.write(f'Test feats, mean H (STD): {feats_test_H_mean} ({feats_test_H_std})\n')
        f.write('######################################################################################################\n')
    

        
        
        

    