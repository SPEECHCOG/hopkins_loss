# A PyTorch implementation of Hopkins loss

**NOTE: The links are not yet correct since the work has not yet been published!**

This repository contains code for training classifiers or autoencoders using [Hopkins loss](www.herewillbethepaperlink.com) for speech, text, and image data. Hopkins loss is a simple loss function that can be integrated into the training process of machine-learning models to modify or enforce the feature space into either a regularly-spaced, randomly-spaced, or clustered topology. The code has been implemented using PyTorch. For a thorough description of the Hopkins loss, see [the publication](www.herewillbethepaperlink.com).

**The present Hopkins loss implementation has been used in the following publication:**
[E. Vaaras and M. Airaksinen, "Feature Space Topology Control via Hopkins Loss", _(publication venue here)_](www.herewillbethepaperlink.com).

If you use the present code or its derivatives, please cite the [repository URL](https://github.com/SPEECHCOG/hopkins_loss) and/or the [aforementioned publication](www.herewillbethepaperlink.com).

## Requirements
Any `PyTorch` version newer than version 1.9.0 should work fine. You can find out how to install PyTorch here: https://pytorch.org/get-started/locally/. You also need to have `NumPy`, `scikit-learn`, `Librosa`, and `SciPy` installed.

## Repository contents
- `conf_train_autoencoder_and_linear_classifier_hopkins_loss_images.py`: Example configuration file for first training an autoencoder with Hopkins loss on image data, and then training a linear classifier to test model performance. The configuration file uses the same configuration settings that were used in the [present paper](www.herewillbethepaperlink.com).
- `conf_train_autoencoder_and_linear_classifier_hopkins_loss_speech.py`: Example configuration file for first training an autoencoder with Hopkins loss on speech data, and then training a linear classifier to test model performance. The configuration file uses the same configuration settings that were used in the [present paper](www.herewillbethepaperlink.com).
- `conf_train_autoencoder_and_linear_classifier_hopkins_loss_text.py`: Example configuration file for first training an autoencoder with Hopkins loss on text data, and then training a linear classifier to test model performance. The configuration file uses the same configuration settings that were used in the [present paper](www.herewillbethepaperlink.com).
- `conf_train_classifier_hopkins_loss_images.py`: Example configuration file for training a classifier with Hopkins loss on image data, using the same configuration settings that were used in the [present paper](www.herewillbethepaperlink.com).
- `conf_train_classifier_hopkins_loss_speech.py`: Example configuration file for training a classifier with Hopkins loss on speech data, using the same configuration settings that were used in the [present paper](www.herewillbethepaperlink.com).
- `conf_train_classifier_hopkins_loss_text.py`: Example configuration file for training a classifier with Hopkins loss on text data, using the same configuration settings that were used in the [present paper](www.herewillbethepaperlink.com).
- `hopkins_data_loader.py`: A file containing the data loaders for the experiments of the [present paper](www.herewillbethepaperlink.com).
- `hopkins_loss.py`: The Hopkins loss PyTorch implementation.
- `hopkins_model.py`: A file containing the models which were used in the experiments of the [present paper](www.herewillbethepaperlink.com).
- `py_conf_file_into_text.py`: An auxiliary script for converting _.py_ configuration files into lists of text that can be used for printing or writing the configuration file contents into a text file.
- `train_autoencoder_and_linear_classifier_hopkins_loss.py`: A script for first training an autoencoder with Hopkins loss, and then training a linear classifier to test model performance.
- `train_classifier_hopkins_loss.py`: A script for training a classifier with Hopkins loss.


## Examples of how to use the code


### How to run classifier training using Hopkins loss:
For running the classifier training script, use the command
```
python train_classifier_hopkins_loss.py <configuration_file>
```
where _<configuration_file>_ is a _.py_ configuration file containing the hyperparameters you want to use. Note that using the command
```
python train_classifier_hopkins_loss.py
```
runs the training script using the default configuration file, which is _conf_train_model_hopkins_loss_images.py_.

### How to run autoencoder training using Hopkins loss:
For running the autoencoder training script, use the command
```
python train_autoencoder_and_linear_classifier_hopkins_loss.py <configuration_file>
```
where _<configuration_file>_ is a _.py_ configuration file containing the hyperparameters you want to use. Note that using the command
```
python train_autoencoder_and_linear_classifier_hopkins_loss.py
```
runs the training script using the default configuration file, which is _conf_train_autoencoder_and_linear_classifier_hopkins_loss_images.py_.
