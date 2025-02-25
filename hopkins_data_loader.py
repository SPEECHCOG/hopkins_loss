# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere University
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

The data loaders for the Hopkins loss-based experiments.

"""

import numpy as np
import os
import sys
import opensmile
import torch
from transformers import BertTokenizer, BertModel
from torch import no_grad
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms
from tqdm import tqdm




# Z-score normalization (zero mean, unit standard deviation)
def normalize_features(feats) -> np.ndarray:
    
    normalized = (feats - feats.mean(axis=0)) / feats.std(axis=0)

    # Remove NaN values by converting them to zero
    normalized = np.nan_to_num(normalized)

    return normalized





class fashion_mnist_dataset(Dataset):
    """
    Dataloader for the Fashion-MNIST dataset (Xiao et al., 2017).
    
    """
    
    def __init__(self, root_dir = './fashion_mnist', train_val_test = 'train', num_samples_train = 50000,
                 num_samples_validation = 10000, random_seed = 24):
        
        # Normalize pixel values to be in the range [-1.0, 1.0] by subtracting the mean (0.5) and dividing by the
        # standard deviation (0.5). Originally, the pixel values are in the range [0.0, 1.0] since transforms.ToTensor()
        # scales the pixel values in the range [0, 255] to [0.0, 1.0].
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = datasets.FashionMNIST(root='./fashion_mnist', train=True, download=True, transform=transform)
        training_set, validation_set = random_split(train_dataset, [num_samples_train, num_samples_validation],
                                                    generator=torch.Generator().manual_seed(random_seed))
        test_set = datasets.FashionMNIST(root='./fashion_mnist', train=False, download=True, transform=transform)
        
        if train_val_test == 'train':
            self.X = torch.cat([feature for feature, _ in training_set], dim=0).view(-1, 28 * 28)
            self.Y = torch.tensor([label for _, label in training_set])
        elif train_val_test == 'validation':
            self.X = torch.cat([feature for feature, _ in validation_set], dim=0).view(-1, 28 * 28)
            self.Y = torch.tensor([label for _, label in validation_set])
        else:
            self.X = torch.cat([feature for feature, _ in test_set], dim=0).view(-1, 28 * 28)
            self.Y = torch.tensor([label for _, label in test_set])

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]





class ravdess_egemaps_dataset(Dataset):
    """
    Dataloader for the RAVDESS emotional speech data dataset (Livingstone & Russo, 2018), using the eGeMAPS
    features from the OpenSMILE toolkit.
    
    """

    def __init__(self, wav_file_dir = './audio_speech_actors_01-24', preprocess_data = False,
                 preprocessed_data_dir = './preprocessed_ravdess_feats', train_val_test = 'train',
                 train_val_ratio = 0.75, train_test_ratio = 0.8, random_seed = 42, norm_feats = True,
                 data_sampling_rate=1.0):
        super().__init__()
        
        # Preprocess the data
        if not os.path.exists(preprocessed_data_dir):
            os.makedirs(preprocessed_data_dir)
            preprocess_data = True
        else:
            if preprocess_data and len(os.listdir(preprocessed_data_dir)) != 0:
                # Remove old files from the given directory
                filenames_old_files = os.listdir(preprocessed_data_dir)
                for filename in filenames_old_files:
                    os.remove(os.path.join(preprocessed_data_dir, filename))
        
        if preprocess_data or len(os.listdir(preprocessed_data_dir)) == 0:
            # Find out our WAV files in the given directory
            try:
                # This is used to spot nonexisting directories since os.walk() is silent about them
                error_variable = os.listdir(wav_file_dir)
                del error_variable
                
                filenames_wav = []
                for dir_path, dir_names, file_names in os.walk(wav_file_dir):
                    if len(file_names) > 0:
                        for file_name in file_names:
                            filenames_wav.append(os.path.join(dir_path, file_name))
            except FileNotFoundError:
                sys.exit(f'Given .wav file directory {wav_file_dir} does not exist!')
            
            wav_file_names = [filename for filename in filenames_wav if filename.endswith('.wav')]
            wav_file_names = sorted(wav_file_names, key=lambda x: (int(x.split(os.sep)[-1].split('.')[0].split('-')[0]),
                                                                   int(x.split(os.sep)[-1].split('.')[0].split('-')[1]),
                                                                   int(x.split(os.sep)[-1].split('.')[0].split('-')[2]),
                                                                   int(x.split(os.sep)[-1].split('.')[0].split('-')[3]),
                                                                   int(x.split(os.sep)[-1].split('.')[0].split('-')[4]),
                                                                   int(x.split(os.sep)[-1].split('.')[0].split('-')[5]),
                                                                   int(x.split(os.sep)[-1].split('.')[0].split('-')[6])))
            del filenames_wav
            
            # We go through each WAV file and extract eGeMAPS features
            opensmile_feature_extractor = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                                                          feature_level=opensmile.FeatureLevel.Functionals)
            features = []
            speaker_ids = []
            labels = []
            for filename in tqdm(wav_file_names):
                features.append(opensmile_feature_extractor.process_file(filename).to_numpy())
                speaker_ids.append(int(filename.split(os.sep)[-1].split('.')[0].split('-')[-1]))
                labels.append(int(filename.split(os.sep)[-1].split('.')[0].split('-')[2]))
            features = np.squeeze(np.array(features))
            speaker_ids = np.array(speaker_ids)
            labels = np.array(labels) - 1
            
            # Save the pre-processed features, speaker IDs, and labels
            np.save(os.path.join(preprocessed_data_dir, 'ravdess_egemaps_features.npy'), features)
            np.save(os.path.join(preprocessed_data_dir, 'ravdess_speaker_ids.npy'), speaker_ids)
            np.save(os.path.join(preprocessed_data_dir, 'ravdess_labels.npy'), labels)
        
        # Load the pre-processed features, speaker IDs, and labels
        features = np.load(os.path.join(preprocessed_data_dir, 'ravdess_egemaps_features.npy'))
        speaker_ids = np.load(os.path.join(preprocessed_data_dir, 'ravdess_speaker_ids.npy'))
        labels = np.load(os.path.join(preprocessed_data_dir, 'ravdess_labels.npy'))
        
        # Normalize the features (z-score normalization)
        if norm_feats:
            features = normalize_features(features)
        
        # We split the data into training, validation, and test sets based on the speaker ID
        num_speaker_ids = len(np.unique(speaker_ids))
        train_test_permutation = np.random.RandomState(seed=random_seed).permutation(num_speaker_ids)
        speaker_ids_shuffled = np.unique(speaker_ids)[train_test_permutation]
        num_trainval_speakers = round(train_test_ratio * num_speaker_ids)
        trainval_speaker_ids = speaker_ids_shuffled[:num_trainval_speakers]
        test_speaker_ids = speaker_ids_shuffled[num_trainval_speakers:]
        num_train_speakers = round(train_val_ratio * num_trainval_speakers)
        train_speaker_ids = trainval_speaker_ids[:num_train_speakers]
        val_speaker_ids = trainval_speaker_ids[num_train_speakers:]
            
        if train_val_test == 'train':
            speaker_ids_data_split = train_speaker_ids
        elif train_val_test == 'validation':
            speaker_ids_data_split = val_speaker_ids
        else:
            speaker_ids_data_split = test_speaker_ids
        
        X = []
        Y = []
        for i in range(len(features)):
            if speaker_ids[i] in speaker_ids_data_split:
                X.append(features[i])
                Y.append(labels[i])
        X = np.array(X)
        Y = np.array(Y)
        self.X = X
        self.Y = Y
        
        if data_sampling_rate < 1.00 and train_val_test != 'test':
            # We randomly select a subset of the data
            num_sampled = int(data_sampling_rate * len(X))
            np.random.seed(3*random_seed)
            sampling_indices = np.random.choice(np.arange(len(X)), num_sampled, replace=False)
            self.X = self.X[sampling_indices, :]
            self.Y = self.Y[sampling_indices]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]





class imdb_dataset(Dataset):
    """
    Dataloader for the Large Movie Review Dataset (Maas et al., 2011), also known as the IMDB movie
    review dataset.
    
    """

    def __init__(self, dataset_dir = './aclImdb', train_val_test = 'train', preprocess_data = False,
                 preprocessed_data_dir_base = './preprocessed_imdb_feats', train_val_ratio = 0.8,
                 tokenizer_max_length = 512, random_seed = 42, data_sampling_rate=1.0):
        super().__init__()
        
        preprocessed_data_dir = os.path.join(preprocessed_data_dir_base, train_val_test)
        
        # Preprocess the data
        if not os.path.exists(preprocessed_data_dir):
            os.makedirs(preprocessed_data_dir)
            preprocess_data = True
        else:
            if preprocess_data and len(os.listdir(preprocessed_data_dir)) != 0:
                # Remove old files from the given directory
                filenames_old_files = os.listdir(preprocessed_data_dir)
                for filename in filenames_old_files:
                    os.remove(os.path.join(preprocessed_data_dir, filename))
        
        if preprocess_data or len(os.listdir(preprocessed_data_dir)) == 0:
            if train_val_test != 'test':
                data_dir = os.path.join(dataset_dir, 'train')
            else:
                data_dir = os.path.join(dataset_dir, 'test')
            
            # We first load the dataset
            texts = []
            labels = []
            for label_type in ['pos', 'neg']:
                dir_name = os.path.join(data_dir, label_type)
                dir_files = os.listdir(dir_name)
                dir_file_names = [filename for filename in dir_files if filename.endswith('.txt')]
                dir_file_names = sorted(dir_file_names, key=lambda x: (int(x.split('.')[0].split('_')[0]),
                                                                       int(x.split('.')[0].split('_')[1])))
                for file_name in dir_file_names:
                    with open(os.path.join(dir_name, file_name), 'r', encoding='utf-8') as f:
                        texts.append(f.read())
                    if label_type == 'pos':
                        labels.append(1)
                    else:
                        labels.append(0)
            
            # Then we split the training data into training and validation sets
            if train_val_test != 'test':
                texts = np.array(texts, dtype=object)
                labels = np.array(labels)
                train_val_permutation = np.random.RandomState(seed=random_seed).permutation(len(texts))
                num_train_texts = round(train_val_ratio * len(texts))
                train_indices = train_val_permutation[:num_train_texts]
                val_indices = train_val_permutation[num_train_texts:]
                if train_val_test == 'train':
                    texts = texts[train_indices]
                    labels = labels[train_indices]
                else:
                    texts = texts[val_indices]
                    labels = labels[val_indices]
                texts = texts.tolist()
                labels = labels.tolist()
            
            cls_embeddings = []
            model = BertModel.from_pretrained('bert-base-uncased')
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            
            for i in tqdm(range(len(texts))):
                
                # Next, we use the BERT tokenizer for the text data
                text_embedding = tokenizer(texts[i], padding=True, truncation=True,
                                           max_length=tokenizer_max_length, return_tensors='pt')
                
                # Now we use a pre-trained BERT model to compute the CLS embeddings for the data
                with no_grad():
                    outputs = model(text_embedding['input_ids'], attention_mask=text_embedding['attention_mask'])
                    cls_embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze().numpy())
            
            # Save the pre-processed features and labels
            np.save(os.path.join(preprocessed_data_dir, 'imdb_cls_features.npy'), np.array(cls_embeddings))
            np.save(os.path.join(preprocessed_data_dir, 'imdb_labels.npy'), np.array(labels))
            del texts
            del cls_embeddings
        
        # Load the pre-processed features and labels
        self.X = np.load(os.path.join(preprocessed_data_dir, 'imdb_cls_features.npy'))
        self.Y = np.load(os.path.join(preprocessed_data_dir, 'imdb_labels.npy'))
        
        if data_sampling_rate < 1.00 and train_val_test != 'test':
            # We randomly select a subset of the data
            num_sampled = int(data_sampling_rate * len(self.X))
            np.random.seed(3*random_seed)
            sampling_indices = np.random.choice(np.arange(len(self.X)), num_sampled, replace=False)
            self.X = self.X[sampling_indices, :]
            self.Y = self.Y[sampling_indices]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]




