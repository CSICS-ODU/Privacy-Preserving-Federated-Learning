import cv2
import csv
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST, CIFAR100, SVHN, FashionMNIST
from torch.utils.data import  Dataset, DataLoader, ConcatDataset, Subset, TensorDataset, random_split
from torch.utils.data import IterableDataset
import pdb,traceback
from typing import List
import pandas as pd
import pprint
import matplotlib.pyplot as plt
import pickle

from src.utilities.lib import blockPrinting
from src.utilities.cifar100_fine_coarse_labels import remapping
import pdb,traceback
from typing import List


def get_remapping(choice='ABCD', n=20):
    # Generate initial mapping based on consecutive division
    
    initial_mapping = [list(range(i, min(i + n // len(choice), n))) for i in range(0, n, n // len(choice))]
    
    # Create a dictionary to hold the remappings
    remapping = [initial_mapping[ord(c) - ord('A')] for c in choice]
    
    return remapping

class IncrementalDatasetWraper():
    def __init__(self, dataset_name = 'incremental_SVHN', data_path="~/datasets", audit_mode = False, addetive_train = False):
        self.name = dataset_name
        self.audit_mode = audit_mode
        self.splits = self._load_datasets(dataset_name)
        if addetive_train:
            self.splits = implement_addetive_dataset(self.splits, additive_train =True)

    def _load_datasets(self, dataset_name, data_path="~/dataset"):
        try: 
            dataset_name, remapping = dataset_name.split('=')
        except: 
            remapping = 'ABCD'
        
        if dataset_name == 'incrementalSVHN':
            remapping = get_remapping(remapping, n=10)
            data_splits = load_incremental_local_SVHN(data_path, remapping=remapping, uniform_test = True)
        elif dataset_name == 'incrementaltestSVHN':
            remapping = get_remapping(remapping, n=10)
            data_splits = load_incremental_local_SVHN(data_path, remapping=remapping, uniform_test = False)
        elif dataset_name == 'incrementalCIFAR100':
            remapping = get_remapping(remapping, n=20)
            data_splits = load_incremental_CIFAR20(data_path, remapping=remapping, uniform_test = True)
        elif dataset_name == 'incrementaltestCIFAR100':
            remapping = get_remapping(remapping, n=20)
            data_splits = load_incremental_CIFAR20(data_path, remapping=remapping, uniform_test = False)
        elif dataset_name == 'incrementalCIFAR10' or dataset_name == 'incrementalMNIST' or dataset_name == 'incrementalFashionMNIST':
            remapping = get_remapping(remapping, n=10)
            data_splits = load_incremental(dataset_name, data_path, remapping=remapping, uniform_test = True)
        elif dataset_name == 'incrementaltestCIFAR10' or dataset_name == 'incrementaltestMNIST' or dataset_name == 'incrementaltestFashionMNIST':
            remapping = get_remapping(remapping, n=10)
            data_splits = load_incremental(dataset_name, data_path, remapping=remapping, uniform_test = False)
        else:
            print(f'Unknown dataset name: {dataset_name}')
            raise NotImplementedError
        
        # for index, (train_subset, test_subset, num_channels, num_classes) in enumerate(data_splits):
        #     modified_trainset, modified_testset = remap_dataset(self.audit_mode, train_subset, test_subset)
        #     updated_split = (modified_trainset, modified_testset, num_channels, num_classes)
        #     data_splits[index] = updated_split

        return data_splits
        
    def select_split(self, split):
        self.trainset, self.testset, self.num_channels, self.num_classes = self.splits[split]
        self.data_split = [self.trainset, self.testset, self.num_channels, self.num_classes]

def load_datasets_by_name(dataset_name, data_path ):
        if dataset_name == 'CIFAR10':
            return load_CIFAR10(data_path)
        elif dataset_name == 'CIFAR100':
            return load_CIFAR100(data_path)
        elif dataset_name == 'MNIST':
            return load_MNIST(data_path)
        elif dataset_name == 'FashionMNIST':
            return load_FashionMNIST(data_path)
        elif dataset_name == "SVHN":
            return load_SVHN(data_path)
        else:
            # import pdb; pdb.set_trace()
            print(f'Unknown dataset name: {dataset_name}')
            raise NotImplementedError
        

# class DatasetWrapper(IterableDataset):
#     def __init__(self, dataset):
#         """
#         Initialize the dataset wrapper with support for streaming data.
#         # """
#         # self.data_path = data_path
#         # self.name = dataset_name
#         # self.mode = mode  # 'train' or 'test'

#         # Optionally lazy-load datasets
#         # self.trainset, self.testset, self.num_channels, self.num_classes = self._load_datasets(dataset_name)

#         self.dataset = dataset

#     # def _load_datasets(self, dataset_name):
#     #     """
#     #     Load datasets by name. For streaming, you might load incrementally here.
#     #     """
#     #     # if self.mode == 'train':
#     #         # Simulate loading a training dataset (e.g., CIFAR10)
#     #     return load_datasets_by_name(dataset_name, self.data_path)

#     def __iter__(self):
#         """
#         Yield batches iteratively for streaming.
#         """
#         for sample in iter(self.dataset):
#             yield sample

#     def __len__(self):
#         """
#         Optional: Return length of dataset if possible (for compatibility).
#         """
#         return len(self.dataset)

class DatasetWrapper():
    def __init__(self, dataset_name = 'CIFAR10', data_path="~/dataset"):
        
        self.data_path = data_path
        self.name = dataset_name
        self.trainset, self.testset, self.num_channels, self.num_classes = self._load_datasets(dataset_name)


    # @blockPrinting  
    def _load_datasets(self, dataset_name):
        return load_datasets_by_name(dataset_name, self.data_path )
   


def load_CIFAR10(data_path="~/dataset"):
    # Download and transform CIFAR-10 (train and test)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10(root=data_path, train=True, download=True, transform=transform)
    testset = CIFAR10(root=data_path, train=False, download=True, transform=transform)

    num_channels=3
    num_classes=10


    #full_dataset = ConcatDataset([train_dataset,test_dataset])

    #train_size = int(len(full_dataset)*train_percent)
    #test_size = len(full_dataset) - train_size

    #trainset, testset = random_split(full_dataset, [train_size, test_size])

    return trainset, testset, num_channels, num_classes
    # return train

def load_CIFAR100(data_path="~/dataset"):
    # Download and transform CIFAR-100 (train and test)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR100(root=data_path, train=True, download=True, transform=transform)
    testset = CIFAR100(root=data_path, train=False, download=True, transform=transform)

    num_channels = 3
    num_classes = 100

    return trainset, testset, num_channels, num_classes

def load_SVHN(data_path="~/dataset"):
    # Download and transform SVHN (train and test)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = SVHN(root=data_path, split="train", download=True, transform=transform)
    testset = SVHN(root=data_path, split="test", download=True, transform=transform)

    num_channels=3
    num_classes = 10

    return trainset, testset, num_channels, num_classes

def load_MNIST(data_path="~/dataset"):
    # Download and transform MNIST (train and test)
    transform = transforms.Compose(
        [transforms.Grayscale(num_output_channels=3), #expand to 3 channels
        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = MNIST(root=data_path, train=True, download=True, transform=transform)
    testset = MNIST(root=data_path, train=False, download=True, transform=transform)


    num_channels = 3
    #num_channels = 1
    num_classes = 10

    return trainset, testset, num_channels, num_classes


def load_FashionMNIST(data_path="~/dataset"):
    # Download and transform FashionMNIST (train and test)
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),  # Expand to 3 channels
            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    trainset = FashionMNIST(root=data_path, train=True, download=True, transform=transform)
    testset = FashionMNIST(root=data_path, train=False, download=True, transform=transform)

    num_channels = 3  # Set to 3 channels
    num_classes = 10

    return trainset, testset, num_channels, num_classes

def load_incremental_local_SVHN(data_path, remapping, uniform_test):
    data_path = os.path.expanduser(data_path)
    splits_paths = [
        os.path.join(data_path, 'SVHN','extra_A'),
        os.path.join(data_path, 'SVHN','extra_B'),
        os.path.join(data_path, 'SVHN','extra_C'),
        os.path.join(data_path, 'SVHN','train_cropped_images'),
        os.path.join(data_path, 'SVHN','test_cropped_images')
    ]

    return load_incremental_local_dataset(splits_paths, remapping, uniform_test)

def load_incremental_local_dataset(splits_paths, remapping, uniform_test = True):
    data_splits = []
    print('Loading custom incremental dataset...')
    for directory in tqdm(splits_paths, leave=False):
        train_dataset, test_dataset, num_channels, num_classes = load_custom_image_dataset(directory, test_size=0.4)
        data_splits.append((train_dataset, test_dataset, num_channels, num_classes))

    # combine the train test and the extras into a single monolithic dataset
    data_splits = combine_subsets(data_splits, [list(range(len(data_splits)))])

    #now seperate the monolithic dataset into 10 subsets [0-9]

    
    train_subsets = split_dataset_into_subsets(data_splits[0][0], num_classes)
    test_subsets = split_dataset_into_subsets(data_splits[0][1], num_classes)

    


    # Combine the train and test subsets along with num_channels and num_classes into a list of tuples
    data_splits = [(train_subsets[i], test_subsets[i], num_channels, num_classes) for i in range(len(train_subsets))]
    # mix the  classes in the dataset together
    data_splits = mix_subsets(data_splits)

    # combine the subsets with remapping
    data_splits = combine_subsets(data_splits, remapping)


    if uniform_test:
        data_splits = implement_combined_uniform_test(data_splits)
    else:
        data_splits = implement_addetive_dataset(data_splits)

    return data_splits

def load_custom_image_dataset(directory, test_size=0.4):
    images = []
    labels = []
    try:
        train_dataset, test_dataset, num_channels, num_classes =  load_pickle(directory+'.pkl')
    except:
        print(f'\nPresaved dataset not found, Loading custom dataset from {directory}')
        for label in tqdm(os.listdir(directory), leave=False):
            label_dir = os.path.join(directory, label)
            for img_file in tqdm(os.listdir(label_dir), leave=False):
                img_path = os.path.join(label_dir, img_file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (32, 32))  # Resize image to a fixed size
                images.append(img)
                labels.append(int(label))
    
        # Transform the dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        dataset = [(transform(img), label) for img, label in zip(images, labels)]
    
        # Calculate the sizes of the train and test sets based on the test_size ratio
        test_size = int(len(dataset) * test_size)
        train_size = len(dataset) - test_size
    
        # Split the dataset into training and test sets
        torch.manual_seed(42)
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size]) # type: ignore
    
        num_channels = 3
        num_classes = len(np.unique(labels))

        try:
            save_pickle( (train_dataset, test_dataset, num_channels, num_classes), directory+'.pkl')
        except Exception as e:
            print('Error saving dataset:', e)


   
    return train_dataset, test_dataset, num_channels, num_classes

def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)



def load_incremental(dataset_name, data_path, remapping= None, uniform_test = False):
    if 'test' in dataset_name:
        datasetname = dataset_name.replace('incrementaltest','')
    else:
        datasetname = dataset_name.replace('incremental','')
        
    trainset, testset, num_channels, num_classes = load_datasets_by_name(datasetname, data_path )

    
    #now seperate the monolithic dataset into 10 subsets [0-9]
    train_subsets = split_dataset_into_subsets(trainset, num_classes)
    test_subsets = split_dataset_into_subsets(testset, num_classes)
    

    # Combine the train and test subsets along with num_channels and num_classes into a list of tuples
    data_splits = [(train_subsets[i], test_subsets[i], num_channels, num_classes) for i in range(len(train_subsets))]
    # mix the  classes in the dataset together
    data_splits = mix_subsets(data_splits)

    # combine the subsets with remapping
    data_splits = combine_subsets(data_splits, remapping)


    if uniform_test:
        data_splits = implement_combined_uniform_test(data_splits)
    else:
        data_splits = implement_addetive_dataset(data_splits)

    return data_splits
    
    

def load_incremental_CIFAR20(data_path, remapping= None, uniform_test = False):
    trainset, testset, num_channels, _ = load_CIFAR100(data_path)
    num_classes = 20
    # Split both train and test sets into 20 subsets
    train_subsets = split_dataset_into_subsets(trainset, num_classes)
    test_subsets = split_dataset_into_subsets(testset, num_classes)

    train_subsets = [CIFAR_20_Dataset(subset) for subset in train_subsets]
    test_subsets = [CIFAR_20_Dataset(subset) for subset in test_subsets]


    # Combine the train and test subsets along with num_channels and num_classes into a list of tuples
    data_splits = [(train_subsets[i], test_subsets[i], num_channels, num_classes) for i in range(len(train_subsets))]

    data_splits = mix_subsets(data_splits)
    if uniform_test:
        data_splits = implement_combined_uniform_test(data_splits)
    else:
        data_splits = implement_addetive_dataset(data_splits)


    if remapping is not None:
        data_splits = combine_subsets(data_splits, remapping)

    return data_splits

def split_dataset_into_subsets(dataset, num_subsets=10):
    # without assuming the number of classes, grouped into 'num_subsets' subsets
    class_groups = {k: [] for k in range(num_subsets)}  # Dict to hold subsets

    # Iterate through the dataset to group indices by class
    for idx, (_, label) in enumerate(dataset):
        # group_key = label // (100 // num_subsets)  # Determine the subset group
        group_key = label % num_subsets  # Group by modulo 'num_subsets' of the label
        class_groups[group_key].append(idx)

    # Create a subset for each group
    subsets = [Subset(dataset, indices) for indices in class_groups.values()]
    return subsets

def remap_dataset(audit_mode, trainset, testset,  train_percent = 0.35, test_percent = 0.35 , audit_percent = 0.3, preserve_original_propertion = True):
    """
    Remaps the given train and test datasets based on the provided percentages. Holds out a portion of the training set for auditing purposes. 
    Depending on the wheather the audit_mode flag is set, the train and test sets are returned in different ways.

    Args:
    - trainset: The training dataset to be remapped.
    - testset: The testing dataset to be remapped.
    - train_percent: The percentage of samples to allocate to the training set (default is 0.35).
    - test_percent: The percentage of samples to allocate to the testing set (default is 0.35).
    - audit_percent: The percentage of samples to allocate to the audit set (default is 0.3).
    - preserve_original_propertion: A boolean indicating whether to preserve the original proportion of the training set overwriting the  train and test percentages (default is False).

    Returns:
    - train_set: The remapped training dataset.
    - test_set: The remapped testing dataset.
    """
    # Concatenate train and test sets
    full_dataset = ConcatDataset([trainset, testset])

    if preserve_original_propertion:        
        original_train_percentage = len(trainset) /len(full_dataset)
        train_total_percentage  = 1 - audit_percent        
        train_percent = original_train_percentage * train_total_percentage
        test_percent = (1-original_train_percentage) * train_total_percentage

    # Determine sizes of subsets
    num_samples = len(full_dataset)

    audit_train_percent = audit_percent *0.6  
    
    train_size = int(num_samples * train_percent)
    test_size = int(num_samples * test_percent)   
    audit_train_size = int(num_samples * audit_train_percent)
    audit_test_size = num_samples - (audit_train_size + train_size + test_size)       
        

    # Split the concatenated dataset into subsets
    train_set, test_set, audit_train_set, audit_test_set = random_split(full_dataset, [train_size, test_size, audit_train_size, audit_test_size], torch.Generator().manual_seed(42) )
    
        
    
    if audit_mode:
        train_set = audit_train_set
        test_set = audit_test_set
    return train_set, test_set

def combine_subsets(data_splits, subsets_groups):
    new_data_splits = []
    for group in subsets_groups:
        if isinstance(group, list):  # Group is a list of indices to combine
            train_datasets = [data_splits[i][0] for i in group]
            test_datasets = [data_splits[i][1] for i in group]
            # Assume num_channels and num_classes are consistent within the group
            num_channels = data_splits[group[-1]][2]
            num_classes = data_splits[group[-1]][3]
            combined_train_dataset = ConcatDataset(train_datasets)
            combined_test_dataset = ConcatDataset(test_datasets)
            new_data_splits.append((combined_train_dataset, combined_test_dataset, num_channels, num_classes))
        else:
            # Group is a single index, include as is
            new_data_splits.append(data_splits[group])
    return new_data_splits

def implement_addetive_dataset(data_splits, additive_train =False):
    new_data_splits = []
    expanding_dataset = []
    for i, split in tqdm(enumerate(data_splits), leave=False):
        train_dataset_i, test_dataset_i, num_channels, num_classes = split
        if additive_train:
            expanding_dataset.append(train_dataset_i)
            split = (ConcatDataset(expanding_dataset), test_dataset_i, num_channels, num_classes)
        else:
            expanding_dataset.append(test_dataset_i)
            split = (train_dataset_i, ConcatDataset(expanding_dataset), num_channels, num_classes)
        new_data_splits.append(split)
    return new_data_splits

def implement_combined_uniform_test(data_splits):    
    expanding_dataset = []
    for _, split in tqdm(enumerate(data_splits), leave=False):
        _, test_dataset_i, _, _ = split
        expanding_dataset.append(test_dataset_i)    
    combined_uniform_test= ConcatDataset(expanding_dataset)

    new_data_splits = []    
    for i, split in tqdm(enumerate(data_splits), leave=False):
        train_dataset_i, _, num_channels, num_classes = split
        split = (train_dataset_i, combined_uniform_test, num_channels, num_classes)
        new_data_splits.append(split)
    return new_data_splits


def get_mixing_proportions(num_classes=20, seed_value=42):

    # Set the seed for reproducibility
    np.random.seed(seed_value)

    # Generate a random matrix
    matrix = np.random.rand(num_classes, num_classes)

    # Normalize each row to sum up to 1
    matrix_normalized_row = matrix / matrix.sum(axis=1)[:, np.newaxis]

    # Normalize each column to sum up to 1
    matrix_normalized = matrix_normalized_row / matrix_normalized_row.sum(axis=0)

    # Round the normalized matrix to 2 decimal places
    matrix_rounded = np.around(matrix_normalized, decimals=2)

    # Adjust each row to sum to 1, correcting for rounding errors
    for i in range(num_classes):
        row_diff = 1 - matrix_rounded[i, :].sum()
        matrix_rounded[i, np.argmax(matrix_rounded[i, :])] += row_diff

    # Adjust each column to sum to 1, correcting for rounding errors
    for j in range(num_classes):
        col_diff = 1 - matrix_rounded[:, j].sum()
        matrix_rounded[np.argmax(matrix_rounded[:, j]), j] += col_diff

    return matrix_rounded

def mix_subsets(subsets, proportions=None, seed_value=42):
    """
    Mix subsets according to user-defined proportions.

    Args:
    - subsets: A list of dataset subsets to mix.
    - proportions: A list of proportions for each subset.
    
    Returns:
    - A new dataset consisting of mixed subsets.
    """
    num_splits = len(subsets)
    if proportions is None:
        proportions = get_mixing_proportions(num_splits, seed_value)
    else:
        assert num_splits == len(proportions)

    # Initialize empty lists for the new subsets
    new_train_datasets = [[] for _ in range(num_splits)]
    new_test_datasets = [[] for _ in range(num_splits)]
    new_data_splits = []

    generator = torch.Generator().manual_seed(seed_value)


    try:

        # Loop through each original subset
        for i, subset in enumerate(subsets):
            trainset_i, testset_i, num_channels, num_classes = subset
            
            # Calculate lengths for the new subsets
            lengths_train = [int(p * len(trainset_i)) for p in proportions[i]]
            lengths_test = [int(p * len(testset_i)) for p in proportions[i]]
            

            #fix for rounding errors
            lengths_train[-1] = len(trainset_i)-sum(lengths_train[:-1]) 
            lengths_test[-1] = len(testset_i)-sum(lengths_test[:-1])

            # Split the original subsets into new subsets based on the calculated lengths
            train_splits = random_split(trainset_i, lengths_train, generator=generator)
            test_splits = random_split(testset_i, lengths_test, generator=generator)

            # Accumulate the splits into the corresponding new datasets arrays
            for i, split in enumerate(train_splits):
                new_train_datasets[i].append(split)  
            
            for i, split in enumerate(test_splits):
                new_test_datasets[i].append(split)  

        # Now, concatenate the accumulated subsets
        for i, (trn_datasets, tst_datasets) in enumerate(zip(new_train_datasets, new_test_datasets)):
            split = (ConcatDataset(trn_datasets), ConcatDataset(tst_datasets), num_channels, num_classes)
            new_data_splits.append(split)
    except Exception as e:
        traceback.print_exc()
        pdb.set_trace()

    return new_data_splits

class CIFAR_20_Dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.remap = remapping()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # # Modify the label to be the new label based on modulo 10
        # mod_label = label % 10
        coarse_label = self.remap.fine_id_coarse_id[label]
        return img, coarse_label

class Loss_Label_Dataset(Dataset):
    """Loss_label_Dataset."""

    def __init__(self, original_dataset, target_model, device, batch_size = 32, loss_batchwise = False):
        self.batch_size         = batch_size  
        self.loss_batchwise     = loss_batchwise
        trainset                = original_dataset[0]
        testset                 = original_dataset[1]
        seen_count              = trainset.dataset.__len__()
        unseen_count            = testset.dataset.__len__()
        self.target_model       = target_model
        self.device             = device

        try:
            assert abs(seen_count - unseen_count) < seen_count/10  # roughly ballanced dataset
            # print(f'Ballanced dataset: seen {seen_count}, unseen {unseen_count}')
        except AssertionError as e:
            type  = 'batchwise' if loss_batchwise else 'samplewise'
            print(f'\tUnballanced {type} dataset: seen {seen_count}, unseen {unseen_count}')
            # pdb.set_trace()

        self.data   = []
        self.label  = []

        self.append_data_label(trainset, 1.0)
        self.append_data_label(testset, 0.0)

        # pdb.set_trace()
        

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):        
        sample = [self.data[idx], self.label[idx]]
        return sample
    
    def append_data_label(self, dataLoader, seen_unseen_label, criterion=None):
        if not criterion:
            criterion = torch.nn.CrossEntropyLoss( )


        for images, labels in dataLoader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.target_model(images)
            if self.loss_batchwise:
                loss = criterion(outputs, labels).item()
                self.data.append(loss)
                self.label.append(seen_unseen_label)               

            else:
                for i, label in enumerate(labels):
                    instance_loss = criterion(outputs[i], label).item()
                    self.data.append(instance_loss)
                    self.label.append(seen_unseen_label)

        return 


class Wrapper_Dataset(Dataset):
    def __init__(self, data, label):
        self.data   = data
        self.label  = label
         
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):        
        sample = [self.data[idx], self.label[idx]]
        return sample

    
class Error_Label_Dataset(Loss_Label_Dataset):
    def __init__(self, original_dataset, target_model, device, batch_size=32):
        super().__init__(original_dataset, target_model, device, batch_size)

    def append_data_label(self, dataLoader, seen_unseen_label, criterion=None):
        if not criterion:
            criterion = torch.nn.CrossEntropyLoss()


        for images, _ in dataLoader:
            images  = images.to(self.device)
            outputs = self.target_model(images)           

            # pdb.set_trace()

            self.data.append(outputs)
            self.label.append(seen_unseen_label)

        return 



# def split_dataloaders(trainset, testset, num_splits: int, split_test = False, val_percent = 10, batch_size=32):#-> tuple[List, List, DataLoader, DataLoader]: 
    

#     # Split training set into `num_clients` partitions to simulate different local datasets
#     total_size = len(trainset)
#     partition_size = total_size // num_splits
#     lengths = [partition_size] * num_splits
#     lengths[-1] += total_size% num_splits          # adding the reminder to the last partition

#     datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

#     # Split each partition into train/val and create DataLoader
#     trainloaders = []
#     valloaders = []
#     val_datasets = []
    
#     for ds in datasets:
#         if val_percent == 0:
#             len_val = 0
#         else:
#             len_val = len(ds) // val_percent  # 10 % validation set
#         len_train = len(ds) - len_val
#         lengths = [len_train, len_val]
#         ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
#         try:            
#             trainloaders.append(DataLoader(ds_train, batch_size, shuffle=True))
#             valloaders.append(DataLoader(ds_val, batch_size))
#         except Exception as e:
#             traceback.print_exc()
#             pdb.set_trace()

        
#         val_datasets.append(ds_val)
#     if split_test:
#         total_size = len(testset)
#         partition_size = total_size // num_splits
#         lengths = [partition_size] * num_splits
#         lengths[-1] += total_size% num_splits          # adding the reminder to the last partition

#         datasets = random_split(testset, lengths, torch.Generator().manual_seed(42))
#         testloaders = []
#         for ds in datasets:
#             testloaders.append(DataLoader(ds, batch_size))
#         unsplit_valloader = None
#     else: 
#         testloader = DataLoader(testset, batch_size)
#         unsplit_valloader = DataLoader(torch.utils.data.ConcatDataset(val_datasets), batch_size) #type:ignore

#     return trainloaders, valloaders, testloader, unsplit_valloader

def validate_mapping(mapping, num_classes):
    num_clients = len(mapping)
    class_percentages = {i: 0.0 for i in range(num_classes)}

    for client, class_map in mapping.items():
        for class_idx, percentage in class_map.items():
            if class_idx >= num_classes:
                raise ValueError(f"Class index {class_idx} exceeds number of classes {num_classes}")
            class_percentages[class_idx] += percentage

    tolerance = 1e-5
    for class_idx, total in class_percentages.items():
        if total < 1.0 - tolerance:
            print(f"Class {class_idx} percentages sum to {total:.2f}, less than 100%.")
        elif total > 1.0 + tolerance:
            print(f"Class {class_idx} percentages sum to {total:.2f}, exceeds 100%. Overlap will be used.")
    print("Mapping validation completed.")

def split_dataloaders(trainset, testset, mapping, val_percent=10, batch_size=32, split_test=False):
    num_classes = len(trainset.classes)
    num_clients = len(mapping)

    validate_mapping(mapping, num_classes)

    class_indices = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(trainset):
        class_indices[label].append(idx)

    for class_idx in class_indices:
        np.random.shuffle(class_indices[class_idx])

    client_train_indices = {client: [] for client in mapping}
    client_val_indices = {client: [] for client in mapping}
    client_class_indices = {client: {i: [] for i in range(num_classes)} for client in mapping}  # Track per-class indices

    for class_idx in range(num_classes):
        total_indices = class_indices[class_idx].copy()  # Copy to preserve original
        total_samples = len(total_indices)
        remaining_samples = total_samples

        if remaining_samples == 0:
            continue

        start_idx = 0
        client_list = [(client, class_map[class_idx]) for client, class_map in mapping.items() if class_idx in class_map]

        # First pass: Sequential split up to 100% or until data runs out
        for client, percentage in client_list:
            num_samples = int(remaining_samples * percentage)
            if num_samples == 0 and percentage > 0:
                num_samples = 1

            end_idx = min(start_idx + num_samples, len(total_indices))
            client_indices = total_indices[start_idx:end_idx]

            # Split into train and validation
            if val_percent > 0:
                val_size = int(len(client_indices) * (val_percent / 100))
                train_size = len(client_indices) - val_size
            else:
                val_size = 0
                train_size = len(client_indices)

            train_indices = client_indices[:train_size]
            val_indices = client_indices[train_size:]

            client_train_indices[client].extend(train_indices)
            client_val_indices[client].extend(val_indices)
            client_class_indices[client][class_idx].extend(client_indices)  # Store for overlap later

            remaining_samples -= (end_idx - start_idx)
            start_idx = end_idx

            if remaining_samples <= 0:
                break

        # Second pass: Handle excess percentage (>100%)
        total_percentage = sum(class_map.get(class_idx, 0) for class_map in mapping.values())
        if total_percentage > 1.0 + 1e-5:
            for client, percentage in client_list:
                required_samples = int(total_samples * percentage)
                current_samples = len(client_class_indices[client][class_idx])
                if required_samples > current_samples:
                    excess_samples = required_samples - current_samples
                    donor_clients = [(c, client_class_indices[c][class_idx]) for c, m in mapping.items() 
                                     if class_idx in m and c != client and len(client_class_indices[c][class_idx]) > 0]
                    
                    while excess_samples > 0 and donor_clients:
                        for donor_client, donor_indices in donor_clients[:]:  # Copy to modify during iteration
                            donor_size = len(donor_indices)
                            max_overlap = int(donor_size * 0.5)  # Max 50% overlap
                            samples_to_take = min(max_overlap, excess_samples)
                            
                            if samples_to_take > 0:
                                overlap_indices = np.random.choice(donor_indices, samples_to_take, replace=False).tolist()
                                client_train_indices[client].extend(overlap_indices[:int(samples_to_take * (1 - val_percent / 100))])
                                client_val_indices[client].extend(overlap_indices[int(samples_to_take * (1 - val_percent / 100)):])
                                client_class_indices[client][class_idx].extend(overlap_indices)
                                excess_samples -= samples_to_take

                            if excess_samples <= 0:
                                break

    trainloaders = []
    valloaders = []
    val_datasets = []
    for client in mapping:
        train_dataset = Subset(trainset, client_train_indices[client])
        val_dataset = Subset(trainset, client_val_indices[client])

        trainloaders.append(DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True))
        valloaders.append(DataLoader(val_dataset, batch_size=batch_size, drop_last=True))
        val_datasets.append(val_dataset)

    if split_test:
        test_class_indices = {i: [] for i in range(num_classes)}
        for idx, (_, label) in enumerate(testset):
            test_class_indices[label].append(idx)

        client_test_indices = {client: [] for client in mapping}
        for class_idx in range(num_classes):
            total_indices = test_class_indices[class_idx].copy()
            remaining_samples = len(total_indices)
            start_idx = 0
            for client, class_map in mapping.items():
                if class_idx in class_map:
                    percentage = class_map[class_idx]
                    num_samples = int(remaining_samples * percentage)
                    if num_samples == 0 and percentage > 0:
                        num_samples = 1
                    if client == list(mapping.keys())[-1]:
                        num_samples = remaining_samples

                    end_idx = min(start_idx + num_samples, len(total_indices))
                    client_test_indices[client].extend(total_indices[start_idx:end_idx])
                    remaining_samples -= (end_idx - start_idx)
                    start_idx = end_idx
                    if remaining_samples <= 0:
                        break

            # Handle excess for test set (simplified, no overlap limit here)
            total_percentage = sum(class_map.get(class_idx, 0) for class_map in mapping.values())
            if total_percentage > 1.0 + 1e-5:
                for client, percentage in mapping.items():
                    if class_idx in percentage:
                        required_samples = int(len(test_class_indices[class_idx]) * percentage[class_idx])
                        current_samples = len(client_test_indices[client])
                        if required_samples > current_samples:
                            excess_samples = required_samples - current_samples
                            overlap_indices = np.random.choice(test_class_indices[class_idx], 
                                                              min(excess_samples, len(test_class_indices[class_idx])), 
                                                              replace=False).tolist()
                            client_test_indices[client].extend(overlap_indices)

        testloaders = [DataLoader(Subset(testset, client_test_indices[client]), batch_size=batch_size)
                       for client in mapping]
        unsplit_valloader = None
    else:
        testloader = DataLoader(testset, batch_size=batch_size)
        unsplit_valloader = DataLoader(torch.utils.data.ConcatDataset(val_datasets), batch_size=batch_size)

    return trainloaders, valloaders, testloader, unsplit_valloader

def load_dataloaders(trainset, testset, batch_size=32):
    trainloader    = DataLoader(trainset, batch_size, shuffle=True)
    testloader     = DataLoader(testset, batch_size)
    return trainloader,  testloader

def get_dataloaders_subset(dataloader, random_subset_size):
    dataset  = dataloader.dataset
    lengths  = [random_subset_size, len(dataset) - random_subset_size]
    truncated_dataset = random_split(dataset, lengths, torch.Generator().manual_seed(42))
    return DataLoader(truncated_dataset[0], dataloader.batch_size, shuffle=True)

def merge_dataloaders(trainloaders):    
    trn_datasets = []
    for loader in trainloaders:
        trn_datasets.append(loader.dataset)
    return DataLoader(ConcatDataset(trn_datasets), trainloaders[0].batch_size)


def load_partitioned_datasets(num_clients: int, dataset_name = 'CIFAR10', data_path="~/dataset", val_percent = 10, batch_size=32, split=None):
    if split is None:
        dataset = DatasetWrapper(dataset_name, data_path)
        
        # # 2 clients mapping - overlap
        # mapping = {
        #     "Client 1": {0: 0.60, 1: 1.0, 2: 0.40, 4: 0.50, 5: 0.70, 6: 0.70, 7: 0.30},
        #     "Client 2": {3: 0.55, 2: 0.70, 4: 0.80, 6: 0.65, 8: 1.0, 9: 0.95}
        # }

        # 2 clients mapping heterogenous - no overlap
        # mapping = {
        #     "Client 1": {0: 1.0, 1: 1.0, 2: 1.0, 4: 1.0, 7: 1.0, 8: 1.0},
        #     "Client 2": {3: 1.0, 5: 1.0, 6: 1.0, 9: 1.0}
        # }


        # 5 clients mapping - overlap
        # mapping = {
        #                     "Client 1": {0: 0.30, 2: 0.25, 5: 0.20, 6: 0.30},
        #                     "Client 2": {1: 0.60, 8: 0.20},
        #                     "Client 3": {4: 0.40, 7: 0.35, 9: 0.30},
        #                     "Client 4": {3: 0.45, 5: 0.50, 6: 0.60, 9: 0.70},
        #                     "Client 5": {0: 0.70, 1: 0.40, 2: 0.75, 3: 0.55, 4: 0.60, 5: 0.30, 6: 0.30, 7: 0.65, 8: 0.60}
        #                 } 

        # # 5 clients mapping heterogenous - no overlap
        # mapping = {
        #                     "Client 1": {0: 1.0, 2: 1.0, 5: 1.0},
        #                     "Client 2": {1: 1.0, 8: 1.0},
        #                     "Client 3": {4: 1.0, 7: 1.0},
        #                     "Client 4": {3: 1.0},
        #                     "Client 5": {9: 1.0, 6: 1.0}
        #                 } 
        
        # 10 clients mapping - overlap
        mapping =   {
                    "Client 1": {0: 0.20, 1: 0.25, 5: 0.30},           # Classes 0, 1, 5
                    "Client 2": {2: 0.25, 3: 0.20, 8: 0.10},           # Classes 2, 3, 8
                    "Client 3": {4: 0.35, 6: 0.25, 9: 0.20},           # Classes 4, 6, 9
                    "Client 4": {0: 0.30, 7: 0.40, 2: 0.15},           # Classes 0, 7, 2
                    "Client 5": {1: 0.25, 5: 0.40, 8: 0.30},           # Classes 1, 5, 8
                    "Client 6": {3: 0.35, 6: 0.30, 9: 0.25},           # Classes 3, 6, 9
                    "Client 7": {0: 0.50, 4: 0.20, 7: 0.60},           # Classes 0, 4, 7
                    "Client 8": {2: 0.40, 5: 0.20, 8: 0.40},           # Classes 2, 5, 8
                    "Client 9": {1: 0.40, 3: 0.25, 6: 0.35},           # Classes 1, 3, 6
                    "Client 10": {4: 0.45, 7: 0.30, 9: 0.75}           # Classes 4, 7, 9
                }

        # 10 clients mapping heterogenous - no overlap
        # mapping =   {
        #             "Client 1": {0: 0.20, 1: 0.25, 5: 0.30},           # Classes 0, 1, 5
        #             "Client 2": {2: 0.25, 3: 0.20, 8: 0.20},           # Classes 2, 3, 8
        #             "Client 3": {4: 0.35, 6: 0.35, 9: 0.20},           # Classes 4, 6, 9
        #             "Client 4": {0: 0.30, 7: 0.40, 2: 0.15},           # Classes 0, 7, 2
        #             "Client 5": {1: 0.25, 5: 0.40, 8: 0.40},           # Classes 1, 5, 8
        #             "Client 6": {3: 0.35, 6: 0.30, 9: 0.25},           # Classes 3, 6, 9
        #             "Client 7": {0: 0.50, 4: 0.20, 7: 0.60},           # Classes 0, 4, 7
        #             "Client 8": {2: 0.60, 5: 0.30, 8: 0.40},           # Classes 2, 5, 8
        #             "Client 9": {1: 0.50, 3: 0.45, 6: 0.35},           # Classes 1, 3, 6
        #             "Client 10": {4: 0.45, 9: 0.55}           # Classes 4, 9
        #         }

        return split_dataloaders(dataset.trainset, dataset.testset, mapping, split_test=False,val_percent=val_percent, batch_size=batch_size), dataset.num_channels, dataset.num_classes
    
    else:
        continous_datasets = IncrementalDatasetWraper(dataset_name, data_path)
        dataset = continous_datasets.splits[split]
        [train_dataset, test_dataset, num_channels, num_classes] = dataset
        return split_dataloaders(train_dataset, test_dataset, num_clients, split_test=False, val_percent=val_percent, batch_size=batch_size), num_channels, num_classes 