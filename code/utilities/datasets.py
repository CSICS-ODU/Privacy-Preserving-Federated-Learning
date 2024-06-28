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
import pdb,traceback
from typing import List
import pandas as pd
import pprint
import matplotlib.pyplot as plt

from utilities.lib import blockPrinting
from utilities.cifar100_fine_coarse_labels import remapping
import pdb,traceback
from typing import List


class IncrementalDatasetWraper():
    def __init__(self, dataset_name = 'incremental_SVHN', data_path="~/datasets", audit_mode = False, addetive_train = False):
        self.name = dataset_name
        self.audit_mode = audit_mode
        self.splits = self._load_datasets(dataset_name)
        if addetive_train:
            self.splits = implement_addetive_dataset(self.splits, additive_train =True)

    def _load_datasets(self, dataset_name):
        if dataset_name == 'incrementalCIFAR100':
            data_splits = load_incremental_CIFAR100(remapping=[[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15],[16,17,18,19]], uniform_test = True)
        elif dataset_name == 'incrementaltestCIFAR100':
            data_splits = load_incremental_CIFAR100(remapping=[[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15],[16,17,18,19]], uniform_test = False)
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



class DatasetWrapper():
    def __init__(self, dataset_name = 'CIFAR10', data_path="~/dataset"):
        
        self.data_path = data_path
        self.name = dataset_name
        self.trainset, self.testset, self.num_channels, self.num_classes = self._load_datasets(dataset_name)


    # @blockPrinting  
    def _load_datasets(self, dataset_name):
        if dataset_name == 'CIFAR10':
            return load_CIFAR10(self.data_path)
        elif dataset_name == 'CIFAR100':
            return load_CIFAR100(self.data_path)
        elif dataset_name == 'MNIST':
            return load_MNIST(self.data_path)
        elif dataset_name == 'FashionMNIST':
            return load_FashionMNIST(self.data_path)
        elif dataset_name == "SVHN":
            return load_SVHN(self.data_path)
        else:
            # import pdb; pdb.set_trace()
            print(f'Unknown dataset name: {dataset_name}')
            raise NotImplementedError
    
   


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


def load_incremental_CIFAR100(remapping= None, uniform_test = False):
    trainset, testset, num_channels, _ = load_CIFAR100()
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
    # Assuming CIFAR-100 has 100 classes, grouped into 10 subsets
    class_groups = {k: [] for k in range(num_subsets)}  # Dict to hold subsets

    # Iterate through the dataset to group indices by class
    for idx, (_, label) in enumerate(dataset):
        # group_key = label // (100 // num_subsets)  # Determine the subset group
        group_key = label % num_subsets  # Group by modulo 10 of the label
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



def split_dataloaders(trainset, testset, num_splits: int, split_test = False, val_percent = 10, batch_size=32):#-> tuple[List, List, DataLoader, DataLoader]: 
    

    # Split training set into `num_clients` partitions to simulate different local datasets
    total_size = len(trainset)
    partition_size = total_size // num_splits
    lengths = [partition_size] * num_splits
    lengths[-1] += total_size% num_splits          # adding the reminder to the last partition

    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    val_datasets = []
    for ds in datasets:
        if val_percent == 0:
            len_val = 0
        else:
            len_val = len(ds) // val_percent  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        try:            
            trainloaders.append(DataLoader(ds_train, batch_size, shuffle=True))
            valloaders.append(DataLoader(ds_val, batch_size))
        except Exception as e:
            traceback.print_exc()
            pdb.set_trace()

        
        val_datasets.append(ds_val)
    if split_test:
        total_size = len(testset)
        partition_size = total_size // num_splits
        lengths = [partition_size] * num_splits
        lengths[-1] += total_size% num_splits          # adding the reminder to the last partition

        datasets = random_split(testset, lengths, torch.Generator().manual_seed(42))
        testloaders = []
        for ds in datasets:
            testloaders.append(DataLoader(ds, batch_size))
        unsplit_valloader = None
    else: 
        testloader = DataLoader(testset, batch_size)
        unsplit_valloader = DataLoader(torch.utils.data.ConcatDataset(val_datasets), batch_size) #type:ignore

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


def load_partitioned_datasets(num_clients: int, dataset_name = 'CIFAR10', data_path="~/datasets", val_percent = 10, batch_size=32, split=None):
    if split is None:
        dataset = DatasetWrapper(dataset_name, data_path)
        return split_dataloaders(dataset.trainset, dataset.testset, num_clients, split_test=False,val_percent=val_percent, batch_size=batch_size), dataset.num_channels, dataset.num_classes
    else:
        continous_datasets = IncrementalDatasetWraper(dataset_name, data_path)
        dataset = continous_datasets.splits[split]
        [train_dataset, test_dataset, num_channels, num_classes] = dataset
        return split_dataloaders(train_dataset, test_dataset, num_clients, split_test=False, val_percent=val_percent, batch_size=batch_size), num_channels, num_classes 
