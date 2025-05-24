from collections import OrderedDict
import copy
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
# import wandb
import pdb,traceback
import os
import csv
import json



from src.utilities.datasets import Wrapper_Dataset
from src.utilities.plot_utils import plot_ROC_curve
from src.utilities.lib import  create_directories_if_not_exist

def wandb_init(
    project="Privacy_Preserving_Federated_Learning", 
    entity='', 
    model_name="basicCNN", 
    dataset_name="CIFAR_10",
    comment= '',  
    lr ='', 
    optimizer = ''
    ):
    config_path = os.path.join('wandb', 'config.json')
    with open(config_path) as config_file:
        config = json.load(config_file)  
        api_key = config.get('api_key') 
        if entity == '':
            entity = config.get('entity')
         
    # wandb.login( key=api_key )
    # wandb.init(
    #   project=project, entity=entity,
    #   config={"learning_rate": lr, "optimiser": optimizer, "comment" : comment, "model": model_name, "dataset": dataset_name}
    # )


def get_device():
    if torch.cuda.is_available():
      return torch.device("cuda")
    else:
      return torch.device("cpu")

def print_info(device, model_name="model", dataset_name="dataset", teacher_name=None, no_FL = False):
    if device.type=="cuda":
        device_type = torch.cuda.get_device_name(0)  
    else:
        device_type = device.type
        
    if teacher_name:
        print(f"\nDistiling {model_name} from {teacher_name} on {dataset_name} in {device_type} using PyTorch {torch.__version__}")
    elif no_FL:
        print(f"\n\tTraining {model_name} with {dataset_name} in {device_type} using PyTorch {torch.__version__}")
    else:
        try:
            import nvflare
            ver = nvflare.__version__
        except:
            ver = ''
        print(f"\nTraining {model_name} with {dataset_name} in {device_type} using PyTorch {torch.__version__} and NVFlare {ver}")

def verify_folder_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"{path} created")
    return path

def save_model(net, optim = None, filename ='filename', print_info=False):
    sanatized_filename = "".join(x for x in filename if x.isalnum())
    save_folder = './saved_models/'
    path = verify_folder_exist(save_folder)+sanatized_filename+'.pt'
    if optim:
        torch.save({'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optim.state_dict()
            }, path)
    else:
        torch.save({'model_state_dict': net.state_dict()}, path)

    if print_info:
        print(f"\nSaved model to {path}")

def load_model(net, optim=None, filename ='filename', print_info=True):
    try:
        #sanatized_filename = "".join(x for x in filename if x.isalnum())
        sanatized_filename = "".join(x for x in filename if x.isalnum() or x in "_=")

        print(f"SANITIZED FILENAME: {sanatized_filename}")
        path = '/home/akapo004/new_nvflare/saved_models/'+sanatized_filename+'.pth.tar'
        checkpoint = torch.load(path)
        # print("**************** checkpoint DICTIONARYYY ************", checkpoint.keys())


        # print("**************** checkpoint value ************", checkpoint)
        try:
            net.load_state_dict(checkpoint['model_state_dict'])
            # net.load_state_dict(checkpoint)         # Temporary fix here
            
        except  KeyError:
            net.load_state_dict(checkpoint) 

            
        if optim:
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
        if print_info:
            print(f"Loaded model from {path}")
    except Exception as e:
        traceback.print_exc()
        pdb.set_trace()

def delete_saved_model(filename ='filename', print_info=False):
    sanatized_filename = "".join(x for x in filename if x.isalnum())
    path = './saved_models/'+sanatized_filename+'.pt'
    if os.path.exists(path):
        os.remove(path)
    if print_info:
        print(f"Deleted model from {path}")


def save_loss_dataset(dataset, filename='datset'):    
    print(f'\tSaving dataset of size {len(dataset)} to {filename}')
    save_path = './saved_models/'+filename+'.csv'
    create_directories_if_not_exist(save_path)
    with open(save_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(['data', 'label'])  # Write the header row

        for data, label in dataset:
            # Convert tensors to numpy arrays if needed
            data = data.numpy() if isinstance(data, torch.Tensor) else data
            label = label.numpy() if isinstance(label, torch.Tensor) else label

            writer.writerow([data, label])  # Write each data and label as a row





def load_loss_dataset(filename='dataset'):
    print(f'\tLoading dataset from {filename}')
    load_path = './saved_models/' + filename + '.csv'
    dataset = []

    with open(load_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        data = []
        label = []

        for row in reader:
            data_i, label_i = row
            # Convert data and label to appropriate types if needed
            try:
                data.append(eval(data_i))
                label.append(eval(label_i))
            except:
                traceback.print_exc()
                # pdb.set_trace()

            

        dataset = Wrapper_Dataset(data, label)

    return dataset


class Trainer():

    def __init__(self, model, trainloader, valloader, testloader, optimizer = None, criterion = None, device = get_device(), is_binary=False, summary_writer=None):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.is_binary = is_binary
        self.epochs = 50
        self.summary_writer = summary_writer
        
    @classmethod
    def from_trainer(cls, existing_trainer, model=None, trainloader=None, valloader=None, testloader=None, optimizer=None, criterion=None, device=None, is_binary=None, summary_writer=None):
        return cls(
            model=model if model is not None else existing_trainer.model,
            trainloader=trainloader if trainloader is not None else existing_trainer.trainloader,
            valloader=valloader if valloader is not None else existing_trainer.valloader,
            testloader=testloader if testloader is not None else existing_trainer.testloader,
            optimizer=optimizer if optimizer is not None else existing_trainer.optimizer,
            criterion=criterion if criterion is not None else existing_trainer.criterion,
            device=device if device is not None else existing_trainer.device,
            is_binary=is_binary if is_binary is not None else existing_trainer.is_binary,
            summary_writer=summary_writer if summary_writer is not None else existing_trainer.summary_writer
        )

def train_single_epoch(trainer, epoch, steps, round_no = 1, verbose=False, wandb_logging=True):
    """Train the network on the training set."""
    if not trainer.criterion:
        trainer.criterion = torch.nn.CrossEntropyLoss()
    if not trainer.optimizer:
        trainer.optimizer = torch.optim.Adam(trainer.model.parameters())
    trainer.model.train()    
    correct, total, epoch_loss = 0, 0, 0.0
    running_loss = 0.0
    for i, (images, labels) in enumerate(trainer.trainloader):
        try:
            if  len(labels) <= 1: #ignore single sample batches
                break
            images, labels = images.to(trainer.device), labels.to(trainer.device)
            trainer.optimizer.zero_grad()
            outputs = trainer.model(images)
            loss = trainer.criterion(outputs, labels)
            loss.backward()
            trainer.optimizer.step()
            # Metrics
            epoch_loss += loss.item()
            total += labels.size(0)
            if trainer.is_binary:
                correct += (torch.round(outputs.data) == labels).sum().item()
            else:
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

            
            if i % 2000 == 1999:  # print every 2000 mini-batches
                running_loss = epoch_loss -running_loss
                if verbose:
                    print(f"[round {round_no}, epoch {epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f} ")
                if trainer.summary_writer:
                    global_step = round_no * steps + epoch * len(trainer.trainloader) + i
                    trainer.summary_writer.add_scalar(tag="loss_for_each_batch", scalar=running_loss, global_step=global_step)
                running_loss = epoch_loss


        except Exception as e:
            traceback.print_exc()
            pdb.set_trace()
    epoch_loss /= len(trainer.trainloader.dataset)
    epoch_acc = correct / total
    # print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
    return epoch_loss, epoch_acc


def test(model, loader, device, mode='test', plot_ROC=False):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    model.eval()
    predictions = None

    # if mode =='val':
    #     target_dataloader = trainer.valloader
    # elif mode =='train':
    #     target_dataloader = trainer.trainloader
    # else:
    target_dataloader = loader

    try:
        # if plot_ROC:
        gold = []
        pred = []
        with torch.no_grad():
            for images, labels in target_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss += criterion(outputs, labels).item()                
                total += labels.size(0)
                # if trainer.is_binary:
                #     correct += (torch.round(outputs.data) == labels).sum().item()
                # else:
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()


                if plot_ROC:
                    gold = np.append(gold, labels.cpu().numpy()) # type: ignore
                    # if trainer.is_binary:
                    #     pred = np.append(pred, torch.round(outputs).cpu().numpy())# type: ignore
                    # else:
                    pred = np.append(pred, torch.max(outputs, 1)[1])  # unverified # type: ignore
                else:
                    outputs = torch.nn.functional.softmax(outputs, dim=1)
                    # pdb.set_trace()
                    # if trainer.is_binary:
                    #     correct += (torch.round(outputs.data) == labels).sum().item()
                    #     confidence = torch.sigmoid(outputs).cpu().numpy()
                    #     prediction = (confidence >= 0.5).astype(np.int64)
                    # else:
                    ( confidence, prediction) = torch.max(outputs, 1)
                    confidence = confidence.cpu().numpy()
                    prediction = prediction.cpu().numpy()
                    truth =labels.cpu().numpy()
                    result = (prediction == truth).astype(np.int64)
                    pred = np.append(pred, confidence) #prediction comfidece
                    # pdb.set_trace()

                    gold = np.append(gold, result)

                    try:
                        assert len(pred) == len(gold)
                    except AssertionError:
                        traceback.print_exc()
                        pdb.set_trace()
                

        loss /= len(target_dataloader.dataset)
        accuracy = correct / total
        predictions = [pred, gold]  # type: ignore   
        if plot_ROC:            
            plot_ROC_curve(gold, pred)  # type: ignore          
            # pdb.set_trace()
            
    except Exception as e:
        traceback.print_exc()
        pdb.set_trace()
        
    return loss, accuracy, predictions # type: ignore


def train(trainer,
          epochs: int,  
          verbose=False, 
          wandb_logging=True, 
          patience= 2, 
          loss_min = 100000, 
          is_binary=False,
          savefilename=None,          
          round_no=1
          ):
    
    # (optional) calculate total steps
    steps = epochs * len(trainer.trainloader)

    trainer.epochs = epochs
    trainer.model = trainer.model.to(trainer.device)
    """Train the network on the training set."""
    

    if savefilename is None:
        savefilename = trainer.model.__class__.__name__

    

    record_mode = False

    initial_patience = patience

    if not verbose:
        pbar = tqdm(total=epochs, position=1, leave=False)
        pbar2 = tqdm(total=patience, position=2, leave=False)
        pbar2.update(patience)
        pbar.set_description(f"Epoch {1}")
        pbar2.set_description(f"patience")
    for epoch in range(epochs):
        if record_mode:
            if wandb_logging:
                # wandb.log({"acc": accuracy,"loss": loss}) # type: ignore
                pass
            if not verbose:
                pbar.update(1) # type: ignore
        elif patience<= 0:
            try:
                load_model(trainer.model, trainer.optimizer, savefilename)
            except Exception as e:
                print(traceback.print_exc())
                pdb.set_trace()
            loss, accuracy, _ = test(trainer)
            if wandb_logging:
                # wandb.log({"acc": accuracy,"loss": loss}) 
                pass
            if not verbose:
                pbar.update(1)  # type: ignore
                pbar2.set_description(f"Early stopped at epoch {epoch+1}, train_loss: {train_loss:.4f}, loss: {loss:.4f}, train_acc {train_acc:.4f}, acc: {accuracy:.4f}") # type: ignore
            # break
            record_mode = True
        else:
            train_loss, train_acc =  train_single_epoch(trainer, epoch, steps, round_no) 
            loss, accuracy, _ = test(trainer, mode='val')

            if loss_min > loss: # validation loss improved
                patience = initial_patience
                loss_min = loss
                save_model(trainer.model, trainer.optimizer, savefilename)
            else:
                patience -= 1

            if wandb_logging:
                # wandb.log({"train_acc": train_acc, "train_loss": train_loss,"acc": accuracy,"loss": loss}) 
                pass

            if verbose:
                print(f"\n\n\nEpoch {epoch+1}: train loss {train_loss:.3f}, val loss: {loss:.3f}, train acc {train_acc:.3f}, val acc: {accuracy:.3f}\n\n\n")
            else:
                pbar.update(1)# type: ignore
                pbar2.update(patience-pbar2.n) # type: ignore
                pbar.set_description(f"Epoch: {epoch+1}") # type: ignore
                pbar2.set_description(f"train_loss: {train_loss:.4f}, loss: {loss:.4f}, train_acc {train_acc:.4f}, acc: {accuracy:.4f}, Patience: ") # type: ignore
    if not verbose:
        pbar.close()# type: ignore
        pbar2.close()# type: ignore
    return trainer, loss, accuracy, record_mode # type: ignore
