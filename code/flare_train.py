# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os.path
import pdb, traceback

import torch
from pt_constants import PTConstants
#from simple_network import EfficientNet #SimpleNetwork
from utilities.models import load_model_defination

from torch import nn
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

from utilities.training_utils import Trainer, save_model, wandb_init,  print_info, get_device, train, test, load_model as load_saved_weights
from utilities.models import load_model_defination
from utilities.datasets import load_partitioned_datasets

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.model import make_model_learnable, model_learnable_to_dxo
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_opt.pt.model_persistence_format_manager import PTModelPersistenceFormatManager


class Fl_Trainer(Executor):
    def __init__(
        self,
        data_path="~/dataset",
        model_name="efficientnet",
        dataset_name="CIFAR10",
        lr=0.01,
        epochs=5,
        num_clients=2,
        train_task_name=AppConstants.TASK_TRAIN,
        submit_model_task_name=AppConstants.TASK_SUBMIT_MODEL,
        exclude_vars=None,
        pre_train_task_name=AppConstants.TASK_GET_WEIGHTS,
    ):
        """Cifar10 Trainer handles train and submit_model tasks. During train_task, it trains a
        simple network on CIFAR10 dataset. For submit_model task, it sends the locally trained model
        (if present) to the server.

        Args:
            lr (float, optional): Learning rate. Defaults to 0.01
            epochs (int, optional): Epochs. Defaults to 5
            train_task_name (str, optional): Task name for train task. Defaults to "train".
            submit_model_task_name (str, optional): Task name for submit model. Defaults to "submit_model".
            exclude_vars (list): List of variables to exclude during model loading.
            pre_train_task_name: Task name for pre train task, i.e., sending initial model weights.
        """
        super().__init__()

        self._lr = lr
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.model_name = model_name
        self._epochs = epochs
        self._train_task_name = train_task_name
        self._pre_train_task_name = pre_train_task_name
        self._submit_model_task_name = submit_model_task_name
        self._exclude_vars = exclude_vars
        self.is_initialized = False


    def initialize(self, fl_ctx: FLContext):
        self.app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        fl_args = fl_ctx.get_prop(FLContextKey.ARGS)
        self.client_id = fl_ctx.get_identity_name()

        print("***********", fl_args.n_clients)
        print("***********", self.client_id)

        dataset_parameters_list = self.dataset_name.split("_")

        # if len(dataset_parameters_list) == 2:
        #     dataset_name, data_index = dataset_parameters_list
        #     data_index = int(data_index)
        #     split=None
        # else:
        #     dataset_name, split, data_index = dataset_parameters_list
        #     data_index= int(data_index)
        #     split=int(split)
        
        # incrementalCIFAR10=ACBD_0_0
    
        dataset_name, split, _ = dataset_parameters_list  # Only the base dataset name
        split = int(split)  # Split will be picked up dynamically
        data_index = int(self.client_id.split("-")[1]) - 1  # Client ID from FLContext
        
        # else:
        #     dataset_name = dataset_parameters_list[0]
        #     split = int(fl_args.n_clients)  # Number of clients from FLContext
        #     data_index = int(self.client_id.split("-")[1]) - 1  # Client ID from FLContext
        print("\n\n")
        print("The current split is: ", split)
        print("\n\n")
        print(f'current working dir: {os.getcwd()}')  
        [trainloader, valloaders, testloader, _ ], num_channels, num_classes = load_partitioned_datasets(num_clients=fl_args.n_clients, dataset_name=dataset_name, 
                                                                                                         data_path=self.data_path, batch_size=32,split=split) 

        
        train_loader = trainloader[data_index]
        valloader = valloaders[data_index]
        self._n_iterations = len(train_loader)
        

        # Training setup
        model = load_model_defination(self.model_name, num_channels, num_classes) 
        # optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
        optimizer = Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()   
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)


        self.model_information = Trainer(model, 
                                    train_loader, 
                                    valloader, 
                                    testloader, 
                                    optimizer, 
                                    criterion,
                                    device = device, 
                                    is_binary=False, 
                                    summary_writer=None)



        # Setup the persistence manager to save PT model.
        # The default training configuration is used by persistence manager
        # in case no initial model is found.
        self._default_train_conf = {"train": {"model": type(self.model_information.model).__name__}}
        self.persistence_manager = PTModelPersistenceFormatManager(
            data=self.model_information.model.state_dict(), default_train_conf=self._default_train_conf
        )

        # OUTPUT fl_args.__dict__.keys()
        # dict_keys(['job_folder', 'workspace', 'clients', 'n_clients', 'threads', 'gpu', 'max_clients', 'set', 'log_config', 'config_folder', 'job_id', 'client_config', 'env', 'sp_target', 'sp_scheme', 'server_config', 'client_name', 'token'])


    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        try:
            if not self.is_initialized:
                self.is_initialized = True
                self.initialize(fl_ctx)

            if task_name == self._pre_train_task_name:
                # Get the new state dict and send as weights
                return self._get_model_weights()
            elif task_name == self._train_task_name:
                # Get model weights
                try:
                    dxo = from_shareable(shareable)
                except:
                    self.log_error(fl_ctx, "Unable to extract dxo from shareable.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Ensure data kind is weights.
                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_error(fl_ctx, f"data_kind expected WEIGHTS but got {dxo.data_kind} instead.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Convert weights to tensor. Run training
                torch_weights = {k: torch.as_tensor(v) for k, v in dxo.data.items()}
                


                self._local_train(fl_ctx, torch_weights, abort_signal)

                # Check the abort_signal after training.
                # local_train returns early if abort_signal is triggered.
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                # Save the local model after training.
                self._save_local_model(fl_ctx)

                # Get the new state dict and send as weights
                return self._get_model_weights()
            elif task_name == self._submit_model_task_name:
                # Load local model
                ml = self._load_local_model(fl_ctx)

                # Get the model parameters and create dxo from it
                dxo = model_learnable_to_dxo(ml)
                return dxo.to_shareable()
            else:
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except Exception as e:
            self.log_exception(fl_ctx, f"Exception in simple trainer: {e}.")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def _get_model_weights(self) -> Shareable:
        # Get the new state dict and send as weights
        weights = {k: v.cpu().numpy() for k, v in self.model_information.model.state_dict().items()}

        outgoing_dxo = DXO(
            data_kind=DataKind.WEIGHTS, data=weights, meta={MetaKey.NUM_STEPS_CURRENT_ROUND: self._n_iterations}
        )
        return outgoing_dxo.to_shareable()

    def _local_train(self, fl_ctx, weights, abort_signal):
        # Set the model weights
        self.model_information.model.load_state_dict(weights)
        if abort_signal.triggered:
            # If abort_signal is triggered, we simply return.
            # The outside function will check it again and decide steps to take.
            return #todo
        model_information, val_loss, val_accuracy, _  = train(self.model_information, self._epochs, verbose=True, wandb_logging=False, round_no=None)
        # torch.save(model.state_dict(), PATH)


    def _save_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        print(f"\n\nrun_dir: {run_dir}")
        print(f'\n\nPTModelsDir: {PTConstants.PTModelsDir}')
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        print (f"\n\nchecking if {models_dir} exists")
        try:
            if not os.path.exists(models_dir):
                print (f"\n\ncreating {models_dir} exists")
                os.makedirs(models_dir, exist_ok=True)
        except FileExistsError:
            pass
        model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

        ml = make_model_learnable(self.model_information.model.state_dict(), {})
        self.persistence_manager.update(ml)
        torch.save(self.persistence_manager.to_persistence_dict(), model_path)

    def _load_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            return None
        model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

        self.persistence_manager = PTModelPersistenceFormatManager(
            data=torch.load(model_path), default_train_conf=self._default_train_conf
        )
        ml = self.persistence_manager.to_model_learnable(exclude_vars=self._exclude_vars)
        return ml
