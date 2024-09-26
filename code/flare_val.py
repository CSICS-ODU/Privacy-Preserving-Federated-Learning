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

import torch
#from simple_network import EfficientNet
from utilities.training_utils import Trainer, save_model, wandb_init,  print_info, get_device, train, test, load_model as load_saved_weights
from utilities.models import load_model_defination
from utilities.datasets import load_partitioned_datasets

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants


class Fl_Validator(Executor):
    def __init__(self, 
                model_name="efficientnet", 
                dataset_name="CIFAR10_0", 
                num_clients=2, 
                data_path="~/datasets", 
                validate_task_name=AppConstants.TASK_VALIDATION):

        super().__init__()

        self.dataset_name = dataset_name
        self.data_path = data_path
        self.model_name = model_name
        self._validate_task_name = validate_task_name
        self.is_initialized = False

    def initialize(self, fl_ctx: FLContext):

        self.app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        fl_args = fl_ctx.get_prop(FLContextKey.ARGS)
        self.client_id = fl_ctx.get_identity_name()

        dataset_parameters_list = self.dataset_name.split("_")

        # if len(dataset_parameters_list) == 2:
        #     dataset_name, data_index = dataset_parameters_list
        #     data_index = int(data_index)
        #     split=None
        # else:
        #     dataset_name, split, data_index = dataset_parameters_list
        #     data_index= int(data_index)
        #     split=int(split)

        dataset_name, split, _ = dataset_parameters_list
        split = int(split)  # Split will be picked up dynamically
        data_index = int(self.client_id.split("-")[1]) - 1  # Client ID from FLContext

        # Preparing the dataset for testing.
        [trainloader, valloaders, testloader, _ ], num_channels, num_classes = load_partitioned_datasets(num_clients=fl_args.n_clients, dataset_name=dataset_name, 
                                                                                                         data_path=self.data_path, batch_size=32,split=split) 
        

        train_loader = trainloader[data_index] # unused
        valloader = valloaders[data_index]    # unused
        
        # Setup the model
        model = load_model_defination(self.model_name, num_channels, num_classes) # SimpleNetwork()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        
        self.model_information = Trainer(model, 
                                    train_loader, 
                                    valloader, 
                                    testloader,   #check this
                                    optimizer = None, 
                                    criterion = None,
                                    device = device, 
                                    is_binary=False, 
                                    summary_writer=None)

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if not self.is_initialized:
            self.is_initialized = True
            self.initialize(fl_ctx)
        
        if task_name == self._validate_task_name:
            model_owner = "?"
            try:
                try:
                    dxo = from_shareable(shareable)
                except:
                    self.log_error(fl_ctx, "Error in extracting dxo from shareable.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Ensure data_kind is weights.
                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_exception(fl_ctx, f"DXO is of type {dxo.data_kind} but expected type WEIGHTS.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Extract weights and ensure they are tensor.
                model_owner = shareable.get_header(AppConstants.MODEL_OWNER, "?")
                weights = {k: torch.as_tensor(v, device=self.model_information.device) for k, v in dxo.data.items()}

                # Get validation accuracy
                val_accuracy = self._validate(weights, abort_signal)
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                self.log_info(
                    fl_ctx,
                    f"Accuracy when validating {model_owner}'s model on"
                    f" {fl_ctx.get_identity_name()}"
                    f"s data: {val_accuracy}",
                )

                dxo = DXO(data_kind=DataKind.METRICS, data={"val_acc": val_accuracy})
                return dxo.to_shareable()
            except:
                self.log_exception(fl_ctx, f"Exception in validating model from {model_owner}")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)

    def _validate(self, weights, abort_signal):
        self.model_information.model.load_state_dict(weights)

        loss, accuracy, _ = test(self.model_information)

        

        return accuracy
