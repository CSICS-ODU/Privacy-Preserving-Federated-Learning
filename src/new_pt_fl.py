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
import copy
import os
import argparse
import torch
import pdb, traceback
import requests


from src.utilities.models import load_model_defination
from src.utilities.datasets import load_partitioned_datasets, get_dataloaders_subset
from src.utilities.training_utils import save_model, wandb_init,  print_info, get_device, train, test, load_model as load_saved_weights
from src.simple_network import SimpleNetwork

from torch import nn
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

import nvflare.client as flare
from nvflare.client.tracking import SummaryWriter
from nvflare.app_opt.pt.fedproxloss import PTFedProxLoss

from nvflare.apis.controller_spec import ClientTask, Task
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.streamers.container_retriever import ContainerRetriever



class DatasetStreamingController(Controller):
    def __init__(self, dict_retriever_id=None, batch_size=32, task_timeout=60, task_check_period=0.5):
        super().__init__(task_check_period=task_check_period)
        self.dict_retriever_id = dict_retriever_id
        self.dict_retriever = None
        self.task_timeout = task_timeout
        self.batch_size = batch_size
        self.data_iterator = None
        

    def start_controller(self, fl_ctx: FLContext):
        dataset_wrapper = DatasetWrapper("CIFAR10", "~/dataset")
        trainset = dataset_wrapper.trainset
        dataloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        self.data_iterator = iter(dataloader)
        self.dict_retriever.add_container("dataset_batches", self)

    def stop_controller(self, fl_ctx: FLContext):
        pass

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            engine = fl_ctx.get_engine()
            if self.dict_retriever_id:
                c = engine.get_component(self.dict_retriever_id)
                if not isinstance(c, ContainerRetriever):
                    self.system_panic(f"Invalid dict_retriever {self.dict_retriever_id}", fl_ctx)
                    return
                self.dict_retriever = c

    def get_next_batch(self):
        try:
            batch = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.dataloader)  # Reset iterator when exhausted
            batch = next(self.data_iterator)
        inputs, labels = batch
        return {"inputs": inputs.tolist(), "labels": labels.tolist()}

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        task = Task(name="fetch_batch", data=Shareable(), timeout=self.task_timeout)
        self.broadcast_and_wait(task=task, fl_ctx=fl_ctx, min_responses=1, abort_signal=abort_signal)


# --------------------------------------
# Client-Side Executor: Fetches Dataset Batches
# --------------------------------------
class DatasetStreamingExecutor(Executor):
    def __init__(self, dict_retriever_id=None):
        super().__init__()
        self.dict_retriever_id = dict_retriever_id
        self.dict_retriever = None
        self.is_initialized = False

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            engine = fl_ctx.get_engine()
            if self.dict_retriever_id:
                c = engine.get_component(self.dict_retriever_id)
                if not isinstance(c, ContainerRetriever):
                    self.system_panic(f"Invalid dict_retriever {self.dict_retriever_id}", fl_ctx)
                    return
                self.dict_retriever = c

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        
        self.main(fl_ctx)
        
        if task_name == "fetch_batch":
            if not self.dict_retriever:
                return make_reply(ReturnCode.SERVICE_UNAVAILABLE)
            rc, batch = self.dict_retriever.retrieve_container(from_site="server", fl_ctx=fl_ctx, timeout=10.0, name="dataset_batches")
            if rc != ReturnCode.OK:
                return make_reply(rc)
            return make_reply(ReturnCode.OK, data=batch)
        return make_reply(ReturnCode.TASK_UNKNOWN)
    
    def initialize_flare_components(self, fl_ctx: FLContext):
        flare.init()
        engine = fl_ctx.get_engine()
        engine.register_component("dataset_streaming_controller", DatasetStreamingController(dict_retriever_id="container_retriever"))
        engine.register_component("dataset_streaming_executor", DatasetStreamingExecutor(dict_retriever_id="container_retriever"))


    def fetch_batch_from_nvflare(self, fl_ctx: FLContext):
        if not self.is_initialized:
                self.is_initialized = True
                self.initiainitialize_flare_componentslize(fl_ctx)

        engine = fl_ctx.get_engine()

        try:
            task = Task(name="fetch_batch", data=Shareable(), timeout=10)
            response = engine.broadcast_and_wait(task, min_responses=1)
            if response and "data" in response:
                batch = response["data"]
                inputs, labels = torch.tensor(batch["inputs"]), torch.tensor(batch["labels"])
                return inputs, labels
            else:
                raise RuntimeError("Invalid batch response")
        except Exception as e:
            print(f"Error fetching batch: {e}")
            raise


def main(fl_ctx: FLContext):

    object1 = DatasetStreamingExecutor(dict_retriever_id="container_retriever")
    obect1.fetch_batch_from_api(fl_ctx)


    parser = argparse.ArgumentParser(description='A description of your program')
    parser.add_argument('-r', '--num_rounds', type=int, default=2, help='Number of rounds')
    parser.add_argument('-n', '--num_clients', type=int, default=2, help='Number of clients')
    parser.add_argument('-d', '--dataset_name', type=str, default='CIFAR10', help='Dataset name')
    parser.add_argument('-m', '--model_name', type=str, default='efficientnet', help='Model name')
    parser.add_argument('-a', '--aggregation_function', type=str, default='fedavg', help='Aggregation function')
    args = parser.parse_args()

    n_clients = args.num_clients
    num_rounds = args.num_rounds
    dataset_name = args.dataset_name
    model_name = args.model_name
    aggregation_function = args.aggregation_function
    data_path = "~/dataset"

    [trainloader, valloaders, testloader, _ ], num_channels, num_classes = load_partitioned_datasets(num_clients=n_clients, dataset_name=dataset_name, data_path=data_path, batch_size=32,split=None) 
    
    print("******** MODEL NAME: ", model_name)
    print("******** DATASET NAME: ", dataset_name)
    print("******** num channels: ", num_channels)
    print("******** num classes: ", num_classes)

    # initialize_flare_components()
    model = load_model_defination(model_name=model_name, num_channels=num_channels, num_classes=num_classes)


    model_global = copy.deepcopy(model)

    batch_size = 4
    epochs = 50
    lr = 0.001

    # initialize_dataloader_api(dataset_name, data_path, batch_size)

    # model = SimpleNetwork()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_global.to(device)
    
    loss = nn.CrossEntropyLoss() # Same as criterion
    criterion_prox = None

    fedproxloss_mu = 0.0
    if aggregation_function == 'fedprox':
        fedproxloss_mu = 0.00001

    if fedproxloss_mu > 0:
        criterion_prox = PTFedProxLoss(mu=fedproxloss_mu)

    optimizer = Adam(model.parameters(), lr=lr)
    transforms = Compose(
        [
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    flare.init()
    sys_info = flare.system_info()
    client_name = sys_info["site_name"]

    # train_dataset = CIFAR10(
    #     root=os.path.join(DATASET_PATH, client_name), transform=transforms, download=True, train=True
    # )
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    summary_writer = SummaryWriter()
    while flare.is_running():
        input_model = flare.receive()
        print(f"current_round={input_model.current_round}")
        print(f"input model={input_model.meta}")


        model.load_state_dict(input_model.params)
        model.to(device)

        steps = epochs * len(trainloader[0])
        for epoch in range(epochs):
            running_loss = 0.0
            print("**********************************")
            # print("NVIDIA TL", train_loader)
            print("MY TL", trainloader[0])
            print("**********************************")
            for i in range(len(trainloader[0])):
                inputs, labels = self.fetch_batch_from_nvflare(fl_ctx)
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                predictions = model(inputs)
                cost = loss(predictions, labels)

                if fedproxloss_mu > 0:
                    fed_prox_loss = criterion_prox(model, model_global)
                    cost += fed_prox_loss
                    # print("FedProx loss calculated!!")

                cost.backward()
                optimizer.step()

                running_loss += cost.cpu().detach().numpy() / inputs.size()[0]
                if i % 3000 == 0:
                    print(f"Epoch: {epoch}/{epochs}, Iteration: {i}, Loss: {running_loss / 3000}")
                    global_step = input_model.current_round * steps + epoch * len(trainloader[0]) + i
                    summary_writer.add_scalar(tag="loss_for_each_batch", scalar=running_loss, global_step=global_step)
                    running_loss = 0.0

        print("Finished Training")

        PATH = f"/home/akapo004/new_nvflare/saved_models/{dataset_name}_{n_clients}_{aggregation_function}_net.pth.tar"
        torch.save(model.state_dict(), PATH)

        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )

        flare.send(output_model)

        
        evaluate(evaluation_model=f"{dataset_name}_{n_clients}_{aggregation_function}_net", device=device, dataset_name=dataset_name, model_name=model_name, n_clients = n_clients , data_path= data_path)

def evaluate(evaluation_model, device, dataset_name, model_name, n_clients, data_path):
    
    print_info(device, model_name, dataset_name)    
    try:
        
        print("************ NOW RUNNING EVALUATE **************")
        [train_loaders, val_loaders, test_loader, _], num_channels, num_classes = load_partitioned_datasets(num_clients=n_clients, dataset_name=dataset_name, data_path=data_path, batch_size=32,split=None) 
        
        val_loader = val_loaders[0]   
        train_loader = train_loaders[0]

        test_loader_size = len(test_loader.dataset)


        train_loader = get_dataloaders_subset(train_loader, test_loader_size)
        
        model = load_model_defination(model_name=model_name, num_channels=num_channels, num_classes=num_classes).to(device) 
        # model = load_model_defination(model_name, num_channels, num_classes)
        optimizer = torch.optim.Adam(model.parameters())

        print("************ model and dataset loaded perfectly in evaluate function **************")
        # model, optimizer, train_loader = make_private(differential_privacy, model, optimizer, train_loader)


        load_saved_weights(model, filename = evaluation_model)

        print("************ weights loaded perfectly **************")

        comment = 'Test_Centralized_('+evaluation_model+')_'+model_name+'_'+dataset_name
        # if wandb_logging:
        #     wandb_init(comment=comment, model_name=model_name, dataset_name=dataset_name)
        #     wandb.watch(model, log_freq=100)
            
        trn_loss, trn_accuracy, predA = test(model=model, loader=train_loader, device=device)
        val_loss, val_accuracy, _ = test(model=model, loader=val_loader, device=device)
        tst_loss, tst_accuracy, predB = test(model=model, loader=test_loader, device=device)


        
        # pdb.set_trace()




        print(f"Final training set performance:\n\tloss {trn_loss}\n\taccuracy {trn_accuracy}")








        # if wandb_logging:
        #     wandb.log({"train_acc": trn_accuracy, "train_loss": trn_loss})
        #     wandb.log({"acc": val_accuracy, "loss": val_loss}, step = 100)
        #     wandb.log({"test_acc": tst_accuracy, "test_loss": tst_loss})
        #     wandb.finish()
        print(f"Final validation set performance:\n\tloss {val_loss}\n\taccuracy {val_accuracy}")
        print(f"Final test set performance:\n\tloss {tst_loss}\n\taccuracy {tst_accuracy}")

        # plot_histogram(predA, predB)
            
        # if wandb_logging:
        #     wandb.finish()
    except Exception as e:
        traceback.print_exc()
        # pdb.set_trace()


if __name__ == "__main__":
    # initialize_dataloader_api("CIFAR10", "~/dataset", 32)
    main(fl_ctx: FLContext)