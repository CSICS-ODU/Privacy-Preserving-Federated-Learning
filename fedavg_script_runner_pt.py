# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import argparse
from code.utilities.models import load_model_defination
from code.utilities.datasets import load_partitioned_datasets
from code.simple_network import SimpleNetwork

from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob
from nvflare.job_config.script_runner import ScriptRunner

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='A description of your program')
    parser.add_argument('-r', '--num_rounds', type=int, default=2, help='Number of rounds')
    parser.add_argument('-n', '--num_clients', type=int, default=2, help='Number of clients')
    parser.add_argument('-d', '--dataset_name', type=str, default='CIFAR10', help='Dataset name')
    parser.add_argument('-m', '--model_name', type=str, default='efficientnet', help='Model name')
    args = parser.parse_args()

    n_clients = args.num_clients
    num_rounds = args.num_rounds
    dataset_name = args.dataset_name
    model_name = args.model_name
    data_path = "~/dataset"

    [trainloader, valloaders, testloader, _ ], num_channels, num_classes = load_partitioned_datasets(num_clients=n_clients, dataset_name=dataset_name, 
                                                                                                         data_path=data_path, batch_size=32,split=None) 

    model = load_model_defination(model_name=model_name, num_channels=num_channels, num_classes=num_classes)

    train_script = "src/pt_fl.py"

    job = FedAvgJob(
        name=f"pt_{dataset_name}_{model_name}_fedavg",
         n_clients=n_clients, 
         num_rounds=num_rounds, 
         initial_model=model
    )

    # Add clients
    for i in range(n_clients):
        executor = ScriptRunner(
            script=train_script, script_args=f"--num_rounds {num_rounds} --num_clients {n_clients} --dataset_name {dataset_name} --model_name {model_name}"  # f"--batch_size 32 --data_path /tmp/data/site-{i}"
        )
        job.to(executor, f"site-{i + 1}")

    # job.export_job("/tmp/nvflare/jobs/job_config")
    job.simulator_run("/tmp/nvflare/jobs/workdir", gpu="0")
