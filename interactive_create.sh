#!/bin/bash

#SBATCH -c 8
#SBATCH -p gpu --gres=gpu:1
#SBATCH -o testing-%j.out

module load container_env pytorch-gpu


crun -p ~/envs/ppfl nvflare config -jt ./templates

find ./templates/sag_custom -mindepth 1 -type d -exec rm -rf {} +

# Create meta.conf content
meta_conf_content="{
  name = \"sag_custom\"
  resource_spec = {}
  deploy_map {
    app = [ \"@ALL\" ]
    }
  min_clients = 2
  mandatory_clients = []
}"

echo -e "$meta_conf_content" > "./templates/sag_custom/meta.conf"

# mkdir -p ./templates/sag_custom/app_0
# cp ./templates/reference_code/config_fed_client.conf  ./templates/sag_custom/app_0/config_fed_client.conf

# mkdir -p "./templates/sag_custom/app_1"
# cp ./templates/reference_code/config_fed_client.conf ./templates/sag_custom/app_1/config_fed_client.conf

# mkdir -p ./templates/sag_custom/app_server
# cp ./templates/reference_code/config_fed_server.conf ./templates/sag_custom/app_server/config_fed_server.conf

# mkdir -p ./templates/sag_custom/config
cp ./templates/reference_code/config_fed_client.conf  ./templates/sag_custom/config_fed_client.conf
cp ./templates/reference_code/config_fed_server.conf ./templates/sag_custom/config_fed_server.conf


# Base job creation command

crun -p ~/envs/ppfl nvflare job create -force -j ./jobs -w sag_custom -sd ./code/ \-f config_fed_client.conf \
executors[0].executor.args.model_name="efficientnet" \
executors[0].executor.args.dataset_name="incrementalCIFAR10=ACBD_0_0" \
executors[0].executor.args.num_clients=2 \
executors[1].executor.args.model_name="efficientnet" \
executors[1].executor.args.dataset_name="incrementalCIFAR10=ACBD_0_0" \
executors[1].executor.args.num_clients=2 \
-f config_fed_client.conf \
executors[0].executor.args.model_name="efficientnet" \
executors[0].executor.args.dataset_name="incrementalCIFAR10=ACBD_0_1" \
executors[0].executor.args.num_clients=2 \
executors[1].executor.args.model_name="efficientnet" \
executors[1].executor.args.dataset_name="incrementalCIFAR10=ACBD_0_1" \
executors[1].executor.args.num_clients=2 \
-f config_fed_server.conf  \
components[3].args.model_name="efficientnet" \
workflows[1].args.num_rounds=25


# Run the NVFlare simulator
crun -p ~/envs/ppfl nvflare simulator -n 2 -t 2 ./jobs -w ./workspace
