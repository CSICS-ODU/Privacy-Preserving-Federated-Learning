
# Default values
model_name="efficientnet"
dataset_name="CIFAR100_0"
num_rounds=10
num_clients=2
threads=2
epochs=5
root="saved_models/"
weight=""

# Parse flagged arguments
while getopts m:d:n:r:t:w: flag
do
    case "${flag}" in
        m) model_name=${OPTARG};;
        d) dataset_name=${OPTARG};;
        n) num_clients=${OPTARG};;
        r) num_rounds=${OPTARG};;
        t) threads=${OPTARG};;
        w) weight=${OPTARG};;
    esac
done

# Define modelweight variable
modelweight="$(pwd)/$root$weight"


crun -p ~/envs/NVFlarev2.4.0rc8 nvflare config -jt ./templates
# Base template directory
template_dir="./templates/sag_custom"

experiment_name="${num_clients}_${model_name}_${dataset_name}"

if [ ! -d "./workspace" ]; then
    # Create the directory if it does not exist
    mkdir ./workspace
    echo "Directory 'workspace' created."
fi

if [ -d "./workspace/$experiment_name" ]; then
    # Prompt for confirmation with a 30-second timeout
    read -t 30 -p "Directory ./workspace/$experiment_name exists. Press 't' to terminate the process or any other key to continue with deletion: " response

    # Check the response
    if [ "$response" = "t" ] || [ "$response" = "T" ]; then
        echo "Process terminated by user."
        exit 1
    else
        echo "Deleting the existing workspace/$experiment_name directory."
        rm -r "./workspace/$experiment_name"
    fi
fi


if [ ! -d "./jobs" ]; then
    # Create the directory if it does not exist
    mkdir ./jobs
    echo "Directory 'jobs' created."
fi

if [ ! -d "./jobs/$experiment_name" ]; then
    # Create the directory if it does not exist
    mkdir ./jobs/$experiment_name
    echo "Directory 'jobs/$experiment_name' created."
fi




# Delete all subdirectories in sag_custom
find "$template_dir" -mindepth 1 -type d -exec rm -rf {} +

reference_code_dir="./templates/reference_code"

# Create client app directories dynamically
for ((i=0; i<$num_clients; i++))
do
    client_dir="$template_dir/app_$i"
    mkdir -p "$client_dir"
    cp "$reference_code_dir/config_fed_client.conf" "$client_dir/config_fed_client.conf"
done

# Create meta.conf content
meta_conf_content="{
  name = \"sag_custom\"
  resource_spec = {}
  deploy_map {
    app_server = [\"server\"]"
for ((i=0; i<$num_clients; i++))
do
    meta_conf_content+="\n    app_$i = [\"site-$(($i + 1))\"]"
done
meta_conf_content+="\n  }
  min_clients = $num_clients
  mandatory_clients = []
}"

# Write meta.conf content to file
echo -e "$meta_conf_content" > "$template_dir/meta.conf"


# Ensure the server app directory exists
server_dir="$template_dir/app_server"
mkdir -p "$server_dir"
cp "$reference_code_dir/config_fed_server.conf" "$server_dir/config_fed_server.conf"


# Base job creation command
command="crun -p ~/envs/NVFlarev2.4.0rc8 nvflare job create -force -j ./jobs/$experiment_name -w $template_dir -sd ./code/"   

# Loop through each client and add their specific configurations
for ((i=0; i<$num_clients; i++))
do
    app_name="app_$i"
    command+=" -f $app_name/config_fed_client.conf executors[0].executor.args.model_name=\"$model_name\"   executors[0].executor.args.dataset_name=\"${dataset_name}_$i\"     executors[0].executor.args.num_clients=$num_clients    executors[1].executor.args.model_name=\"$model_name\"     executors[1].executor.args.dataset_name=\"${dataset_name}_$i\"    executors[1].executor.args.num_clients=$num_clients"
done

# Add server configurations
command+=" -f app_server/config_fed_server.conf components[3].args.model_name=\"$model_name\" workflows[1].args.num_rounds=$num_rounds"

# Conditionally add the source checkpoint file argument
if [[ -n "$weight" ]]; then
    command+=" components[0].args.source_ckpt_file_full_name=\"$modelweight\""
fi
# Execute the command
eval $command

echo "job created successfully!\n\n"

Run the NVFlare simulator
crun -p ~/envs/NVFlarev2.4.0rc8 nvflare simulator -n $num_clients -t $threads ./jobs/$experiment_name -w workspace/$experiment_name
