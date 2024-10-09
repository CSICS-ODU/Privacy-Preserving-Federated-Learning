# Privacy Preserving Federated Learning
- [ ]  Train and save a ML model first
 - Run '**[script_run.sh](https://github.com/CSICS-ODU/Privacy-Preserving-Federated-Learning/blob/aayush/script_run.sh "script_run.sh")**' with appropriate arguments to train and save a ML model  on a dataset with **[Nvidia FLARE](https://nvflare.readthedocs.io/en/main/index.html)** 

	    bash script_run.sh 	-m <model_name> 
							-d <dataset_name> 
							-n <num_clients> 
							-r <num_FL_rounds>   
			
    example:
    
	    bash script_run.sh -m efficientnet -d CIFAR10  -n 2 -r 25
