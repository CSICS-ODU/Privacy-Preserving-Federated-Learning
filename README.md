# Privacy Preserving Federated Learning
- [ ]  Train and save a ML model first
 - Run '**[create_and_run_simulation.sh](https://github.com/soumyaxyz/Privacy-Preserving-Federated-Learning/blob/main/create_and_run_simulation.sh "create_and_run_simulation.sh")**' with appropriate arguments to train and save a ML model  on a dataset with **[Nvidia FLARE](https://nvflare.readthedocs.io/en/main/index.html)** 

	    bash create_and_run_simulation.sh 	-m <model_name> 
							-d <dataset_name> 
							-n <num_clients> 
							-r <num_FL_rounds>   
							-w <saved_model_weights>
    example:
    
	    bash create_and_run_simulation.sh -m efficientnet -d CIFAR10  -n 2 -r 25
     
    OR

 - Run all commands in '**[interactive_create.sh](https://github.com/soumyaxyz/Privacy-Preserving-Federated-Learning/blob/main/interactive_create.sh "interactive_create.sh")**' in sequence to create a job  with **2** clients to train **efficientnet** on **CIFAR10** for **25** rounds and and run the simulation.



 
- [ ]  Run '**[membership_inference_attack.py](https://github.com/soumyaxyz/Privacy-Preserving-Federated-Learning/blob/main/code/membership_inference_attack.py "membership_inference_attack.py")**' with appropriate arguments to execute membership inference attack on a saved ML model.



Note that run **[modify.py](https://github.com/soumyaxyz/Privacy-Preserving-Federated-Learning/blob/main/modify.py "modify.py")** can be utilized to convert the scripts for a [SLURM](https://slurm.schedmd.com/sbatch.html) managed system.

### Change log
-  Added unified interface for running MIA

 - Added unified interface for running simulated Federated job

		#To run a standalone job  
		sbatch run_fedreated_simulation.sh -r 2 -n 2 -d CIFAR100 -m efficeintnet

		#To run an incremental job (-s number of splits)  
		sbatch run_fedreated_simulation.sh -r 7 -n 2 -t 2 -s 4 -d incrementalCIFAR100=ABCD -m efficientnet
