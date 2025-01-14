from flask import Flask, jsonify, request
import torch
from torch.utils.data import IterableDataset, DataLoader
import itertools
import torch
import requests
import json
from src.utilities.datasets import DatasetWrapper
from threading import Lock

# Global threading lock
data_lock = Lock()

# Flask App
app = Flask(__name__)

# Global variables for the dataset and DataLoader
dataloader = None
data_iterator = None

@app.route('/initialize', methods=['POST'], endpoint='initialize_data_2') # Change endpoint name
def initialize_dataloader():
    print("Initializing dataloader...")
    global dataloader, data_iterator

    # Parse request data
    params = request.get_json()
    dataset_name = params.get("dataset_name", "CIFAR10")
    data_path = params.get("data_path", "~/dataset")
    batch_size = params.get("batch_size", 32)

    # Load dataset
    dataset_wrapper = DatasetWrapper(dataset_name, data_path)
    trainset = dataset_wrapper.trainset
    dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # Create an iterator for streaming
    data_iterator = iter(dataloader)

    return jsonify({"message": "Dataloader initialized", "batch_size": batch_size})

@app.route('/get_data', methods=['GET'], endpoint='get_data_batch_2')
def get_batch():
    global data_iterator

    if data_iterator is None:
        return jsonify({"error": "Dataloader not initialized"}), 400

    with data_lock:  # Synchronize access to the iterator
        try:
            # Fetch the next batch
            batch = next(data_iterator)
            inputs, labels = batch
            return jsonify({
                "inputs": inputs.tolist(),  # Convert tensors to lists for JSON serialization
                "labels": labels.tolist()
            })
        except StopIteration:
            # Reset the iterator if the end of the dataset is reached
            data_iterator = iter(dataloader)
            return jsonify({"error": "End of dataset reached"}), 400
        except Exception as e:
            # Catch other unexpected errors
            return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004)
