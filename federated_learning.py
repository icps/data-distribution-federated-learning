import os
import time
from multiprocessing import Process

import client
import server
import dataset_partitions as dataset

import tensorflow as tf

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import random
random.seed(10)

from variables import dataset_name, partition_type, num_rounds, num_clients, fraction_fit



def run_simulation(num_rounds: int, num_clients: int, fraction_fit: float):
    """Start a FL simulation."""

    # This will hold all the processes which we are going to create
    processes = []

    # Start the server
    server_process = Process(target = server.start_server, args = (num_rounds, num_clients, fraction_fit))
    server_process.start()
    processes.append(server_process)

    # Optionally block the script here for a second or two so the server has time to start
    time.sleep(2)

    # Load the dataset partitions
    partitions = dataset.load(num_clients, dataset_name, partition_type)
    
    # Start all the clients
    for partition in partitions:
        client_process = Process(target = client.start_client, args = (partition,))
        client_process.start()
        processes.append(client_process)

    # Block until all processes are finished
    for p in processes:
        p.join()


if __name__ ==  '__main__':
    run_simulation(num_rounds = num_rounds, num_clients = num_clients, fraction_fit = fraction_fit)
