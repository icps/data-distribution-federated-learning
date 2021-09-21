# How does data distribution influences results in Federated Learning?

I was trying to replicate the experiment shown [here](https://mike.place/talks/fl/) about data distribution in Federated Learning. 

Using MNIST dataset and 10 clients, we have four scenarios:
1. **non-federated:** all clients have the whole dataset;
2. **uniform:** all clients have the same data distribution (10% of each class);
3. **90% federated:** each client contains 90% of one single class and 10% of others;
4. **100% federated:** each client contains only one class.

I did the same experiment do CIFAR-10, which contains more complex data than MNIST.

## About the code

### Software Requirements

We tested this code on Linux Ubuntu 20.04.
We use Python (version 3.7.3) with the Anaconda distribution (version 4.7.11). 
Prior to running the experiments, make sure to install the following required libraries:

- [Numpy](https://numpy.org/) (version 1.16.4)
- [Tensorflow](https://www.tensorflow.org/) (version 2.3.1)
- [Flower](https://flower.dev/) (version 0.17.0)


### Project Structure and tips

* To run, change the values in ```variables.py``` and run ```federated-learning.py```
* The results are in ```plots.ipynb```. It plots from the folder ```{DATASET}_tests/```. We automatically save the results in this folder.
* The neural networks are implementd in ```client.py```
* To reproduce this experiment:
  * MNIST
    * IID, uniform, and 90% federated: epoch = 1; batch_size = 32; step_per_epoch = 0 (uses the tensorflow default)
    * 100% federated: epoch = 1; batch_size = 32; steps_per_epoch = 10
  * CIFAR-10
    * IID: epoch = 1; batch_size = 32; step_per_epoch = 0
    * uniform: epoch = 5; batch_size = 32; step_per_epoch = 0
    * 90% percent: epoch = 5; batch_size = 64; step_per_epoch = "divide" (len(train) // batch_size)
    * 100% percent: epoch = 7; batch_size = 128; step_per_epoch = "divide"




