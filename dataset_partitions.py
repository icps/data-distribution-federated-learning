from typing import List, Tuple, cast

import numpy as np
import tensorflow as tf

import random
random.seed(10)

XY                 = Tuple[np.ndarray, np.ndarray]
XYList             = List[XY]
PartitionedDataset = List[Tuple[XY, XY]]


def create_partitions_iid(source_dataset: XY, num_partitions: int) -> XYList:
    """ 
        Create partitioned version of a source dataset.
    
        Each client receives the entire dataset (non-federated).
    """
    
    x, y = source_dataset
    
    x = np.broadcast_to(x, (num_partitions,) + x.shape).copy()
    y = np.broadcast_to(y, (num_partitions,) + y.shape).copy()

    return list(zip(x, y))


def create_partitions_uniform(source_dataset: XY, num_partitions: int) -> XYList:
    """
        Create partitioned version of a source dataset.
    
        Each client receives the same amount of each class (uniform distribution).
    """
    
    x, y = source_dataset
    
    classes     = range(0, num_partitions)
    x_partition = [[] for i in classes]
    y_partition = [[] for i in classes]

    for num in classes:

        ## each class
        class_idx = np.where(y == num)[0]

        ## shuffle to save the items of each class at random
        np.random.shuffle(class_idx)

        ## divide to each partition
        vals = list(range(0, len(class_idx) + 1, (len(class_idx) // num_partitions)))

        for enum, (start, end) in enumerate(zip(vals, vals[1:])):
            part_index = class_idx[start:end]
            x_partition[enum].extend(x[part_index])
            
            y_categorical = tf.keras.utils.to_categorical(y[part_index], 10)
            y_partition[enum].extend(y_categorical)
                  
    return list(zip(x_partition, y_partition))


def create_partitions_90percent(source_dataset: XY, num_partitions: int) -> XYList:
    """
        Create partitioned version of a source dataset.
    
        Each client receives 90% of a class. The 10% of this class is divided evenly to the other clients.
    """
    
    x, y = source_dataset
    
    classes     = range(0, num_partitions)
    x_partition = [[] for i in classes]
    y_partition = [[] for i in classes]

    for num in classes:

        ## each class
        class_idx = np.where(y == num)[0]

        ## shuffle to save the items of each class at random
        np.random.shuffle(class_idx)

        ## divide to each partition
        part90 = int(len(class_idx) * 0.9)
        vals   = list(range(0, len(class_idx), (len(class_idx) - part90) // num_partitions - 1))
        vals   = vals[:num + 1] + vals[-num_partitions + num:]

        for enum, (start, end) in enumerate(zip(vals, vals[1:])):
            part_index = class_idx[start:end]
            x_partition[enum].extend(x[part_index])
            
            y_categorical = tf.keras.utils.to_categorical(y[part_index], 10)
            y_partition[enum].extend(y_categorical)

    return list(zip(x_partition, y_partition))


def create_partitions_100percent(source_dataset: XY, num_partitions: int) -> XYList:
    """
        Create partitioned version of a source dataset.
    
        Each client receives 100% of a class.
    """
    
    x, y = source_dataset
    
    classes     = range(0, num_partitions)
    x_partition = [[] for i in classes]
    y_partition = [[] for i in classes]

    for num in classes:

        ## each class
        class_idx = np.where(y == num)[0]

        ## shuffle to save the items of each class at random
        np.random.shuffle(class_idx)

        ## divide to each partition
        part_index = class_idx[0:len(class_idx)]
        x_partition[num].extend(x[part_index])

        y_categorical = tf.keras.utils.to_categorical(y[part_index], 10)
        y_partition[num].extend(y_categorical)

    return list(zip(x_partition, y_partition))


def load(num_partitions: int, dataset_name: str, partition: str) -> PartitionedDataset:
    """Create partitioned version of datasets."""
    
    if dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
    elif dataset_name == "cifar10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train  = x_train.astype("float32") / 255
    x_test   = x_test.astype("float32") / 255

    if dataset_name == "mnist":
        x_train  = np.expand_dims(x_train, -1)
        x_test   = np.expand_dims(x_test, -1)
    
    xy_train = (x_train, y_train)
    xy_test  = (x_test, y_test)

    if partition == "iid":
        xy_train_partitions = create_partitions_iid(xy_train, num_partitions)
        xy_test_partitions  = create_partitions_iid(xy_test, num_partitions)
        
    elif partition == "uniform":
        xy_train_partitions = create_partitions_uniform(xy_train, num_partitions)
        xy_test_partitions  = create_partitions_uniform(xy_test, num_partitions)
        
    elif partition == "90percent":
        xy_train_partitions = create_partitions_90percent(xy_train, num_partitions)
        xy_test_partitions  = create_partitions_90percent(xy_test, num_partitions)
        
    elif partition == "100percent":
        xy_train_partitions = create_partitions_100percent(xy_train, num_partitions)
        xy_test_partitions  = create_partitions_100percent(xy_test, num_partitions)

    return list(zip(xy_train_partitions, xy_test_partitions))
