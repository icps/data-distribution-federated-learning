
#### Change these values ####


dataset_name   = "cifar10"
## options: cifar10, mnist

partition_type = "100percent"
## options: iid, uniform, 90percent, 100percent

num_rounds   = 100
num_clients  = 10
fraction_fit = 1


### client
epochs          = 7
batch_size      = 256
steps_per_epoch = "divide"
# if steps_per_epoch = 0, the clients do not use this variable
# if steps_per_epoch = "divide", the value is len(train) // batch_size
# common values: 10 when 100 percent



### server
filename = f"{dataset_name}_tests/{dataset_name}_{partition_type}.txt"
