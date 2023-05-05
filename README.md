# autograd_numpy
## Update: there is now an option to use cuda (only cupy supported for now but its extensible) ofcourse if there is no such package installed in a system the program can still run normally using only numpy (maybe , i think , i hope) , it can be somewhat slow sometimes and i havent fully tested it so it may just break, the code is at [experimental-cuda](https://github.com/EnaAlogo/autograd_numpy/tree/experimental-CUDA)
## This lil thing was done so i can visualize a quick prototype of what i will do in c++ with gpu support and more operations later.
## Implementations of the basic supported operations are at [ml/autograd](ml/autograd/), and some other operations are composition of multiple basic ops. 
## Optimizers (i will propably add more and use this thingy to quickly test an algorithm before implementing it in c++) are at [ml/nn/Optimizers](ml/nn/Optimizers.py)
## Losses like cosine similarity cross entropy etc are at [ml/nn/Losses](ml/nn/Losses.py)
## The main Variable class that represents a tensor of parameters is at [ml/Variable](ml/Variable.py)
## Theres also an example with the mnist dataset doing handwritten digits classification and an example using cuda mode and the cifar10 dataset to classify animal images with a CNN, the results are below :
## CNN cifar , clearly overfitting and thats okay its just meant to demonstrate that the training works(maybe)
![image](cnn_cifar10.png)
## MNIST digit classification
![image](mnist_results.png)

