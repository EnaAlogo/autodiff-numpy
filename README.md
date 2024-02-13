## There is also an option to use cuda with cupy at [experimental-cuda](https://github.com/EnaAlogo/autograd_numpy/tree/experimental-CUDA)
## Implementations of the basic supported operations are at [ml/autograd](ml/autograd/), and some other operations are composition of multiple basic ops. 
## Optimizers (i will propably add more and use this thingy to quickly test an algorithm before implementing it in c++) are at [ml/nn/Optimizers](ml/nn/Optimizers.py)
## Losses like cosine similarity cross entropy etc are at [ml/nn/Losses](ml/nn/Losses.py)
## The main Variable class that represents a tensor of parameters is at [ml/Variable](ml/Variable.py)
## Theres also an example with the mnist dataset doing handwritten digits classification and an example using cuda mode and the cifar10 dataset to classify animal images with a CNN, the results are below :
## MNIST digit classification
![image](mnist_results.png)

