from ml import nn
from ml.nn.functional import activations , linalg , math , ops , random , reduce
from ml.nn import initializers
from ml.nn.functional.ops import to_tensor
from ml.nn.initializers import zeros , ones, eye , arange
from ml.nn.functional.linalg import dot , tensordot
from ml.nn import Losses as losses , Optimizers as optimizers
from ml.Variable import Variable as tensor
from ml.AutoGradContext import stop_gradient , cuda_is_available


