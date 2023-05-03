import ml.autograd.Activations as _A
from ml.autograd.Activations import Variable  as __var , register_gradient as __register_gradient
from ml.nn.functional.math import exp as exponential
from ml.nn.functional.ops import where  as _Where
from math import pi as _pi , sqrt 

@__register_gradient(_A.Relu)
def relu(x:__var) -> __var:...

@__register_gradient(_A.Sigmoid)
def sigmoid(x:__var) -> __var:...

@__register_gradient(_A.Tanh)
def tanh(x:__var) -> __var:...

@__register_gradient(_A.Identity)
def identity(x:__var) -> __var:...

@__register_gradient(_A.Celu)
def celu(x:__var) -> __var:...

def silu(x:__var , beta : float = 1.0) ->__var:
    u = x if beta == 1 else x * beta 
    return x * sigmoid(u)

swish = silu

def leaky_relu(x : __var , α : float = 1e-2 ) ->__var:
    return x.where( x>=0 , x * α )

def elu(x: __var , α :float = 1.0) ->__var:
    return _Where(x < 0 , α * ( x.exp() - 1 ) , x  )

def gelu(x : __var , use_approx : bool = False) ->__var: 
    @__register_gradient(_A.Gelu)
    def _gelu(x:__var )->__var:...

    return _gelu(x) if not use_approx else \
            0.5 * x * (1 + tanh( sqrt(2 / _pi) * (x + 0.044715 * x**3)  )  )

def glu(x : __var , dim : int = -1)->__var:
    x , gate = x.split(2 , dim)
    return sigmoid(gate) * x

def swiglu(x : __var , dim : int = -1 )->__var: 
    x , gate = x.split(2 , dim)
    return silu(x) * gate

def softmax(x : __var , dim : int | tuple = -1) ->__var : 
    z = x - x.max(dim , keepdims= True)
    z = z.exp()
    return z / z.sum(dim , keepdims= True)


