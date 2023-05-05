from typing import Any
from ml.AutoGradContext import get_backend_from_device
from ml.nn.functional.random import Variable ,math, np ,  glorot_uniform , uniform
from typing import overload
import functools

def zeros(shape , requires_grad :bool= None , dtype = np.float32, device = None )-> Variable:
    device = device.upper() if isinstance(device , str) else device
    backend = get_backend_from_device(device)
    return Variable(backend.zeros(shape).astype(dtype), requires_grad= requires_grad)

def ones(shape , requires_grad:bool = None , dtype = np.float32, device = None)-> Variable:
    device = device.upper() if isinstance(device , str) else device
    backend = get_backend_from_device(device)
    return Variable(backend.ones(shape).astype(dtype), requires_grad= requires_grad)

def arange( x , y = None , step =1  , requires_grad:bool = None , dtype = np.float32, device = None)-> Variable:
    device = device.upper() if isinstance(device , str) else device
    backend = get_backend_from_device(device)
    if y is None:
        start , stop = 0 , x
    else:
        start , stop = x,y
    return Variable(backend.arange(start , stop , step ).astype(dtype), requires_grad= requires_grad)


def eye(N : int , M: int , k: int = 0 , requires_grad :bool = None ,dtype = np.float32, device = None):
    device = device.upper() if isinstance(device , str) else device
    backend = get_backend_from_device(device)
    return Variable(backend.eye(N , M , k , dtype = dtype), requires_grad= requires_grad)




    
class RandomNormal():
    def __init__(self , mean = 0 , std = 1):
        self.mean , self.std = mean, std

    def __call__(self, shape :tuple , dtype = np.float32 , requires_grad : bool =None, device = None )  -> Variable:
        device = device.upper() if isinstance(device , str) else device
        backend = get_backend_from_device(device)
        return Variable(backend.random.normal(self.mean , self.std , shape ).astype(dtype) , requires_grad= requires_grad)

class Zeros():
    def __call__(self, shape :tuple , dtype = np.float32 , requires_grad : bool =None , device = None)  -> Variable:
        return zeros(shape , requires_grad , dtype , device=device)
    
class Ones():
    def __call__(self, shape :tuple , dtype = np.float32 , requires_grad : bool =None , device = None)  -> Variable:
        return ones(shape , requires_grad , dtype , device=device)

class GlorotUniform():
    def __call__(self, shape :tuple , dtype = np.float32 , requires_grad : bool =None, device = None )  -> Variable:
         device = device.upper() if isinstance(device , str) else device
         backend = get_backend_from_device(device)
         limit:float = math.sqrt( 6. / (shape[0] +  functools.reduce( lambda y , f : y*f , shape[1:] , 1 )))
         size : int =  functools.reduce( lambda y , f : y*f , shape , 1 )
         return Variable(backend.random.uniform( -limit , limit , size).reshape(shape).astype(dtype) , requires_grad= requires_grad)
    
class Uniform():
    def __init__(self , low = 0 , high = 1 ):
        self.high , self.low = high , low 

    def __call__(self, shape :tuple, dtype = np.float32 , requires_grad : bool =None, device = None  )  -> Any:
        return uniform(shape , self.low , self.high , requires_grad , dtype, device = None )

class Identity():
    def __call__(self, shape :tuple , k :int =0, dtype = np.float32 , requires_grad : bool =None, device = None  )  -> Any:
        return eye(shape[0],shape[1] , k , requires_grad , dtype, device = None )  

