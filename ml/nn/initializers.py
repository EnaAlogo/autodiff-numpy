from typing import Any
from ml.nn.functional.random import Variable ,math, np ,  glorot_uniform , uniform
from typing import overload

def zeros(shape , requires_grad :bool= None , dtype = np.float32 )-> Variable:
    return Variable(np.zeros(shape).astype(dtype), requires_grad= requires_grad)

def ones(shape , requires_grad:bool = None , dtype = np.float32)-> Variable:
    return Variable(np.ones(shape).astype(dtype), requires_grad= requires_grad)

def arange( x , y = None , step =1  , requires_grad:bool = None , dtype = np.float32)-> Variable:
    if y is None:
        start , stop = 0 , x
    else:
        start , stop = x,y
    return Variable(np.arange(start , stop , step ).astype(dtype), requires_grad= requires_grad)


def eye(N : int , M: int , k: int = 0 , requires_grad :bool = None ,dtype = np.float32):
    return Variable(np.eye(N , M , k , dtype = dtype), requires_grad= requires_grad)




    
class RandomNormal():
    def __init__(self , mean = 0 , std = 1):
        self.mean , self.std = mean, std

    def __call__(self, shape :tuple , dtype = np.float32 , requires_grad : bool =None )  -> Variable:
        return Variable(np.random.normal(self.mean , self.std , shape ).astype(dtype) , requires_grad= requires_grad)

class Zeros():
    def __call__(self, shape :tuple , dtype = np.float32 , requires_grad : bool =None )  -> Variable:
        return zeros(shape , requires_grad , dtype)
    
class Ones():
    def __call__(self, shape :tuple , dtype = np.float32 , requires_grad : bool =None )  -> Variable:
        return ones(shape , requires_grad , dtype)

class GlorotUniform():
    def __call__(self, shape :tuple , dtype = np.float32 , requires_grad : bool =None )  -> Variable:
         limit:float = math.sqrt( 6. / (shape[0] + np.prod(shape[1:])))
         size : int =  np.prod(shape)
         return Variable(np.random.uniform( -limit , limit , size).reshape(shape).astype(dtype) , requires_grad= requires_grad)
    
class Uniform():
    def __init__(self , low = 0 , high = 1 ):
        self.high , self.low = high , low 

    def __call__(self, shape :tuple, dtype = np.float32 , requires_grad : bool =None )  -> Any:
        return uniform(shape , self.low , self.high , requires_grad , dtype)

class Identity():
    def __call__(self, shape :tuple , k :int =0, dtype = np.float32 , requires_grad : bool =None )  -> Any:
        return eye(shape[0],shape[1] , k , requires_grad , dtype)  

