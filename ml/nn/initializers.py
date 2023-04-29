from typing import Any
from ml.nn.functional.random import Variable ,math, np ,  glorot_uniform , uniform
from typing import overload

def zeros(shape , requires_grad :bool= None , dtype = np.float32 )-> Variable:
    return Variable(np.zeros(shape).astype(dtype), requires_grad= requires_grad)

def ones(shape , requires_grad:bool = None , dtype = np.float32)-> Variable:
    return Variable(np.ones(shape).astype(dtype), requires_grad= requires_grad)

@overload
def arange( stop  , requires_grad:bool = None , dtype = np.float32)-> Variable:
    return Variable(np.arange(stop).astype(dtype), requires_grad= requires_grad)

@overload
def arange( start , stop  , requires_grad:bool = None , dtype = np.float32)-> Variable:
    return Variable(np.arange(start , stop).astype(dtype), requires_grad= requires_grad)

def eye(N : int , M: int , k: int = 0 , requires_grad :bool = None ,dtype = np.float32):
    return Variable(np.eye(N , M , k , dtype = dtype), requires_grad= requires_grad)




    
class RandomNormal():
    def __call__(self, shape :tuple , dtype = np.float32 , requires_grad : bool =None )  -> Any:
        return Variable(np.random.randn(*shape).astype(dtype), requires_grad= requires_grad)

class Zeros():
    def __call__(self, shape :tuple , dtype = np.float32 , requires_grad : bool =None )  -> Any:
        return zeros(shape , requires_grad , dtype)
    
class Ones():
    def __call__(self, shape :tuple , dtype = np.float32 , requires_grad : bool =None )  -> Any:
        return ones(shape , requires_grad , dtype)

class GlorotUniform():
    def __call__(self, shape :tuple , dtype = np.float32 , requires_grad : bool =None )  -> Any:
         limit:float = math.sqrt( 6. / (shape[0] + np.prod(shape[1:])))
         size : int =  np.prod(shape)
         return Variable(np.random.uniform( -limit , limit , size).reshape(shape).astype(dtype) , requires_grad= requires_grad)
    
class Uniform():
    def __call__(self, shape :tuple ,low:float = 0 , high :float = 1 , dtype = np.float32 , requires_grad : bool =None )  -> Any:
        return uniform(shape , low , high , requires_grad , dtype)

class Identity():
    def __call__(self, shape :tuple , k :int =0, dtype = np.float32 , requires_grad : bool =None )  -> Any:
        return eye(shape[0],shape[1] , k , requires_grad , dtype)  

