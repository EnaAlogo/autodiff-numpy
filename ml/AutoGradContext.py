import numpy as np
import traceback
from enum import Enum , auto
import warnings

try :
    import cupy
except ModuleNotFoundError:
    warnings.warn('cupy is not available on this machine , using cpu only')
    import ml.dummy as cupy


class Context:
    no_grad :bool = False

class stop_gradient:

    def __init__(self , arg = None)->None:
        self.arg = arg

    def __call__(self , *args , **kwds):
        Context.no_grad = True 
        try:
         ret = self.arg(*args,**kwds)
         Context.no_grad = False if Context.no_grad == True else True
         return ret
        except Exception as e :
            Context.no_grad = False if Context.no_grad == True else True
            print(f'exception occured {e}')
            traceback.print_exc()
            return None
        
    
    def __enter__(self):
        Context.no_grad = True

    def __exit__(self , type, value, traceback):
        Context.no_grad = False if Context.no_grad == True else True

def get_backend(self):
   if self.device() == Device.CUDA:
       return cupy
   return np

def get_backend_from_device(dev):
    return np if dev == Device.CPU or dev is None  else cupy
   
def cuda_is_available():
    return cupy.cuda is not None

class Device(Enum):
    CPU = auto()
    CUDA = auto()


