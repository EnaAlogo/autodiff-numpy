import numpy as np
import ml 
from ml import tensor as Variable


def moments(x : Variable , axis : list | tuple = -1 , 
            keepdims = False , correction : int = 1 ):
    if isinstance(axis , int):
        axis = (axis,)
    mean : Variable = x.mean(axis , keepdims = True)
    shift = x - mean
    scale =  1.0 / (np.prod( [ x.shape[ax] for ax in axis ] )-correction)
             
    variance = (shift**2).sum(axis , keepdims) * scale

    return mean if keepdims else mean.squeeze(axis) , variance

def batch_norm(
               x : Variable , 
               mean : Variable , 
               variance : Variable , 
               gamma : Variable = None,
               beta : Variable = None ,
               eps : float =  1e-5
               ):
    inv_std = (variance+eps).rsqrt()
    y = (x - mean) * inv_std
    if gamma is not None:
        y = y * gamma
    if beta is not None:
        y =  y + beta
    return y
