from ml.Variable import Function 
from ml.Variable import np


class Variance(Function):

    def __init__(self , x  ) ->None:
        super(Variance , self ).__init__(x)

    def __call__(self , x : np.ndarray , axis:tuple = None , keepdims : bool = False,
                 correction : int = 1 ) -> np.ndarray :    
        # var(x) = 1/N-1 Σ{1..Ν}(x-μ)² 
        μ : np.ndarray  = x.mean(axis , True)
        self.shifted: np.ndarray  = x - μ
        self.unsqueezed_shape = μ.shape
        self.scale = 1. / (np.prod([x.shape[a] for a in self.axes]) - correction)
        ret = (self.shifted**2).sum(axis , keepdims)
        ret *= self.scale
        return  ret
    
    def backward(self , g :  np.ndarray ) ->  np.ndarray:

        # var'(x) = (x-μ)*(2/Ν-1) && chain rule
        broadcasted_grad : np.ndarray = np.broadcast_to(np.reshape(g, self.unsqueezed_shape)
                                      ,self.shifted.shape)
        self.shifted *= broadcasted_grad
        self.shifted *= (self.scale * 2)
        return self.shifted
    

class _MinMax(Function):
    def __call__(self , x : np.ndarray , axis:tuple = None , keepdims : bool = False) ->np.ndarray:
         y :np.ndarray =  self.meta_reduction(x , axis , True)
         self.input_shape = x.shape
         self.unsqueezed_shape = y.shape
         self.mask = x == y # saves the positions where the maxes where found during reduction
         return  y if keepdims else y.squeeze(axis)
    
    def backward(self , g : np.ndarray) -> np.ndarray :
        broadcasted_grad = np.broadcast_to(np.reshape(g,self.unsqueezed_shape),self.input_shape)
        #max backwards is the gradients in the positions where the maxes were found and rest is zeros
        return broadcasted_grad * self.mask

class Max(_MinMax ):
    def __init__(self , x ) ->None:
        super(Max , self).__init__(x)
    def meta_reduction(self, x : np.ndarray , axes:tuple = None , keepdims : bool = False) ->np.ndarray:
        return x.max(axes , keepdims= keepdims)
class Min( _MinMax):
    def __init__(self , x ) ->None:
        super(Min , self).__init__(x)
    def meta_reduction(self,x : np.ndarray , axes:tuple = None , keepdims : bool = False) ->np.ndarray:
        return x.min(axes , keepdims= keepdims)


class Mean(Function):

    def __init__(self , x ) ->None:
        super(Mean ,self ).__init__(x)

    def __call__(self ,  x : np.ndarray , axis : tuple| int = None , keepdims :bool = False) ->np.ndarray:
        axes = (axis,) if isinstance(axis , int) else\
        tuple( ax if ax >= 0 else ax + x.ndim for ax in axis)\
                    if axis is not None else tuple(range(x.ndim))
        self.scale = 1.0/ np.prod([x.shape[a] for a in axes]) # mean is sum / the product of the dimensions that are being reduced 
        out : np.ndarray = x.mean(axes , keepdims= True)
        self.inputshape = x.shape
        self.outputshape = out.shape
        return out if  keepdims else out.squeeze(axes)
    
    def backward(self , g :np.ndarray) -> np.ndarray:
               # derivative of x * y for dx is y * chaingrad, y doesnt require gradient its a scalar = 1/reduced dims
        return self.backend.broadcast_to(self.backend.reshape(g * self.scale,self.outputshape)#keep the 1s before the squeeze for correct broadcast
                              ,self.inputshape) #sum backwards is broadcasting

class Sum(Function):

    def __init__(self , x  ) ->None:
        super(Sum ,self ).__init__(x)

    def __call__(self ,  x : np.ndarray , axis : tuple | int = None , keepdims : bool = False) ->np.ndarray:     
        out : np.ndarray = x.sum(axis, keepdims= True)
        self.inputshape = x.shape
        self.outputshape = out.shape
        return out if  keepdims else out.squeeze(axis)
    
    def backward(self , g :np.ndarray) -> np.ndarray:
        return self.backend.broadcast_to(self.backend.reshape(g,self.outputshape)  #keep the 1s before the squeeze for correct broadcast
                               ,self.inputshape) #sum backwards is broadcasting

class Norm(Function):

    def __init__(self , x  ) ->None:
        super(Norm ,self ).__init__(x)

    def __call__(self ,  x : np.ndarray , axis : tuple | int = None , keepdims : bool = False) ->np.ndarray:     
        self.out : np.ndarray = self.backend.linalg.norm(x , axis = axis , keepdims= keepdims)
        self.x = x
        self.outputshape = self.out.shape
        return self.out if  keepdims else self.out.squeeze(axis)
    
    def backward(self , g :np.ndarray) -> np.ndarray:
        broadcasted_grad :np.ndarray = self.backend.broadcast_to(
            self.backend.reshape(g , self.outputshape) , self.x.shape
        )
        div_anorm : np.ndarray = self.x / self.out

        div_anorm *= broadcasted_grad

        return div_anorm
