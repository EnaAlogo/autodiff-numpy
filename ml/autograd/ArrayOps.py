from ml.Variable import Function
from ml.Variable import np
from typing import Optional , Tuple , List , Any


class Copy(Function):

    def __init__(self, x ) -> None:
        super(Copy , self).__init__(x)
    
    def __call__(self , x : np.ndarray ) -> np.ndarray :  
        return np.copy(x)
    
    def backward(self , g :  np.ndarray ) -> np.ndarray :
        return g

class Contiguous(Function):

    def __init__(self, x ) -> None:
        super(Contiguous , self).__init__(x)
    
    def __call__(self , x : np.ndarray ) -> np.ndarray :  
        return np.ascontiguousarray(x)
    
    def backward(self , g :  np.ndarray ) -> np.ndarray :
        return g
    
class Transpose(Function):

    def __init__(self, x ) -> None:
        super(Transpose , self).__init__(x)
    
    @staticmethod
    def __transpose(x : np.ndarray , axes : Tuple[int] | List[int]  = None) ->np.ndarray:
        if x.ndim == 1:
            return x , None
        if axes is None:
            return x.T , None
        if len(axes) == 2 :
            return np.swapaxes(x , axes[0] , axes[1]) , axes
        axes_ = [None]* len(axes)
        for i , dim in enumerate(axes):
            axes_[dim] = i
        return np.transpose(x , axes) , axes_
    
    def __call__(self , x : np.ndarray , axes : Tuple[int] | List[int]   = None ) -> np.ndarray :  
        ret , self.axes = Transpose.__transpose(x, axes)
        return ret
    
    def backward(self , g :  np.ndarray ) -> np.ndarray :
        return Transpose.__transpose(g, self.axes)[0]
    
class Reshape(Function):

    def __init__(self, x ) -> None:
        super(Reshape , self).__init__(x)
    
    def __call__(self , x : np.ndarray , shape: tuple |  list) -> np.ndarray :  
        return np.reshape(x , shape)
    
    def backward(self , g :  np.ndarray ) -> np.ndarray:
        return np.reshape(g ,self.parents[0].shape)
    
class Broadcast(Function):

    def __init__(self, x ) -> None:
        super(Broadcast , self).__init__(x)
    
    def __call__(self , x : np.ndarray , shape: tuple |  list ) -> np.ndarray :  
        return np.broadcast_to(x , shape)
    
    def backward(self , g :  np.ndarray ) -> np.ndarray :
        return Function.reverse_broadcast(self.parents[0].shape , g )
    
    
class Pad(Function):

    def __init__(self, x ) -> None:
        super(Pad , self).__init__(x)
    
    def __call__(self , x : np.ndarray , paddings: tuple | list ) -> np.ndarray :  
        if self.requires_grad:
            pads = [ pad if isinstance(pad , tuple) else tuple((pad,pad,)) for pad in (paddings 
                    if not isinstance(paddings , int ) else [paddings]) ]
            for i in range(len(pads) , x.ndim):
                pads.append( tuple((0,0,)) )
            self.slices : tuple[slice] = tuple( slice(f[0] , dim + f[0] ) for f , dim in zip(pads , x.shape ) )
        return np.pad(x , paddings)
    
    def backward(self , g :  np.ndarray ) -> np.ndarray :
        return g[self.slices]
    
class Flip(Function):

    def __init__(self, x ) -> None:
        super(Flip , self).__init__(x)
    
    def __call__(self , x : np.ndarray , axis: tuple | int | list = None ) -> np.ndarray :  
        self.axes = axis
        return np.flip(x , axis)
    
    def backward(self , g :  np.ndarray ) -> np.ndarray :
        return np.flip( g , self.axes )
    
class Squeeze(Function):

    def __init__(self, x ) -> None:
        super(Squeeze , self).__init__(x)
    
    def __call__(self , x : np.ndarray , axis : tuple | int | list = None ) -> np.ndarray :  
        return np.squeeze(x , axis)
    
    def backward(self , g :  np.ndarray ) ->  np.ndarray :
        return np.reshape(g , self.parents[0].shape)
    
class Unsqueeze(Function):

    def __init__(self, x ) -> None:
        super(Unsqueeze , self).__init__(x)
    
    def __call__(self , x : np.ndarray , axis : Tuple | int | list ) -> np.ndarray :  
        return np.expand_dims(x , axis)
    
    def backward(self , g :  np.ndarray ) ->  np.ndarray :
        return np.reshape(g , self.parents[0].shape)
    
    
class Cast(Function):
    def __init__(self, x ) -> None:
        super(Cast , self).__init__(x)
    
    def __call__(self , x : np.ndarray , dtype : np.dtype= 'float' ) -> np.ndarray :  
        return x.astype(dtype)
    
    def backward(self , g :  np.ndarray ) ->  np.ndarray :
         return g.astype(self.parents[0].dtype)
    

class Where(Function):
    def __init__(self, put , mask , otherwise ) -> None:
        super(Where , self).__init__(put , mask ,otherwise)
    
    def __call__(self , x : np.ndarray , mask: np.ndarray , y : np.ndarray ) -> np.ndarray :  
        self.mask = mask
        return np.where(mask , x , y )
    
    def backward(self , g :  np.ndarray ) -> tuple[ Optional[np.ndarray] ]:                     #mask gradient is always None
         return  Function.reverse_broadcast(self.parents[0].shape,np.where(self.mask , g ,0 )) if self.needs_grad(0) else None, None ,\
                  Function.reverse_broadcast(self.parents[2].shape,np.where(self.mask , 0  , g)) if self.needs_grad(2) else None 
    
class Concatenate(Function):
    def __init__(self, *tensors : tuple ) -> None:
        super(Concatenate , self).__init__(*tensors)
    
    def __call__(self , *tensors : tuple[np.ndarray] , axis: int = 0 ) -> np.ndarray : 
        self.axis = axis 
        return np.concatenate(tensors , axis= axis)
    
    def backward(self , g :  np.ndarray ) -> tuple[ Optional[np.ndarray] ]:
         indices : list = np.cumsum([self.parents[i].shape[self.axis] for i in range(len(self.parents))])
         return np.array_split( g , indices , axis = self.axis )
    
class Stack(Function):
    def __init__(self, *tensors : tuple ) -> None:
        super(Stack , self).__init__(*tensors)
    
    def __call__(self , *tensors : tuple[np.ndarray] , axis: int = 0 ) -> np.ndarray : 
        self.axis = axis
        return np.stack(tensors , axis= axis)
    
    def backward(self , g :  np.ndarray ) -> tuple[ Optional[np.ndarray] ]:
         res =  np.split( g , len(self.parents) , axis = self.axis )
         return ( s.squeeze(self.axis) if self.needs_grad(i) else None for s,i in zip(res , range(len(self.parents))) )
    

class Index(Function):

    def __init__(self , arr ):
        super(Index , self).__init__(arr)

    def __call__(self , x :np.ndarray , index :Any )-> np.ndarray:
        self.index = index
        self.input_shape = x.shape
        return x[index]
    
    def backward(self , g : np.ndarray ) -> Tuple[Optional[np.ndarray]]:
        output = np.zeros(self.input_shape)
        output[self.index] = g
        return output 
    

class Assign(Function):
        def __init__(self , x , y ):
            super(Assign , self).__init__(x ,y )

        def __call__(self , x : np.ndarray , y : np.ndarray , index :Any =None ) -> np.ndarray :    
            self.index = index
            x[index] = y
            return x
        
        def backward(self , g :  np.ndarray ) ->  np.ndarray:
           if self.needs_grad(0):
             dx = np.copy(g)
             dx[self.index] = 0
           else:
               dx = None
           return dx, g[self.index].reshape(self.parents[1].shape) if self.needs_grad(1) else None



    

    
