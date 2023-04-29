from ml.Variable import Function
from ml.Variable import np
from typing import Optional


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
    
    def __call__(self , x : np.ndarray , axes : tuple |  list= None ) -> np.ndarray :  
        self.axes = axes
        return np.transpose(x , axes)
    
    def backward(self , g :  np.ndarray ) -> np.ndarray :
        return np.transpose(g ,self.axes)
    
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
    
    def __call__(self , x : np.ndarray , axis : tuple | int | list ) -> np.ndarray :  
        return np.expand_dims(x , axis)
    
    def backward(self , g :  np.ndarray ) ->  np.ndarray :
        return np.reshape(g , self.parents[0].shape)
    
class Slice(Function):
    def __init__(self, x ) -> None:
        super(Slice , self).__init__(x)
    
    def __call__(self , x : np.ndarray , slices : tuple[ slice | int ]  ) -> np.ndarray :
        self.dims_to_expand : list[int] = []  
        self.paddings : list[int] = []
        if self.requires_grad:
           _slices : list[slice | int] = [None]*x.ndim
           for  i , dim in enumerate(x.shape):
               _slices[i] = slices[i] if i < len(slices) else slice(0,dim)
           for i , (s, p) in enumerate(zip(x.shape , _slices)):
               if isinstance(p , slice):
                   p0 , p1 = 0 if p.start is None else p.start , s if p.stop is None else p.stop
                   self.paddings.append( (p0 if p0 >=0 else s + p0 ,
                                           s - p1 if p1 >=0 else  s - (p1 + s )  ) ) 
               else:
                    self.dims_to_expand.append(i)
                    self.paddings.append((0,s-1))
        return x[slices]
    
    def backward(self , g :  np.ndarray ) -> np.ndarray :
        return np.pad(np.expand_dims(g , self.dims_to_expand) , self.paddings)
    
    
class Cast(Function):
    def __init__(self, x ) -> None:
        super(Cast , self).__init__(x)
    
    def __call__(self , x : np.ndarray , dtype : np.dtype ) -> np.ndarray :  
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
         return  np.where(self.mask , g ,0 ) if self.needs_grad(0) else None, None ,\
                 np.where(self.mask , 0  , g) if self.needs_grad(2) else None 
    
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
        print(tensors) 
        return np.stack(tensors , axis= axis)
    
    def backward(self , g :  np.ndarray ) -> tuple[ Optional[np.ndarray] ]:
         res =  np.split( g , len(self.parents) , axis = self.axis )
         return ( s.squeeze(self.axis) if self.needs_grad(i) else None for s,i in zip(res , range(len(self.parents))) )
    
