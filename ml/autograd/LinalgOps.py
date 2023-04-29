from ml.Variable import Function
from ml.Variable import np


class VectorDotProduct(Function):
     def __init__(self, x ,y ) -> None:
        super(VectorDotProduct , self).__init__(x , y)
    
     def __call__(self , x : np.ndarray , y: np.ndarray ) -> np.ndarray :
        if x.ndim != 1 or y.ndim != 1 : raise ValueError(f'invalid inputs for vdot expected 1D got {x.shape} , {y.shape} ')  
        self.x , self.y = x , y
        return np.vdot(x,y)
    
     def backward(self , g :  np.ndarray ) ->  np.ndarray:
         return  np.vdot(self.y , g)  if self.needs_grad(0) else None ,\
                 np.vdot(g , self.x) if self.needs_grad(1) else None
     
class MatMul(Function):
     def __init__(self, x ,y ) -> None:
        super(MatMul , self).__init__(x , y)
    
     def __call__(self , x : np.ndarray , y: np.ndarray ) -> np.ndarray : 
        if x.ndim != 2 or y.ndim != 2 : raise ValueError(f'invalid inputs for matmul expected 2D got {x.shape} , {y.shape} ') 
        self.x , self.y = x , y
        return np.matmul(x,y)
    
     def backward(self , g :  np.ndarray ) ->  np.ndarray:
         return  np.matmul(g , self.y.T)  if self.needs_grad(0) else None ,\
                 np.matmul(self.x.T , g) if self.needs_grad(1) else None

class Diag(Function):
     def __init__(self, x ) -> None:
        super(Diag , self).__init__(x)
    
     def __call__(self , x : np.ndarray , k:int  = 0 ) -> np.ndarray :  
        self.k = k
        return np.diag(x,k)
    
     def backward(self , g :  np.ndarray ) ->  np.ndarray:
         return np.diag(g,self.k)


class Diagonal(Function):
     def __init__(self, x ) -> None:
        super(Diagonal , self).__init__(x)
    
     def __call__(self , x : np.ndarray , k:int  = 0 , axis0 = 0 , axis1 = 1) -> np.ndarray :  
        self.k = k
        self.ax0 , self.ax1 = axis0 , axis1
        return np.diagonal(x , k , axis0 , axis1)
    
     def backward(self , g :  np.ndarray ) ->  np.ndarray:
         zeros:np.ndarray = np.zeros(self.parents[0].shape,
                                     dtype = self.parents[0].dtype)
         diagonal :np.ndarray = np.diagonal(
             zeros , self.k ,self.ax0 , self.ax1
         )
         diagonal.flags['WRITEABLE'] = True
         diagonal[:] = g
         return zeros
    

class _TriuTril(Function):
    
     def __call__(self , x : np.ndarray , k:int  = 0 ) -> np.ndarray :  
        self.k = k
        return self.fn(x , k)
    
     def backward(self , g :  np.ndarray ) ->  np.ndarray:
         return self.fn(g, self.k)
    
class Triu(_TriuTril):
     def __init__(self, x ) -> None:
        super(Triu , self).__init__(x)

     def fn(self , x :np.ndarray , k : int):
         return np.triu(x , k )
     
class Tril(_TriuTril):
     def __init__(self, x ) -> None:
        super(Tril , self).__init__(x)

     def fn(self , x :np.ndarray , k : int):
         return np.tril(x , k )
    

class Inverse(Function):
     
     def __init__(self, x ) -> None:
        super(Inverse , self).__init__(x)

     def __call__(self , x : np.ndarray , k:int  = 0 ) -> np.ndarray :  
        self.inv = np.linalg.inv(x)
        return self.inv
    
     def backward(self , g :  np.ndarray ) ->  np.ndarray:
         return - self.inv.T @ (g @ self.inv.T)
   
class PseudoInverse(Function):
     
     def __init__(self, x ) -> None:
        super(PseudoInverse , self).__init__(x)

     def __call__(self , x : np.ndarray , k:int  = 0 ) -> np.ndarray :
        self.x = x  
        self.inv = np.linalg.pinv(x)
        return self.inv
    
     def backward(self , g :  np.ndarray ) ->  np.ndarray: 
         #is this even correct xd?
         return (1 - (self.inv @ self.x)) @ g.T @ (self.inv.T @ self.inv)
     
   

    
########## util funcs for setting up multi dimensional dot product ###############################################

def will_it_need_transpose(rank : int , perms : list[int] ):
        assert len(perms) == rank
        return any([ perm != i for i , perm in zip(perms , range(rank) ) ])
    

def get_axes(rank :int , axes : int | tuple[int] | list[int] |tuple[tuple[int] | list[int]] | list[list[int] |tuple[int]]  ):
        if isinstance(axes , int):
            if axes < 0 : raise ValueError('axis must be positive')
            return tuple ( range(rank - axes) ) , tuple ( range( 0 , axes) )
        elif isinstance(axes , ( tuple , list ) ):
            if isinstance(axes[0] , int ) and isinstance(axes[1] , int):
                if len(axes) != 2 : raise ValueError('axes cannot be more than 2')
                return tuple( (axes[0] if axes[0] >= 0 else axes[0] + rank , \
                               axes[1] if axes[1] >= 0 else axes[1] + rank , ) )
            elif isinstance(axes[0] , (list , tuple)) and isinstance(axes[1] , (list , tuple)):
                ax0 , ax1 = axes[0] , axes[1]
                if len(ax0) != len(ax1) : raise ValueError('axes must have the same size')
                return tuple( ax if ax >= 0 else ax + rank for ax in ax0) ,\
                       tuple( ax if ax >= 0 else ax + rank for ax in ax1)
        else: raise ValueError('invalid type of axis parameter expected int or list/tuple of ints or'+
                               f'list/tuple of list/tuple of ints but got {axes.__class__}')
            

def get_reshape(shape :tuple[int] , axes: tuple[int] | list[int] | int , flipped : bool = False):
        axes = tuple((axes,)) if isinstance(axes, int) else axes
        free  = [i  for i in range(len(shape)) if i not in axes]
        free_dims = [shape[i] for i in free]

        prod_free :int = np.prod([shape[f] for f  in free])
        prod_axes :int = np.prod([ shape[ax] for ax in axes])

        (perm , new_shape) = ( list(axes) + free , tuple((prod_axes , prod_free),) ) if flipped else\
                           ( free + list(axes) , tuple((prod_free , prod_axes),) )
        
        return perm , free_dims , new_shape