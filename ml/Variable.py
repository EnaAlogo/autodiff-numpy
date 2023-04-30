from __future__ import annotations
from typing import Any
import numpy as np
import math
from ml.AutoGradContext import Context 


class Function: # parent class for all operations trackable by autograd engine
    def __init__(self, *vars : Variable) -> None :
        self.parents :tuple[Variable] = vars #parent nodes of the operation
        self.requires_grad :bool = any([ x.requires_grad for x in self.parents])
    
    def needs_grad(self , i : int ):
        return self.parents[i].requires_grad
    
    @staticmethod #utility method cuz binary ops are doing auto broadcast and we need to undo it on backwards
    def reverse_broadcast(pre :tuple[int] , post :np.ndarray )->np.ndarray:  
        def _axes(pre , post ):
            ones = len(post) - len(pre)
            return [ i  for i, t in enumerate(zip( post , [1 for _ in range(ones)] + list(pre) ))\
                 if t[0] != t[1] and t[0] !=1 ]
        return post.sum( tuple(_axes(pre , post.shape)) ,keepdims=True).reshape(pre)\
               if pre != post.shape else post
    
    @staticmethod
    def is_variable(x :Any): return isinstance(x , Variable)


# decorator that takes care of constructing autograd nodes 
def register_gradient( Op : type[Function] ) -> function : 
    def decorator( _ : function ) -> function :
        def invoke(*args : Variable , **kwargs) -> Variable :
            functor : Function = Op(*args)
            out = Variable ( functor(*[x.numpy for x in args],**kwargs) ,
                             requires_grad = functor.requires_grad, is_leaf = False )
            if functor.requires_grad and not Context.no_grad:
                out.grad_fn = functor
            return out
        return invoke
    return decorator

# decorator ,handles inplace operations that only currently partially supported
def inplace_operation( Op : type[function] ) ->function:
    def decorator( _ : function ) ->function:
        def invoke(*args : Variable) -> None:
            t : tuple[Variable] = tuple( x if isinstance(x,Variable) else Variable(x,requires_grad=False) for x in args  )
            any_grad :bool = any([x.requires_grad for x in t])
            if Context.no_grad or not any_grad:
               Op( *[ x.numpy for x in t ] )
               return t[0] # the first tensor is the one being mutated maybe?? dunno seems to work
            else:
                raise RuntimeError('inplace operations only supported for no_grad mode/tensors')
        return invoke
    return decorator



from ml.autograd import ArithmeticOps , ArrayOps, LinalgOps , Reductions
from ml.autograd.LinalgOps import will_it_need_transpose , get_axes , get_reshape

class Variable: #  tensor of parameters and its gradient

    def __init__(self ,
                  buffer : np.ndarray ,
                  requires_grad : bool = None ,
                  is_leaf : bool = None,
                  dtype = None) ->None:
        dtype = dtype if dtype is not None else np.float32
        if isinstance(buffer , float) or isinstance(buffer , int):
            buffer = np.array([buffer],dtype=dtype)
        elif isinstance(buffer, list) or isinstance(buffer , tuple):
            buffer = np.array(buffer,dtype=dtype)
        elif isinstance(buffer , np.ndarray) and buffer.dtype != dtype:
            dtype = buffer.dtype
        
        self.data :np.ndarray = buffer
        self.grad  : np.ndarray = None
        self.grad_fn : type[Function] = None
        self.__requires_grad :bool = requires_grad if requires_grad is not None else \
                               buffer.dtype in (np.float16 , np.float32 , np.float64 , np.complex64 , np.complex128)\
                               and not Context.no_grad
        self.__is_leaf :bool = is_leaf if is_leaf is not None else True
    

    @property
    def size(self) -> int : return self.data.size
    @property
    def numpy(self) ->np.ndarray :return self.data
    @property
    def gradient(self) -> Variable : return Variable(self.grad  , requires_grad= False) if self.grad  is not None else None
    @property
    def ndim(self) -> int :return self.data.ndim
    @property
    def shape(self) -> tuple[int] :return self.data.shape
    @property
    def T(self)-> Variable: return self.transpose()
    @property
    def dtype(self) -> np.dtype : return self.data.dtype
    @property
    def requires_grad(self) -> bool : return self.__requires_grad
    @requires_grad.setter
    def requires_grad(self , val : bool) ->None:
        if not isinstance(val , bool):raise ValueError(f'value must be of type bool but got {val.__class__}')
        if self.dtype not in (np.float16 , np.float32 , np.float64 , np.complex64 , np.complex128):
            raise ValueError('only floating point and complex number tensors can require gradients')
        self.__requires_grad = val

    def detach(self)-> Variable:
        return Variable(self.data , False)

    def retain_grad(self)->None:
        if not self.__requires_grad:
            raise RuntimeError('variable does not require grad')
        self.__is_leaf = True
    
    def item(self) :
        return self.data.item()

    def __len__(self) ->int : return self.size

    def __hash__(self) ->int : return id(self)

    def __repr__(self) -> str :
        return f'{self.data.__repr__()} ,  grad_fn = {self.grad_fn.__class__}'\
               if self.__requires_grad and self.grad_fn is not None else self.data.__repr__()
    

    def __getitem__(self , slices)-> Variable:
        return self.__index(index = slices)
        
            
    
    def __matmul__(self , y ) -> Variable :
        return self.dot(y)
    def __add__(self , y)-> Variable:
        y = Variable(y , False) if not isinstance(y,Variable) else y
        return self.__add(y)
    def __sub__(self , y)-> Variable:
        y = Variable(y , False) if not isinstance(y,Variable) else y
        return self.__sub(y)
    def __mul__(self , y)-> Variable:
        y = Variable(y , False) if not isinstance(y,Variable) else y
        return self.__mul(y)
    def __truediv__(self , y)-> Variable:
        y = Variable(y , False) if not isinstance(y,Variable) else y
        return self.__div(y)
    def __pow__(self , y):
        y = Variable(y , False) if not isinstance(y,Variable) else y
        return self.__pow(y)

    def __rsub__(self , y )-> Variable:
        y = Variable(y , False) if not isinstance(y,Variable) else y
        return y - self

    def __radd__(self ,y )-> Variable:
        y = Variable(y, False) if not isinstance(y,Variable) else y
        return y + self
    
    def __rtruediv__(self , y )-> Variable:
        y = Variable(y, False) if not isinstance(y,Variable) else y
        return y / self
    
    def __rmul__(self , y )-> Variable:
        y = Variable(y, False) if not isinstance(y,Variable) else y
        return y * self
    
    def __neg__(self)-> Variable:
        return 0-self
     

    def broadcast_to(self , *shape: list | tuple)-> Variable:
        return self.__broadcast_to( shape = shape)

    def reshape(self , *shape : list | tuple)-> Variable:
        return self.__reshape( shape = shape if shape else None)
    
    def transpose(self , *axes : list | tuple )-> Variable:
        return self.__transpose( axes = axes if axes else None)
    
    def flip(self , *axes : list | tuple )-> Variable:
        return self.__flip( axis = axes if axes else None)
    
    def pad(self , pads  : list[tuple[int]] | tuple[tuple[int]] | int )-> Variable:
        return self.__pad( paddings = pads)
    
    def squeeze(self , *axis  )-> Variable:
        if isinstance(axis , (list,tuple)) and len(axis) == 1:
           axis  = axis[0]
        return self.__squeeze( axis = axis )
    
    def unsqueeze(self , dims)-> Variable:
        return self.__unsqueeze( axis = dims)

    def astype(self , type : np.dtype)-> Variable:
        return self.__cast(dtype = type )
    
    def cat(self , *tensors , axis :int = 0 ) -> Variable:
        return Variable.__concat(  *([self,] + list(*tensors)) , axis = axis )
    
    def stack(self , *tensors , axis :int = 0 ) -> Variable:
        return Variable.__stack(  *([self,] + list(*tensors)) , axis = axis )
    
    # this will be tracked by autograd via doing multiple slices
    def unstack(self, axis : int = 0 ) -> tuple[Variable]:
        dim = axis if axis >=0 else axis + self.ndim
        if dim >= self.ndim : raise ValueError('invalid axis not within tensor dims')
        num_splits :int = self.shape[axis]
        splits : list[slice | int ] = [ slice(0,dim) if i != dim else 0  for i ,dim in enumerate(self.shape) ]
        ret : list[Variable] = [None]*num_splits
        for s , i in enumerate(splits):
            splits[dim] = i
            ret[i] = self.__getitem__(tuple(splits))
        return tuple(ret)

    # this will be tracked by autograd via doing multiple slices
    def split(self , num_splits : int ,  axis : int = 0 ) ->tuple[Variable]:
        axis = axis if axis >=0 else axis + self.ndim
        if axis >= self.ndim : raise ValueError('invalid axis not within tensor dims')
        splits : list[list[slice]] = [ [slice( dim ) for dim in self.shape] for _ in range(num_splits)  ]
        inc : int = self.shape[axis] // num_splits
        c : int  = 0 
        rmdr : int  = self.shape[axis] % num_splits
        if rmdr == 0:
            for i in range(0 , self.shape[axis] ,inc):
                splits[c][axis] = slice(i , min(self.shape[axis], i + inc))
                c+=1
        else:
            i = 0
            while c < rmdr:
                 splits[c][axis] = slice(i , min(self.shape[axis], i + inc+1))
                 c+=1
                 i+=inc
            while c < num_splits:
                splits[c][axis] = slice(i , min(self.shape[axis], i + inc))
                c+=1
                i+=inc
        return ( self.__getitem__(tuple(splits[i])) for i in range(num_splits)  )
    

########## these operations are not differentiable ########################################################
    #no need to track them with the autograd engine just return a new variable that doesnt require grad
    def __invert__(self)->Variable:#bool tensors can never require gradients
        return Variable( ~self.data , False)
    def __ne__(self , y)-> Variable:
        y = Variable(y , False) if not isinstance(y,Variable) else y
        return Variable(self.data != y.data , False)
    def __lt__(self , y)-> Variable:
        y = Variable(y , False) if not isinstance(y,Variable) else y
        return Variable(self.data < y.data , False)
    def __le__(self , y)-> Variable:
        y = Variable(y , False) if not isinstance(y,Variable) else y
        return Variable(self.data <= y.data , False)
    def __ge__(self , y)-> Variable:
        y = Variable(y , False) if not isinstance(y,Variable) else y
        return Variable(self.data >= y.data , False)
    def __gt__(self , y)-> Variable:
        y = Variable(y , False) if not isinstance(y,Variable) else y
        return Variable(self.data > y.data , False)
    def __eq__(self , y)-> Variable:
        y = Variable(y , False) if not isinstance(y,Variable) else y
        return Variable(self.data == y.data , False)
    def step_function(self)-> Variable:
        return Variable(np.where(self.data >= .5 , 1 , 0) , False )
    def argsort(self , axis : int = None)-> Variable:
        return Variable(np.argsort(self.data , axis) , False)
    def argmax(self , axis : int = None)-> Variable:
        return Variable(np.argmax(self.data ,axis=axis) , False )
    def argmin(self , axis : int = None)-> Variable:
        return Variable(np.argmin(self.data ,axis=axis) , False )
###############################################################################################################

    def max(self ,axis = None, keepdims : bool = False) -> Variable:
        return self.__max(axis = axis , keepdims = keepdims)
    def min(self ,axis = None, keepdims : bool = False) :
        return self.__min(axis = axis , keepdims = keepdims)
    def variance(self ,axis = None, keepdims : bool = False, correction:int = 1)-> Variable :
        return self.__variance(axis = axis , keepdims = keepdims , correction = correction)
    def sum(self ,axis = None, keepdims : bool = False) -> Variable:
        return self.__sum(axis = axis , keepdims = keepdims)
    def mean(self ,axis = None, keepdims : bool = False) -> Variable:
        return self.__mean(axis = axis , keepdims = keepdims)
    def std(self ,axis = None, keepdims : bool = False, correction:int = 1)-> Variable: 
        return self.variance(axis,keepdims,correction).sqrt()
    def norm(self ,axis = None, keepdims : bool = False) -> Variable:
        return self.__norm(axis = axis , keepdims = keepdims)


    def maximum(self , y)-> Variable:
        y = Variable(y , False) if not isinstance(y,Variable) else y
        return self.__maximum(y)
    def minimum(self , y)-> Variable:
        y = Variable(y , False) if not isinstance(y,Variable) else y
        return self.__minimum(y)
    
    def rsqrt(self)-> Variable:
        return self**(-.5)
    
    def clamp(self , low , high)-> Variable:
        return self.minimum( self.maximum(low) , high )


    def diag(self , k =0 )-> Variable:
        return self.__diag(k = k)
    def diagonal(self , k =0 , axis0 = 0 , axis1 = 1 ):
        return self.__diagonal(k = k , axis0 = axis0 , axis1 = axis1)
    def triu(self , k =0 )-> Variable:
        return self.__triu(k = k)
    def tril(self , k =0 )-> Variable:
        return self.__tril(k = k)
    def trace(self , offset = 0  , axis0 = 0 , axis1 = 1)-> Variable:
        return self.diagonal(offset , axis0 , axis1).sum(self.ndim - 2)
    
    def inner_product(self,  y ) ->Variable:
        y = y if isinstance(y , Variable) else Variable(y , False)
        if self.ndim == 1 and y.ndim == 1:
            return self.vdot(y)
        return self.tensordot(y , [-1 , y.ndim - 2 ])
    
    def outer_product(self , y) ->Variable:
          y = y if isinstance(y , Variable) else Variable(y , False)
          return self.reshape(-1 , 1) * y.reshape(1 , -1)
    
    def where(self , mask : Variable , x )->Variable:
        if mask.dtype != np.dtype('bool') : raise ValueError('condition has to be bool')
        x = x if isinstance(x ,Variable) else Variable(x, False)
        return self.__where(mask , x)

    def dot(self , y) -> Variable:
        y = y if isinstance(y , Variable) else Variable(y , False)
        if self.ndim == 1 and self.size == 1 or y.ndim == 1 and y.size == 1:
            return self * y
        elif self.ndim ==1 and y.ndim ==1 :
            return self.vdot(y)
        elif self.ndim == 2 and y.ndim ==2 :
            return self.matmul(y)
        elif y.ndim == 1:
            return self.tensordot(y,[-1,0])
        else:
            return self.tensordot(y,[-1,y.ndim - 2])
        
    def l2_normalize(self, axis= None , eps = 1e-12) ->Variable:
        return self / (self*self).sum(axis , True)\
                          .maximum(eps)\
                          .sqrt() 

    def tensordot(self , x : Variable , axes) -> Variable:
        if not isinstance(x , Variable):
            raise ValueError(f'tensordot expects the type of x to be Variable but got {x.__class__}')
        def _reshape(t : Variable , axes_ , flipped: bool = False):
            perm , fdims , nshape = get_reshape(t.shape , axes_ , flipped)
            t_trans :Variable = t.T if  will_it_need_transpose(t.ndim , perm) else t
            reshape_t = t_trans.reshape(*nshape) if t_trans.shape != nshape else t_trans
            return reshape_t , fdims
        a_axes , b_axes = get_axes(self.ndim , axes)
        reshape_a , a_fdims = _reshape( self , a_axes)
        reshape_b , b_fdims = _reshape( x , b_axes , True )
        ction : list[int] = a_fdims + b_fdims
        ab_matmul = reshape_a.matmul(reshape_b)
        return ab_matmul.reshape(*ction) if ction != ab_matmul.shape else ab_matmul
       
    
##################### linalg ########################################
    @register_gradient(LinalgOps.PseudoInverse)
    def p_inverse(self)->Variable:...
    @register_gradient(LinalgOps.Inverse)
    def inverse(self)->Variable:...
    @register_gradient(LinalgOps.Diag)
    def __diag(self , k = 0 ):...
    @register_gradient(LinalgOps.Diagonal)
    def __diagonal(self , k = 0 , axis0 = 0 , axis1 = 1):...
    @register_gradient(LinalgOps.Triu)
    def __triu(self , k = 0 ):...
    @register_gradient(LinalgOps.Tril)
    def __tril(self , k = 0 ):...
################### unary ops ####################################
    @register_gradient(ArithmeticOps.Logarithm)
    def log(self) -> Variable :...
    @register_gradient(ArithmeticOps.SquareRoot)
    def sqrt(self) -> Variable :...
    @register_gradient(ArithmeticOps.AbsoluteValue)
    def abs(self) -> Variable :...
    @register_gradient(ArithmeticOps.Exp)
    def exp(self) -> Variable :...
    @register_gradient(ArithmeticOps.Cosine)
    def cos(self) ->Variable:...
    @register_gradient(ArithmeticOps.Sine)
    def sin(self) ->Variable:...
    @register_gradient(ArrayOps.Contiguous)
    def contiguous(self) ->Variable:...
    @register_gradient(ArrayOps.Copy)
    def clone(self) ->Variable:...
############## reduce ops ####################################
    @register_gradient(Reductions.Variance)
    def __variance(self ,axis = None, keepdims : bool = False , correction : int = 1) :...
    @register_gradient(Reductions.Mean)
    def __mean(self ,axis = None, keepdims : bool = False) :...
    @register_gradient(Reductions.Sum)
    def __sum(self ,axis = None, keepdims : bool = False) :...
    @register_gradient(Reductions.Max)
    def __max(self ,axis = None, keepdims : bool = False) :...
    @register_gradient(Reductions.Min)
    def __min(self , axis = None, keepdims : bool = False):...
    @register_gradient(Reductions.Norm)
    def __norm(self , axis = None, keepdims : bool = False):...
############# array ops ###################################
    @register_gradient(ArrayOps.Cast)
    def __cast(self , dtype = np.float32 ):...
    @register_gradient(ArrayOps.Reshape)
    def __reshape(self , shape = [] ):...
    @register_gradient(ArrayOps.Transpose)
    def __transpose(self , axes = None ):...
    @register_gradient(ArrayOps.Flip)
    def __flip(self , axis = None ):...
    @register_gradient(ArrayOps.Pad)
    def __pad(self , paddings = None ):...
    @register_gradient(ArrayOps.Squeeze)
    def __squeeze(self , axis = None ):...
    @register_gradient(ArrayOps.Unsqueeze)
    def __unsqueeze(self , axis = [] ):...
    @register_gradient(ArrayOps.Broadcast)
    def __broadcast_to(self , shape = [] ):...
    @register_gradient(ArrayOps.Where)   
    def __where(self , mask : Variable ,  y : Variable) -> Variable :... 
    @register_gradient(ArrayOps.Concatenate)
    @staticmethod
    def __concat(*seq : tuple[Variable] , axis :int = 0 ) ->Variable : ...
    @register_gradient(ArrayOps.Stack)
    @staticmethod
    def __stack(*seq : tuple[Variable] , axis :int = 0 ) ->Variable : ...
    @register_gradient(ArrayOps.Index)
    def __index(self , index = None) ->Variable:...

############# binary ops ###############################################
    @register_gradient(ArithmeticOps.Add)
    def __add(self , y: Variable):...
    @register_gradient(ArithmeticOps.Subtract)
    def __sub(self , y: Variable ):...
    @register_gradient(ArithmeticOps.Divide)
    def __div(self , y : Variable):...
    @register_gradient(ArithmeticOps.Multiply)
    def __mul(self , y: Variable)  :...
    @register_gradient(ArithmeticOps.Power)
    def __pow(self , y : Variable):...
    @register_gradient(ArithmeticOps.Maximum)
    def __maximum(self , y : Variable):...
    @register_gradient(ArithmeticOps.Minimum)
    def __minimum(self , y : Variable):...
    @register_gradient(LinalgOps.MatMul)
    def matmul(self , x : Variable) -> Variable :...
    @register_gradient(LinalgOps.VectorDotProduct)
    def vdot(self , x : Variable) -> Variable :...

    
################## back prop ####################################

    def __backprop(self) -> list[Variable] : # topological sort 
        def bprop(node, visited, nodes):
          visited.add(node)
          if node.grad_fn:
            for i in node.grad_fn.parents:
              if i not in visited: bprop(i, visited, nodes)
            nodes.append(node)
          return nodes
        return bprop(self, set(), [])
    
    def backward(self , input_grad :Variable = None) -> None :
        self.grad  : np.ndarray = np.ones( shape = self.shape , dtype  = self.dtype) \
                                 if input_grad is None else input_grad
        graph : list[Variable] = self.__backprop()
        for node in reversed(graph):
            if not node.grad_fn.requires_grad: continue
            assert node.grad  is not None
            grads = node.grad_fn.backward(node.grad  ) 
            grads = [grads] if len(node.grad_fn.parents) == 1 else grads
            for parent , grad in zip(node.grad_fn.parents, grads):
                if grad is not None and parent.__requires_grad:
                    assert grad.shape == parent.shape , f'{grad.shape} != {parent.shape}'
                    """
                    if the node is a leaf node which means it will retain gradient
                    we make sure the gradient is contiguous otherwise we dont rly care
                    and then we accumulate the gradients
                    """
                    if parent.__is_leaf and parent.grad  is None:
                        parent.grad  = np.ascontiguousarray(grad)
                    elif parent.grad  is None:
                        parent.grad  = grad
                    else:
                        parent.grad  += grad
        for node in graph: # clean gradient memory from non leaf tensors and delete all graph nodes
            if not node.__is_leaf:
                del node.grad
                node.grad= None
            del node.grad_fn
            node.grad_fn = None


#################### inplace operations ###################################
    @staticmethod 
    def __inplacesub(x : np.ndarray , y : np.ndarray)-> None:
        x -= y
    @staticmethod
    def __inplacemul(x: np.ndarray , y :np.ndarray )-> None:
        x *= y
    @staticmethod
    def __inplaceadd(x : np.ndarray , y : np.ndarray)-> None:
        x += y
    @staticmethod
    def __inplacediv(x: np.ndarray , y: np.ndarray )-> None:
        x /= y
    @staticmethod
    def __inplacesqrt(x : np.ndarray)-> None:
        np.sqrt(x , out = x)
    @staticmethod  
    def __inplaceexp(x : np.ndarray)->None:
        np.exp(x , out= x)
    
    @inplace_operation(__inplaceadd)
    def __iadd__(self , y )-> Variable:...
    @inplace_operation(__inplacesub)
    def __isub__(self , y )-> Variable:...
    @inplace_operation(__inplacemul)
    def __imul__(self , y )-> Variable:...
    @inplace_operation(__inplacediv)
    def __itruediv__(self , y )-> Variable:...
    @inplace_operation(__inplacesqrt)
    def sqrt_(self )-> Variable:...
    @inplace_operation(__inplaceexp)
    def exp_(self )-> Variable:...


