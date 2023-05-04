from ml.Variable import Variable
from typing import Any , Tuple


pad = Variable.pad

slice = Variable.__getitem__

broadcast_to = Variable.broadcast_to

transpose = Variable.transpose

flip = Variable.flip

squeeze = Variable.squeeze

expand_dims = Variable.unsqueeze

unsqueeze = expand_dims

def reshape(x : Variable , shape : tuple):
    return x.reshape(*shape)

cast = Variable.astype

argmax = Variable.argmax

argmin = Variable.argmin

argsort = Variable.argsort

clone = Variable.clone

copy = clone

contiguous = Variable.contiguous

split = Variable.split

chunk = split

unstack = Variable.unstack

def concat(tensors : list[Variable] | tuple[Variable ], axis : int = 0 )->Variable:
    return tensors[0].cat(tensors[1:] , axis = axis )

def stack(tensors : list[Variable] | tuple[Variable ], axis : int = 0 )->Variable:
    return tensors[0].stack(tensors[1:] , axis = axis )


def where(mask: Variable , a , b) ->Variable:
    a = a if isinstance(a, Variable) else Variable(a)
    return a.where(mask , b)

def to_tensor(a: Any, requires_grad = None , dtype = None ) ->Variable:
    return Variable(a,requires_grad=requires_grad , dtype = dtype )

def rot90(x : Variable , k : int = 1  , axes : Tuple[int] = (0,1) ):
    """
    same as numpy's https://numpy.org/doc/stable/reference/generated/numpy.rot90.html
    but trackable by the autograd engine
    """
    if axes[0] == axes[1] or abs(axes[0] - axes[1]) == x.ndim \
       or axes[0] >= x.ndim or axes[1] < -(x.ndim):
        raise ValueError(f'invalid axes arguement expected axes to be different and in range of tensor dims \
                         but got dims = {x.ndim} and axes = {axes} ')
    k%=4
    if not k:
        return x
    if k == 2:
        return x.flip(axes[0]).flip(axes[1])
    axes = list(range(0, x.ndim))
    axes[axes[0]] , axes[axes[1]] = axes[axes[1]] , axes[axes[0]]
    if k == 1:
        return x.flip(axes[1]).transpose(*axes)
    else:
        return x.transpose(*axes).flip(axes[1])