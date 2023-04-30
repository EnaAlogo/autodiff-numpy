from ml.Variable import Variable


pad = Variable.pad

slice = Variable.__getitem__

broadcast_to = Variable.broadcast_to

transpose = Variable.transpose

flip = Variable.flip

squeeze = Variable.squeeze

expand_dims = Variable.unsqueeze

unsqueeze = expand_dims

reshape = Variable.reshape

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

def to_tensor(a) ->Variable:
    return Variable(a)