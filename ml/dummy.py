
class ndarray():... 

def asnumpy(x):
    #unreachable
    raise RuntimeError('you are trying to access cupy methods when cupy is not available on your machine')

def asarray(x):
    #unreachable
    raise RuntimeError('you are trying to access cupy methods when cupy is not available on your machine')

cuda = None