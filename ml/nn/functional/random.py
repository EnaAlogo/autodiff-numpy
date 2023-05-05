from ml.AutoGradContext import get_backend_from_device
from ml.Variable import Variable , np , math
import functools

def randn(*shape , requires_grad:bool = None , dtype = np.float32 , device = None)-> Variable:
    device = device.upper() if isinstance(device , str) else device
    backend = get_backend_from_device(device)
    return Variable(backend.random.randn(*shape).astype(dtype), requires_grad= requires_grad)

def uniform(shape , low = 0 , high = 1 , requires_grad:bool = None, dtype = np.float32, device = None)-> Variable:
    device = device.upper() if isinstance(device , str) else device
    backend = get_backend_from_device(device)
    return Variable(backend.random.uniform(low,high, np.prod(shape)).reshape(shape).astype(dtype), requires_grad= requires_grad)

def glorot_uniform(*shape , requires_grad :bool= None ,dtype = np.float32, device = None)-> Variable: # wij ~ U [ - 6/ √n , 6/ √n ]
    device = device.upper() if isinstance(device , str) else device
    backend = get_backend_from_device(device)
    limit:float = math.sqrt( 6. / (shape[0] + functools.reduce( lambda y , f : y*f , shape[1:] , 1 )))
    size : int =   functools.reduce( lambda y , f : y*f , shape , 1 )
    return Variable(backend.random.uniform( -limit , limit , size).reshape(shape).astype(dtype) , requires_grad= requires_grad)

def binomial(shape , n , p , requires_grad :bool= None ,dtype = np.float32, device = None):
    device = device.upper() if isinstance(device , str) else device
    backend = get_backend_from_device(device)
    return Variable(backend.random.binomial(n,p,np.prod(shape)).reshape(shape).astype(dtype),requires_grad= requires_grad)

def randint(shape ,low , high, device = None ):
    device = device.upper() if isinstance(device , str) else device
    backend = get_backend_from_device(device)
    return Variable(backend.random.randint(low,high,np.prod(shape)).reshape(shape),requires_grad= False)
