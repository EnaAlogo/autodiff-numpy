from ml.Variable import Variable , np , math

def randn(*shape , requires_grad:bool = None , dtype = np.float32)-> Variable:
    return Variable(np.random.randn(*shape).astype(dtype), requires_grad= requires_grad)

def uniform(shape , low = 0 , high = 1 , requires_grad:bool = None, dtype = np.float32)-> Variable:
    return Variable(np.random.uniform(low,high, np.prod(shape)).reshape(shape).astype(dtype), requires_grad= requires_grad)

def glorot_uniform(*shape , requires_grad :bool= None ,dtype = np.float32)-> Variable: # wij ~ U [ - 6/ √n , 6/ √n ]
    limit:float = math.sqrt( 6. / (shape[0] + np.prod(shape[1:])))
    size : int =  np.prod(shape)
    return Variable(np.random.uniform( -limit , limit , size).reshape(shape).astype(dtype) , requires_grad= requires_grad)

def binomial(shape , n , p , requires_grad :bool= None ,dtype = np.float32):
    return Variable(np.random.binomial(n,p,np.prod(shape)).reshape(shape).astype(dtype),requires_grad= requires_grad)
