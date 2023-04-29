from ml.Variable import Function , np , Variable , register_gradient
from math import pi , sqrt
from scipy.special import erf

class Relu(Function): 

    def __init__(self , x )->None:
        super(Relu,self).__init__(x)

    def __call__(self , y :np.ndarray )->np.ndarray: # relu(x) = max( x , 0 )
        self.y =  np.maximum(y , 0)
        return self.y
    
    def backward(self  , g :np.ndarray )->np.ndarray:  # relu'(x) = 1 where x> 0 and 0 where x<0
        return g * self.y.astype("bool") # if we cast the output of relu to bool we get 0 where x was <0 and 1 where >0

class Sigmoid(Function):

    def __init__(self , x )->None:
        super(Sigmoid,self).__init__(x) 

    def __call__(self , y :np.ndarray )->np.ndarray: # σ(x) = 1 / 1 + exp(-x)
        self.output = 1. / (1 + np.exp(-y) )
        return self.output
    
    def backward(self  , g :np.ndarray )->np.ndarray:
        ret = self.output * (1-self.output)
        ret *= g
        return ret # σ'(x) = σ(x)(1-σ(x))
    
class Tanh(Function):

    def __init__(self , x )->None:
        super(Tanh,self).__init__(x) 

    def __call__(self , y :np.ndarray )->np.ndarray:# tanh(x) =  exp(x) - exp(-x) / exp(x) + exp(-x)
        self.output = np.tanh(y)
        return self.output

    def backward(self  , g :np.ndarray )->np.ndarray:
        ret = 1- (self.output**2)
        ret *= g
        return ret # tanh'(x) = 1-tanh²(x)

class Identity(Function):
    
    def __init__(self , x )->None:
        super(Identity,self).__init__(x) 

    def __call__(self , y :np.ndarray )->np.ndarray:
        return y
    
    def backward(self  , g :np.ndarray )->np.ndarray:
        return g 

class Celu(Function):
    def __init__(self , x )->None:
        super(Celu,self).__init__(x) 

    def __call__(self , y :np.ndarray , alpha : float = 1.0 )->np.ndarray:
        self.α = alpha
        self.x  = y
        expxdiva :np.ndarray = y / alpha
        self.expxdiva = np.exp(expxdiva , out= expxdiva)
        out = self.expxdiva - 1
        out*= alpha
        return np.maximum(0, y) + np.minimum( 0 , out )

    
    def backward(self  , g :np.ndarray )->np.ndarray:
        self.expxdiva *= self.α
        out =  np.where(self.x > 0 , 1 , self.expxdiva )
        out *= g
        return out
    
class Gelu(Function):
    def __init__(self , x )->None:
        super(Gelu,self).__init__(x) 
    
    invsqrt2 :float = 1.0 / sqrt(2) # kalpha
    kbeta :float = (2.0 /sqrt(pi) ) * invsqrt2 * .5

    def __call__(self , y :np.ndarray)->np.ndarray:
        self.x = y
        self.cdf = y * Gelu.invsqrt2
        self.cdf = erf(self.cdf , out = self.cdf)
        self.cdf += 1.0
        self.cdf *= 0.5
        return self.cdf  * self.x
    
    def backward(self  , g :np.ndarray )->np.ndarray:
       pdf = self.x * self.x
       pdf *= -0.5
       pdf = np.exp(pdf , pdf)
       pdf *= Gelu.kbeta
       pdf *= self.x
       pdf +=  self.cdf
       pdf *= g
       return pdf
