from ml.Variable import Function
from ml.Variable import np



######################## Binary #####################################################################

class Add(Function):

    def __init__(self , x ,y ) ->None:
        super(Add , self ).__init__( x, y)

    def __call__(self , x : np.ndarray , y : np.ndarray ) -> np.ndarray :    
        return x + y
    
    def backward(self , g :  np.ndarray ) ->  np.ndarray:
        return Function.reverse_broadcast(self.parents[0].shape, g ) if self.needs_grad(0)  else None,\
               Function.reverse_broadcast(self.parents[1].shape, g ) if self.needs_grad(1) else None

class Subtract(Function):

    def __init__(self , x ,y ) ->None:
        super(Subtract , self ).__init__( x, y)

    def __call__(self , x : np.ndarray , y : np.ndarray ) -> np.ndarray :    
        return x - y
    
    def backward(self , g :  np.ndarray ) ->  np.ndarray:
        return Function.reverse_broadcast(self.parents[0].shape, g ) if self.needs_grad(0)  else None,\
               - Function.reverse_broadcast(self.parents[1].shape, g ) if self.needs_grad(1) else None
    
class Multiply(Function):

    def __init__(self , x ,y ) ->None:
        super(Multiply , self ).__init__( x, y)

    def __call__(self , x : np.ndarray , y : np.ndarray) -> np.ndarray :
        self.x = x 
        self.y = y    
        return x * y
    
    def backward(self , g :  np.ndarray ) ->  np.ndarray:
        return Function.reverse_broadcast(self.x.shape, g * self.y ) if self.needs_grad(0)  else None,\
               Function.reverse_broadcast(self.y.shape, g * self.x ) if self.needs_grad(1) else None
    
class Divide(Function):

    def __init__(self , x ,y ) ->None:
        super(Divide , self ).__init__( x, y)

    def __call__(self , x : np.ndarray , y : np.ndarray) -> np.ndarray :  
        self.x = x 
        self.y = y   
        return x / y
    
    def backward(self , g :  np.ndarray ) ->  np.ndarray:
        if self.needs_grad(1):
            dy = -g
            dy *= self.x
            dy /= self.y
            dy /= self.y
            dy = Function.reverse_broadcast(self.y.shape ,dy)
        else :dy = None
        return Function.reverse_broadcast(self.x.shape, g / self.y ) if self.needs_grad(0)  else None,\
               dy

class Power (Function):

    def __init__(self, x , y  ) -> None:
        super(Power , self).__init__(x , y )
    
    def __call__(self , x : np.ndarray , y : np.ndarray | int | float ) -> np.ndarray :  
        self.y = y
        self.x = x
        self.z = x ** y
        return self.z
    
    def backward(self , g :  np.ndarray ) ->  np.ndarray:
        if self.needs_grad(0):
            dx = self.x ** (self.y-1)
            dx *= self.y
            dx *= g
            dx = Function.reverse_broadcast(self.x.shape , dx)
        else : dx = None
        if self.needs_grad(1):
            dy = self.z * np.log(self.x)#this cannot avoid the allocations cuz broadcasting
            dy *= g
            dy = Function.reverse_broadcast(self.y.shape , dy)
        else : dy = None
        return dx , dy


class _MinimumMaximum(Function):
    def __call__(self , x : np.ndarray , y : np.ndarray) -> np.ndarray :    
        ret = self.op(x,y)
        if self.requires_grad:
            self.mask = self.mask_fn(x,y)
        return ret 
    
    def backward(self , g :  np.ndarray ) ->  np.ndarray:
        return Function.reverse_broadcast(self.parents[0].shape , g * self.mask
                                          ) if self.needs_grad(0) else None,\
               Function.reverse_broadcast(self.parents[1].shape , g * ~self.mask
                                          ) if self.needs_grad(1) else None,                          

class Maximum(_MinimumMaximum):
        def __init__(self , x ,y ) ->None:
            super(Maximum , self ).__init__( x, y)
        
        def op(self , x , y):
            return np.maximum(x,y)
        def mask_fn(self,  x , y):
            return x>y
        
class Minimum(_MinimumMaximum):
        def __init__(self , x ,y ) ->None:
            super(Minimum , self ).__init__( x, y)
        
        def op(self , x , y):
            return np.minimum(x,y)
        def mask_fn(self,  x , y):
            return x<y

################### Unary ##############################################################################

class Logarithm(Function):

    def __init__(self, x ) -> None:
        super(Logarithm , self).__init__(x)
    
    def __call__(self , x : np.ndarray ) -> np.ndarray :  
        self.x = x
        return np.log(x)
    
    def backward(self , g :  np.ndarray ) ->  np.ndarray:
        return g / self.x
    
class Exp(Function):

    def __init__(self, x ) -> None:
        super(Exp , self).__init__(x)
    
    def __call__(self , x : np.ndarray ) -> np.ndarray :  
        self.y = np.exp(x)
        return self.y
    
    def backward(self , g :  np.ndarray ) ->  np.ndarray:
        return g * self.y

class SquareRoot(Function):

    def __init__(self, x ) -> None:
        super(SquareRoot , self).__init__(x)
    
    def __call__(self , x : np.ndarray ) -> np.ndarray :  
        self.y = np.sqrt(x)
        return self.y
    
    def backward(self , g :  np.ndarray ) ->  np.ndarray:
        ret = 2* self.y
        ret = np.reciprocal(ret ,out =  ret)
        ret *= g
        return ret


    
class AbsoluteValue(Function):
      def __init__(self, x ) -> None:
        super(AbsoluteValue , self).__init__(x)
    
      def __call__(self , x : np.ndarray ) -> np.ndarray : 
        self.x = x 
        return np.abs(x)
    
      def backward(self , g :  np.ndarray ) ->  np.ndarray:
         ret = np.sign(self.x)
         ret *= g
         return ret
      
class Sine(Function):
      def __init__(self, x ) -> None:
        super(Sine , self).__init__(x)
    
      def __call__(self , x : np.ndarray ) -> np.ndarray : 
        self.x = x 
        return np.sin(x)
    
      def backward(self , g :  np.ndarray ) ->  np.ndarray:
         ret = np.cos(self.x)
         ret *= g
         return ret

class Cosine(Function):
      def __init__(self, x ) -> None:
        super(Cosine , self).__init__(x)
    
      def __call__(self , x : np.ndarray ) -> np.ndarray : 
        self.x = x 
        return np.cos(x)
    
      def backward(self , g :  np.ndarray ) ->  np.ndarray:
         ret = - g
         ret *= np.sin(self.x)
         return ret

