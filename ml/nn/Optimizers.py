from ml.Variable import Variable
from ml.AutoGradContext import stop_gradient
from ml.nn import initializers

# using with stop_gradient when making custom optimizers or detaching tensors is mandatory and more performant

class Optimizer:  #base class for all optimizers
    def __init__(self , parameters : list[Variable] )->None:# parameters that are being optimized
        self.parameters = [ Variable(p) if not isinstance(p,Variable) else p for p in parameters ]
    
    def zero_grad(self)->None: #always reset the gradients after every pass 
        for x in self.parameters:
            #im not deleting this cuz someone somewhere may took a shallow copy of it to print it or whatever
            x.grad = None
    

class SGD(Optimizer) : # stochastic gradient descent
    def __init__(self , 
                 parameters : list[Variable] , 
                 learning_rate : float,
                 weight_decay : float = 0 , 
                 momentum : float = 0 , 
                 nesterov : bool = False )->None:
        super(SGD , self).__init__(parameters)
        self.λ = weight_decay
        self.μ = momentum
        self.γ = learning_rate
        self.nesterov = nesterov
        self.b = [initializers.zeros(p.shape) for p in parameters] \
                if momentum else None
    
    def step(self) -> None :

        with stop_gradient(): 

            for i,x in enumerate(self.parameters):
                assert x.grad is not None and x.data.shape == x.grad.shape
                G = x.gradient
    
                if self.λ :
                    G += self.λ * x
                if self.μ :
                    self.b[i] *= self.μ
                    self.b[i] += G
                    G = G + self.μ * self.b[i] if self.nesterov \
                         else self.b[i]
                x -= self.γ * G # wi+1 = wi - α * ▽wLm(w)



class Adam(Optimizer) : 
    def __init__( self ,
                  parameters : list[Variable] ,
                  learning_rate : float,
                  betas :tuple[float,float] = (.9 , .999) , 
                  eps :float = 1e-8 , 
                  weight_decay : float = 0 , 
                  amsgrad : bool = False ) ->None:
        super(Adam , self).__init__(parameters)
        self.λ = weight_decay
        self.ε =  eps
        self.γ = learning_rate
        self.betas = betas
        self.amsgrad = amsgrad #On the Convergence of Adam and Beyond  Sashank J. Reddi, Satyen Kale, Sanjiv Kumar 15 Feb 2018 
        self.m = [initializers.zeros(p.shape) for p in parameters]
        self.v = [initializers.zeros(p.shape) for p in parameters]
        self.v_max = [initializers.zeros(p.shape) for p in parameters]\
                    if amsgrad else None

    def step(self) -> None :

        with stop_gradient(): 

            for i,x in enumerate(self.parameters):
                assert x.grad is not None and x.data.shape == x.grad.shape
                G = x.gradient
                if self.λ :
                    G += self.λ * x
                
                self.m[i] *= self.betas[0] # m <- b0 * m-1 + (1-b0) * ▽w
                self.m[i] += (1- self.betas[0]) * G 

                self.v[i] *= self.betas[1] # v <- b1 * v-1 + (1-b1) * ▽w²
                self.v[i] += (G**2) * (1- self.betas[1])

                m_hat = self.m[i] / ( 1 - self.betas[0] ** (i+1) )
                v_hat = self.v[i] / ( 1 - self.betas[1] ** (i+1) )

                m_hat *= self.γ #scale with learning rate
                
                if self.amsgrad:
                    self.v_max[i] = self.v_max[i].maximum(v_hat)
                    _v_max = self.v_max[i].sqrt()
                    _v_max += self.ε
                    m_hat /= _v_max
                    x -=  m_hat
                else:
                    v_hat = v_hat.sqrt_()#v_hat is a temporary inplace is prefered
                    v_hat += self.ε 
                    m_hat /= _v_max
                    x -=  m_hat
          
