import math
from ml import ml 
from modules.nn_ops import moments, batch_norm

class Module():
    training:bool = True

class Layer(Module):

    def __init__(self):
        super(Layer,self).__init__()

    def __call__(self, x : ml.tensor ) -> ml.tensor :
        if hasattr(self,'build') and not self.built:
                self.build(x)
        return self.call(x)
    
    def parameters(self) ->list[ml.tensor]:
        if hasattr(self,'parameters_'):
            return self.parameters_
        else:
            return []


class Linear(Layer):
    def __init__(self,
                 units : int ,
                 use_bias : bool = True , 
                 weight_initializer = ml.initializers.GlorotUniform(),
                 bias_initalizer = ml.initializers.Ones() ):
        super(Linear,self).__init__()
        self.units = units
        self.built = False
        self.use_bias = use_bias
        self.weight_initializer=weight_initializer
        self.bias_initalizer=bias_initalizer

    @property
    def parameters_(self) ->list[ml.tensor] : 
        params = []
        if self.w : params.append(self.w)
        if self.bias : params.append(self.bias)
        return params
    
    def call(self , x: ml.tensor) ->ml.tensor:
        y : ml.tensor = x.tensordot( self.w , [ -1 , 0 ] )
        return y if self.bias is None else y + self.bias
    
    def build(self , x):
        shape = x.shape 
        self.w :ml.tensor = self.weight_initializer((shape[-1], self.units))
        self.bias :ml.tensor = self.bias_initalizer((self.units)) if self.use_bias else None
        self.built = True


class LayerNorm(Layer):
    def __init__(self, 
                 axis = -1 ,
                 eps : float  = 1e-5,
                 center :bool = True,
                 scale : bool = True,
                 beta_initializer = ml.initializers.Zeros(),
                 gamma_initializer = ml.initializers.Ones()
                 ):
        super(LayerNorm,self).__init__()
        self.axis = axis
        self.ε = eps
        self.center = center
        self.scale= scale
        self.beta_initializer=beta_initializer
        self.gamma_initializer=gamma_initializer
        self.built = False

    def call(self , y:ml.tensor)->ml.tensor:
        mean , variance = moments(y, self.axis , keepdims = True  )
        return batch_norm(y , mean , variance ,self.γ , self.β , self.ε )
    
    @property
    def parameters_(self) ->list[ml.tensor] : 
        params = []
        if self.γ : params.append(self.γ)
        if self.β : params.append(self.β)
        return params

    def build(self , x):
        shape = x.shape 
        axis  = self.axis if isinstance(self.axis , (list,tuple)) else (self.axis,)
        
        w_shape = (shape[-1],) if self.axis == -1 else \
            ( 1 if i not in axis else shape[i] for i in len(shape))

        self.β :ml.tensor = self.beta_initializer(w_shape)\
                if self.center else None
        self.γ :ml.tensor = self.gamma_initializer(w_shape) \
                if self.scale else None
        self.built = True
            
    

class Embedding(Layer):
    def __init__(self, 
                 num_embedd , 
                 embedd_dim,
                 weight_initializer = ml.initializers.Uniform()
                 ):
        super(Embedding,self).__init__()
        self.w = weight_initializer((num_embedd,embedd_dim))
    
    @property
    def parameters_(self):return [self.w]

    def call(self, x : ml.tensor) ->ml.tensor:
        return self.w[x]



class Flatten(Layer):
    def __init__(self):
         super(Flatten,self).__init__()
         
    def call(self, x : ml.tensor) ->ml.tensor:
        return x.reshape( x.shape[0] , -1 )
    

class Lambda(Layer):
    def __init__(self , functor  ):
        super(Lambda , self).__init__()
        self.call = functor


class Dropout(Layer):
    def __init__(self , rate) -> None:
        super(Dropout , self).__init__()
        self.rate = rate
    
    def call(self,x :ml.tensor ) ->ml.tensor:

        if self.training:
           mask :ml.tensor = ml.random.binomial(x.shape , 1 , 1 - self.rate , False,
                                               'float16')
           mask *=(1 / (1 - self.rate))
           x = x * mask

        return x 


    

