import ml 
from ml.Variable import Variable
from ml.nn.functional.nn_ops import moments, batch_norm

class Module():
    training:bool = True

class Layer(Module):

    def __init__(self):
        super(Layer,self).__init__()

    def __call__(self, x : Variable ) -> Variable :
        if hasattr(self,'build') and not self.built:
                self.build(x)
        self.__call__ = self.call 
        return self.call(x)
    
    def parameters(self) ->list[Variable]:
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
    def parameters_(self) ->list[Variable] : 
        params = []
        if self.w : params.append(self.w)
        if self.bias : params.append(self.bias)
        return params
    
    def call(self , x: Variable) ->Variable:
        y : Variable = x.tensordot( self.w , [ -1 , 0 ] )
        return y if self.bias is None else y + self.bias
    
    def build(self , x):
        shape = x.shape 
        self.w :Variable = self.weight_initializer((shape[-1], self.units))
        self.bias :Variable = self.bias_initalizer((self.units)) if self.use_bias else None
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

    def call(self , y:Variable)->Variable:
        mean , variance = moments(y, self.axis , keepdims = True  )
        return batch_norm(y , mean , variance ,self.γ , self.β , self.ε )
    
    @property
    def parameters_(self) ->list[Variable] : 
        params = []
        if self.γ : params.append(self.γ)
        if self.β : params.append(self.β)
        return params

    def build(self , x):
        shape = x.shape 
        axis  = self.axis if isinstance(self.axis , (list,tuple)) else (self.axis,)
        
        w_shape = (shape[-1],) if self.axis == -1 else \
            ( 1 if i not in axis else shape[i] for i in len(shape))

        self.β :Variable = self.beta_initializer(w_shape)\
                if self.center else None
        self.γ :Variable = self.gamma_initializer(w_shape) \
                if self.scale else None
        self.built = True
            
    

class BatchNorm(Layer):
    def __init__(self, 
                 axis = -1 ,
                 momemtun = .99,
                 eps : float  = 1e-5,
                 center :bool = True,
                 scale : bool = True,
                 beta_initializer = ml.initializers.Zeros(),
                 gamma_initializer = ml.initializers.Ones(),
                 running_mean_initializer = ml.initializers.Zeros(),
                 running_var_initializer = ml.initializers.Ones()
                 ):
        super(BatchNorm,self).__init__()
        self.axis = axis
        self.ε = eps
        self.center = center
        self.scale= scale
        self.beta_initializer=beta_initializer
        self.gamma_initializer=gamma_initializer
        self.running_var_initializer = running_var_initializer
        self.running_mean_initializer = running_mean_initializer
        self.μ = momemtun
        self.built = False

    def call(self , x:Variable)->Variable:
        if self.training:
            batch_mean , batch_variance = moments(x , self.axis , keepdims= True)
            self.running_mean *= 1-self.μ
            self.running_mean += self.μ * batch_mean.detach()
            self.running_var *= 1-self.μ
            self.running_var += self.μ * batch_variance.detach()
        else:
            batch_mean,batch_variance = self.running_mean , self.running_var
        
        return batch_norm(x , batch_mean , batch_variance ,self.γ ,self.β , self.ε)

    @property
    def parameters_(self) ->list[Variable] : 
        params = []
        if self.γ : params.append(self.γ)
        if self.β : params.append(self.β)
        return params

    def build(self , x):
        shape = x.shape 
        axis  = self.axis if isinstance(self.axis , (list,tuple)) else (self.axis,)
        w_shape = (shape[-1],) if self.axis == -1 else \
            ( 1 if i not in axis else shape[i] for i in len(shape))
        self.β :Variable = self.beta_initializer(w_shape)\
                if self.center else None
        self.γ :Variable = self.gamma_initializer(w_shape) \
                if self.scale else None
        self.running_mean :Variable = self.running_mean_initializer(w_shape)
        self.running_var:Variable = self.running_var_initializer(w_shape)
        self.built = True

### TODO this ###############
class GroupNorm(Layer):
    def __init__ (self,
                  groups=32,
                  axis=-1,
                  epsilon=0.001,
                  center=True,
                  scale=True,
                  beta_initializer=ml.initializers.Zeros(),
                  gamma_initializer= ml.initializers.Ones()
                 ):
        self.groups = groups
        self.axis = axis
        self.ε = epsilon
        self.center = center 
        self.scale = scale
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer
              

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

    def call(self, x : Variable) ->Variable:
        return self.w[x]



class Flatten(Layer):
    def __init__(self):
         super(Flatten,self).__init__()
         
    def call(self, x : Variable) ->Variable:
        return x.reshape( x.shape[0] , -1 )
    

class Lambda(Layer):
    def __init__(self , functor  ):
        super(Lambda , self).__init__()
        self.call = functor


class Dropout(Layer):
    def __init__(self , rate) -> None:
        super(Dropout , self).__init__()
        self.rate = rate
    
    def call(self,x :Variable ) ->Variable:

        if self.training:
           mask :Variable = ml.random.binomial(x.shape , 1 , 1 - self.rate , False,
                                               'float16')
           mask *=(1 / (1 - self.rate))
           x = x * mask

        return x 


    

