import ml 
from modules import Module
from ml.Variable import Variable
from ml.nn.functional.nn_ops import moments, batch_norm
from typing import Tuple



class Layer(Module):

    def __init__(self):
        super(Layer,self).__init__()

    def __call__(self, x : Variable ) -> Variable :
        if hasattr(self,'build') and not self.built:
                self.build(x)
        self.__call__ = self.call 
        return self.call(x)
    

    
    



class Linear(Layer):
    def __init__(self,
                 units : int ,
                 use_bias : bool = True , 
                 weight_initializer = ml.initializers.GlorotUniform(),
                 bias_initalizer = ml.initializers.Zeros() ):
        super(Linear,self).__init__()
        self.units = units
        self.built = False
        self.use_bias = use_bias
        self.weight_initializer=weight_initializer
        self.bias_initalizer=bias_initalizer

    def call(self , x: Variable) ->Variable:
        y : Variable = x.tensordot( self.w , [ -1 , 0 ] )
        return y if self.bias is None else y + self.bias
    
    def build(self , x):
        shape = x.shape 
        self.w :Variable = self.weight_initializer((shape[-1], self.units))
        self.w.requires_grad = True
        if self.use_bias:
           self.bias :Variable = self.bias_initalizer((self.units)) 
           self.bias.requires_grad = True
        else : self.bias = None
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

    def call(self , y:Variable)->Variable:#should i use bessels correction ?
        mean , variance = moments(y, self.axis , keepdims = True , correction= 0  )
        return batch_norm(y , mean , variance ,self.γ , self.β , self.ε )

    def build(self , x):
        shape = x.shape 
        ax = self.axis
        self.axis = [self.axis if self.axis>=0 else self.axis+x.ndim,] \
            if isinstance(self.axis , int )else [ax if ax>=0 else ax + len(x.shape) for ax  in self.axis]
        w_shape = (shape[-1],) if ax == -1 else \
            [ 1 if i not in self.axis else shape[i] for i in range(len(x.shape)) ]
        if self.center:
           self.β :Variable = self.beta_initializer(w_shape)
           self.β.requires_grad=True
        else : self.center = None
        if self.scale:
          self.γ :Variable = self.gamma_initializer(w_shape) 
          self.γ.requires_grad = True
        else : self.γ = None
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
        reduction_axes = tuple(dim for dim in range(x.ndim) if dim not in self.axis)
        if self.training():
            batch_mean , batch_variance = moments(x , reduction_axes , keepdims= True , correction= 0)
            self.running_mean *= 1-self.μ
            self.running_mean += self.μ * batch_mean.detach().reshape(*self.running_mean.shape) 
            self.running_var *= 1-self.μ
            self.running_var += self.μ * batch_variance.detach().reshape(*self.running_var.shape) 
        else:
            batch_mean,batch_variance = self.running_mean , self.running_var
        
        return batch_norm(x , batch_mean , batch_variance ,self.γ ,self.β , self.ε)

    def build(self , x):
        shape = x.shape 
        ax = self.axis
        self.axis = [self.axis if self.axis>=0 else self.axis+x.ndim,] \
            if isinstance(self.axis , int )else [ax if ax>=0 else ax + len(x.shape) for ax  in self.axis]
        w_shape = (shape[-1],) if ax == -1 else \
            [ 1 if i not in self.axis else shape[i] for i in range(len(x.shape)) ]
        if self.center:
           self.β :Variable = self.beta_initializer(w_shape)
           self.β.requires_grad=True
        else : self.center = None
        if self.scale:
          self.γ :Variable = self.gamma_initializer(w_shape) 
          self.γ.requires_grad = True
        else : self.γ = None
        self.running_mean :Variable = self.running_mean_initializer(w_shape)
        self.running_mean.requires_grad = False
        self.running_var:Variable = self.running_var_initializer(w_shape)
        self.running_var.requires_grad = False
        self.built = True


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

    def call(self , x : Variable) ->Variable:
        raise NotImplementedError('')

    def build(self , x : Variable):
        shape = x.shape 
        ax = self.axis
        self.axis = [self.axis if self.axis>=0 else self.axis+x.ndim,] \
            if isinstance(self.axis , int )else [ax if ax>=0 else ax + len(x.shape) for ax  in self.axis]
        w_shape = (shape[-1],) if ax == -1 else \
            [ 1 if i not in self.axis else shape[i] for i in range(len(x.shape)) ]
        if self.center:
           self.β :Variable = self.beta_initializer(w_shape)
           self.β.requires_grad=True
        else : self.center = None
        if self.scale:
          self.γ :Variable = self.gamma_initializer(w_shape) 
          self.γ.requires_grad = True
        else : self.γ = None
        self.built = True 


class Embedding(Layer):
    def __init__(self, 
                 num_embedd , 
                 embedd_dim,
                 weight_initializer = ml.initializers.Uniform()
                 ):
        super(Embedding,self).__init__()
        self.w = weight_initializer((num_embedd,embedd_dim))
        self.w.requires_grad = True
    
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

        if self.training():
           mask :Variable = ml.random.binomial(x.shape , 1 , 1 - self.rate , requires_grad = False,
                                              dtype= 'float16' ,device = x.device())
           mask *=(1 / (1 - self.rate))
           x = x * mask

        return x 


    

class Conv2D(Layer):
    def __init__(self,
                 filters : int ,
                 kernel_size : Tuple[int] | int,
                 strides : Tuple[int] | int = 1,
                 padding : str | Tuple[int] | int = 'valid',
                 dilations : Tuple[int] | int = 1,
                 groups : int = 1,
                 use_bias : bool = True , 
                 kernel_initializer = ml.initializers.GlorotUniform(),
                 bias_initalizer = ml.initializers.Zeros(),
                 data_format = 'NHWC' ):
        super(Conv2D,self).__init__()
        self.filters = filters
        self.built = False
        self.use_bias = use_bias
        self.kernel_initializer=kernel_initializer
        self.bias_initalizer=bias_initalizer
        self.kernel_size = tuple((kernel_size,kernel_size)) if isinstance(kernel_size , int) else kernel_size
        self.strides = tuple((strides,strides)) if isinstance(strides, int) else strides
        self.dilations = tuple((dilations,dilations)) if isinstance(dilations,int) else dilations
        self.data_format = data_format
        self.groups = groups
        self.padding = padding
    
    def call(self , x: Variable) ->Variable:
        y : Variable = ml.nn.conv2d(x , 
                                    self.kernel ,
                                    self.strides , 
                                    self.padding , 
                                    self.dilations , 
                                    self.data_format)
        return y if self.bias is None else y + self.bias
    
    def build(self , x):
        if self.data_format == 'NHWC':
            if x.shape[-1] % self.groups != 0 : raise ValueError('number of groups must be evenly divisible with features')
            kernel_size = self.kernel_size + (x.shape[-1] // self.groups, self.filters , ) 
            bias_size = (self.filters ,)
        else:  
            if x.shape[1] % self.groups != 0 : raise ValueError('number of groups must be evenly divisible with features')
            kernel_size =  (self.filters , x.shape[1] // self.groups , ) +self.kernel_size
            bias_size = (1,self.filters,1,1)

        self.kernel :Variable = self.kernel_initializer(kernel_size)
        self.kernel.requires_grad = True
        if self.use_bias:
           self.bias :Variable = self.bias_initalizer(bias_size)
           self.bias.requires_grad = True
        else:
            self.bias = None
        self.built = True


class MaxPool2D(Layer):
    def __init__(self , 
                 pool_size : Tuple[int] | int,
                 strides : Tuple[int] | int = 1,
                 padding : str | Tuple[int] | int = 'valid',
                 data_format = 'NHWC') -> None:
        super(MaxPool2D , self).__init__()
        self.pool_size = tuple((pool_size,pool_size)) if isinstance(pool_size , int) else pool_size
        self.strides = tuple((strides,strides)) if isinstance(strides, int) else strides
        self.padding = padding
        self.data_format = data_format
    
    def call(self,x :Variable ) ->Variable:
        return ml.nn.max_pool2d(x , 
                                self.pool_size ,
                                self.strides ,
                                self.padding ,
                                self.data_format)


class AvgPool2D(Layer):
    def __init__(self , 
                 pool_size : Tuple[int] | int,
                 strides : Tuple[int] | int = 1,
                 padding : str | Tuple[int] | int = 'valid',
                 data_format = 'NHWC') -> None:
        super(AvgPool2D , self).__init__()
        self.pool_size = tuple((pool_size,pool_size)) if isinstance(pool_size , int) else pool_size
        self.strides = tuple((strides,strides)) if isinstance(strides, int) else strides
        self.padding = padding
        self.data_format = data_format
    
    def call(self,x :Variable ) ->Variable:
        return ml.nn.avg_pool2d(x , 
                                self.pool_size ,
                                self.strides ,
                                self.padding ,
                                self.data_format)
