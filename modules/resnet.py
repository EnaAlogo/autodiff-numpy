import ml
from ml.Variable import Variable
from modules import layers
from modules.containers import Sequential

class Block(layers.Layer):
    expansion = 1
    def __init__(self ,
                 planes :int,
                 stride :int = 1,
                 data_format = 'NHWC'
                  ):
        super(Block , self).__init__()
        self.axis = -1 if data_format == 'NHWC' else 1
        self.conv1 = layers.Conv2D(planes ,3 , stride , padding = 1 ,data_format=data_format ,use_bias= False)
        self.nrm1 = layers.BatchNorm(self.axis)
        self.conv2 = layers.Conv2D(planes , 3 , 1 , padding  = 1 , data_format=data_format, use_bias= False)
        self.nrm2 = layers.BatchNorm(self.axis)
        self.downsample :Sequential = None
        self.stride = stride
        self.planes = planes
        self.data_format= data_format
        self.built = False

    def call(self , inputs):
        identity = inputs
        x = self.nrm1(self.conv1(inputs))
        x = ml.activations.relu(x)
        x = self.nrm2(self.conv2(x))
        if self.downsample is not None:
            identity = self.downsample(x)
        x = x + identity
        x = ml.activations.relu(x)
        return x


    def build(self , x): #at this point the inner layers are not yet built but they will be when we call 
        in_features = x.shape[-1] if self.data_format == 'NHWC' else x.shape[1]
        if self.stride != 1 or in_features != self.expansion * self.planes:
            self.downsample = Sequential(
                [
                    layers.Conv2D(self.expansion*self.planes , 1 , self.stride , use_bias= False),
                    layers.BatchNorm(self.axis)
                ]
            )
        self.built = True
       
        

class Bottleneck(layers.Layer):
    expansion = 4
    def __init__(self ,
                 planes :int,
                 stride :int = 1,
                 data_format = 'NHWC'
                  ):
        super(Bottleneck , self).__init__()
        axis = -1 if data_format == 'NHWC' else 1
        self.axis = axis
        self.conv1 = layers.Conv2D(planes ,3 , stride , padding = 1 ,data_format=data_format ,use_bias= False)
        self.nrm1 = layers.BatchNorm(axis)
        self.conv2 = layers.Conv2D(planes , 3 , 1 , padding  = 1,data_format=data_format, use_bias= False)
        self.nrm2 = layers.BatchNorm(axis)
        self.conv3 = layers.Conv2D(self.expansion*planes , 1 , use_bias= False)
        self.nrm3 = layers.BatchNorm(axis)
        self.downsample :Sequential = None
        self.stride = stride
        self.planes = planes
        self.data_format= data_format
        self.built = False
 
    def call(self , inputs):
        identity = inputs
        x = self.nrm1(self.conv1(inputs))
        x = ml.activations.relu(x)
        x = self.nrm2(self.conv2(x))
        x = self.nrm3(self.conv3(x))
        if self.downsample is not None:
            identity = self.downsample(x)
        x = x + identity
        x = ml.activations.relu(x)
        return x


    def build(self , x): #at this point the inner layers are not yet built but they will be when we call 
        in_features = x.shape[-1] if self.data_format == 'NHWC' else x.shape[1]
        if self.stride != 1 or in_features != self.expansion * self.planes:
            self.downsample = Sequential(
                [
                    layers.Conv2D(self.planes*self.expansion , 1 , self.stride , use_bias= False),
                    layers.BatchNorm(self.axis)
                ]
            )
        self.built = True
    


