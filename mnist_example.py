from typing import Any
import numpy as np
from ml import ml
from tensorflow.keras.datasets import mnist



#making a custom layer that will perfom xAT + b transformation
class Linear:
    def __init__(self,
                 units : int , input_shape : tuple ,
                 use_bias : bool = True , 
                 weight_initializer = ml.initializers.GlorotUniform(),
                 bias_initalizer = ml.initializers.Ones() ):
        self.units = units
        self.w :ml.tensor = weight_initializer((input_shape[-1], units))
        self.bias :ml.tensor = bias_initalizer((units)) if use_bias else None

    @property
    def parameters(self) ->list[ml.tensor] : return [self.w , self.bias] if self.bias \
                                             is not None else [self.w]
    #gradients are automatically tracked by the '' engine ''
    def __call__(self , x: ml.tensor) ->ml.tensor:
        y : ml.tensor = x @ self.w
        return y if self.bias is None else y + self.bias
    
    def compute_output_shape(self , x: tuple) ->tuple:
        return  x[:-1] + (self.units,)  
    
#flatten a (N , d0 ,..., dn) tensor where N is the batch to (N , prod(d0...dn) ) 
class Flatten:
    def __call__(self , x:ml.tensor):
        return x.reshape(x.shape[0] , -1 )
    
    def compute_output_shape(self , x: tuple) ->tuple :
        return x[0] , np.prod(x[1:])

# freeze randomly selected weights to avoid overfiting
def dropout(x : ml.tensor  , rate : float ) ->ml.tensor:
     mask = ml.random.binomial(x.shape , 1 , 1 - rate , requires_grad= False )
     mask *= (1 / (1 - rate))
     return x * mask 
          
    
#encode integer labels to a one hot vector eg for label 2 with [0,4) classes -> [ 0 0 1 0 ]
def one_hot_encode(classes : list[int] , data : ml.tensor | np.ndarray) ->ml.tensor :
    if isinstance(data , ml.tensor): data = data.data

    assert data.ndim <= 2
    one_hot = ml.tensor([ [0 if i != g else 1 for i in range(classes.__len__() )] for g  in  data.ravel()  ],
                        requires_grad= False)
    return one_hot

#split the data into batches
def _batch(y , t , batch_size) :
        steps = y.shape[0] // batch_size
        for i in range(steps):
            ptr : int = i * batch_size
            yield ml.tensor(y[ptr:
                    ptr+batch_size,...],requires_grad=False),\
                  ml.tensor(t[ptr:
                    ptr+batch_size,...],requires_grad=False)

def main():
    #loading the dataset
    (x_train , y_train) , (x_test , y_test) = mnist.load_data()
    #hyper params
    lr = 1e-3
    epochs = 5
    batch_size = 32
    input_shape = (batch_size, ) + x_train.shape[1:]
    #making the model
    layer_0 = Linear(32 , input_shape)
    output_shape0 = layer_0.compute_output_shape(input_shape)
    layer_1 = Linear(16 , output_shape0 )
    output_shape1 = layer_1.compute_output_shape(output_shape0)
    layer_2 = Flatten()
    output_shape2 = layer_2.compute_output_shape(output_shape1)

    classifier = Linear(10 , output_shape2)

    J :ml.losses.Loss =  ml.losses.CrossEntropy()
    
    #pass the parameters to the optimizer
    opt :ml.optimizers.Optimizer =  ml.optimizers.Adam(
        layer_0.parameters + layer_1.parameters + classifier.parameters,
        lr , amsgrad= True
    )

    def forward(x , training = False ):
            y = layer_0(x)
            y = ml.activations.tanh(y)
            y = layer_1(y)
            y = ml.activations.celu(y)
            y = layer_2(y)
            if training: 
               y = dropout(y , .5 )
            y = classifier(y)
            y = ml.activations.softmax(y)
            return y
    
    goals = list(range(0,10))
    for e in range(epochs): #training loop
        for (datapoints , labels) in _batch(x_train  , y_train ,batch_size):
            #forward pass
            y = forward(datapoints , training= True)
            
            #calculate metrics and losses
            loss = J(y , one_hot_encode(goals , labels))
            accuracy = (y.argmax(-1) == labels).mean()
            print(f'epoch : {e} ,train_loss : {"%.4f" % loss.item()} , train_acc : {"%.2f" % (accuracy.item() * 100)} % ',
                  end = '\r')
            
            #backward pass
            loss.backward()

            #optimize the model and reset gradients
            opt.step()
            opt.zero_grad()
        
        #validate on test split but make sure you dont fit
        with ml.stop_gradient():
            y = forward(ml.tensor(x_test,requires_grad=False))

            loss = J(y , one_hot_encode(goals , y_test))
            accuracy = (y.argmax(-1) == y_test).mean()
            print(f'\nvalidation : val_loss : {"%.4f" % loss.item()} , val_acc : {"%.2f" % (accuracy.item() * 100)} %')

    print('testing : ')
    with ml.stop_gradient():
        y = forward(ml.tensor(x_test,requires_grad=False))
        loss = J(y , one_hot_encode(goals , y_test))
        accuracy = (y.argmax(-1) == y_test).mean()
        print(f'validation : val_loss : {"%.4f" % loss.item()} , val_acc : {"%.2f" % (accuracy.item() * 100)} %')


if __name__ == '__main__' : main()
    
