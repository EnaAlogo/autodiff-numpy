import ml
import numpy as np
from tensorflow.keras.datasets import mnist
from modules import layers
from modules.containers import Sequential


     
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
     (x_train, y_train), (x_test, y_test) = mnist.load_data()

     model = Sequential([
          layers.Linear(32 , use_bias= False),
          layers.Lambda( ml.activations.tanh ),
          layers.Linear(16),
          layers.Lambda( ml.activations.celu ),
          layers.Flatten(),
          layers.Dropout(.5),
          layers.Linear(10),
          layers.Lambda( ml.activations.softmax )
     ])

#build
     with ml.stop_gradient():
          model(ml.to_tensor(x_test,requires_grad=False))


     loss = ml.losses.CrossEntropy()
     opt = ml.optimizers.Adam(model.parameters() , learning_rate= 2e-3 , amsgrad=True )
     batch_size = 32 
     epochs = 5
     
     classes = list(range(10))
     for e in range(epochs):
         for x,y in _batch(x_train ,y_train , batch_size):
              #forward pass
              pred = model(x)
              #losses and metrics
              l = loss(pred , one_hot_encode(classes , y))

              acc = (pred.argmax(-1) == y).mean()

              print(f'epoch : {e} ,train_loss : {"%.4f" % l.item()} , train_acc : {"%.2f" % (acc.item() * 100)} % ',
                  end = '\r')
              #backward pass
              l.backward()
              
              #optimize
              opt.step()
              opt.zero_grad()
         print('')
     print('testing : ')
     #test
     model.training = False #makes sure dropout is not applied 
     with ml.stop_gradient():
          pred = model( ml.to_tensor(x_test,requires_grad=False))
          l = loss(pred , one_hot_encode(classes , y_test))
          acc = (pred.argmax(-1) == y_test).mean()
          print(f'loss : {l.item()} , acc : {acc.item()}')
    
     

if __name__ == '__main__':main()
