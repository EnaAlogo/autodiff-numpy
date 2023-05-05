import ml
import numpy as np
from tensorflow.keras.datasets import cifar10
from modules import layers
from modules.containers import Sequential


@ml.stop_gradient #the dataset doesnt require gradient anyways but why not
def normalize(data : ml.tensor):
     #normalize
     data = data.astype('float32') / 255.0
     #subtract pixel mean
     mean = data.mean(axis = 0)
     data -= mean
     return data

def to_sparse(numpy):#invert one hot transformation
     return numpy.argmax(-1)
 
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
            yield y[ptr:
                    ptr+batch_size,...],\
                  t[ptr:
                    ptr+batch_size,...]  
from einops import rearrange
def main():
     (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # x_train  =  rearrange(x_train , 'n h w c -> n c h w')
     #x_test  =  rearrange(x_test , 'n h w c -> n c h w')

     model = Sequential([ 
          layers.Conv2D(16 , 3 , padding='same' ),
          layers.BatchNorm(eps=1e-3),
          layers.Lambda( ml.activations.relu ),
          layers.Conv2D(20 , 3 , padding='same' ),
          layers.BatchNorm(eps=1e-3),
          layers.Lambda( ml.activations.relu ),
          layers.MaxPool2D(pool_size = 2 ),
          layers.Flatten(),
          layers.Dropout(rate=.5),
          layers.Linear(units = 10),
          layers.Lambda( ml.activations.softmax )
     ])
#build ,im probably gonna make the way you build params nicer and more efficient in the future
     model.set_training(False)
     with ml.stop_gradient():
          model(ml.tensor(x_test[:1]))
     model.set_training(True)
     
     device = 'cuda' if ml.cuda_is_available() else 'cpu'
     #move the data (im only taking 1/5 cuz it will literally blow my computer out of existance)
     x_train, y_train, x_test, y_test  = ml.tensor(x_train[:10000],requires_grad=False,device=device),\
      ml.tensor(y_train[:10000],requires_grad=False,device=device), ml.tensor(x_test[:1000],requires_grad=False,device=device),\
       ml.tensor(y_test[:1000],requires_grad=False,device=device) 
     #move the model's parameters to cuda
     model.cuda()

     #set hyper params
     loss = ml.losses.CrossEntropy()
     opt = ml.optimizers.Adam(
          model.parameters() , 
          learning_rate= 1e-3 ,
          weight_decay= .001 #do i really need a seperate AdamW for this? i prefer it to just be like this
          )
     batch_size = 64 
     epochs = 7
     
     classes = list(range(10))
     x_train = normalize(x_train) #normalize integer pixels to [-1,1] floats

     """
     for now im not supporting the more efficient sparse crossentropy but i have written a custom kernel for it
     and since cupy allows me to use custom kernels i may do it one day and then use cython for the numpy version or smth
     """
     labels = one_hot_encode(classes , y_train).to(device)
     for e in range(epochs):
         for x,y in _batch(x_train ,labels , batch_size) :
              #forward pass
              pred = model(x)
              #losses and metrics
              l = loss(pred , y)
              #argmax cupy is not working on my pc for some reasson
              acc = (pred.numpy().argmax(-1) == to_sparse(y.numpy()) ).mean()
              print(f'epoch : {e} ,train_loss : {"%.4f" % l.item()} , train_acc : {"%.4f" % acc.item()}',end = '\r' )

              #backward pass
              l.backward()
              
              #optimize
              opt.step()
              opt.zero_grad()
         print('')

     print('testing : ')
     
     test_labels = one_hot_encode(classes , y_test).to(device) 
     x_test = normalize(x_test)
     #test
     model.set_training(False) #makes sure dropout is not applied and batch_norm will use its internal moments
     with ml.stop_gradient():
          pred = model(x_test) 
          l = loss(pred ,test_labels)
          acc = (pred.numpy().argmax(-1) == to_sparse(test_labels.numpy())).mean()
          print(f'loss : {l.item()} , acc : {acc.item()}')
    
     

if __name__ == '__main__':main()
