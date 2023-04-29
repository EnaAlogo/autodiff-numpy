from ml.Variable import Variable

class Loss: #base class for all losses
     _reductions = {'mean' : Variable.mean , 'sum' : Variable.sum , 'none' : lambda x : x }

     def __init__(self , reduction : str ) -> None  :
        reduction = reduction.lower() 
        if reduction not in ('mean' , 'sum' , 'none'):raise ValueError('only acceptable reduction inputs are mean sum or None')
        self.reduction  = Loss._reductions[reduction]



class MSE(Loss):#mean squared error loss
    """
    mean[ ( Y - y_approx )² ]
    """
    def __init__(self , reduction : str = 'mean') -> None  :
        super(MSE , self).__init__(reduction)

    def __call__(self , y :Variable  , Y :Variable  ) -> Variable : # mean[ ( Y - y_approx)² ]
        shifted = Y - y
        return self.reduction( # optionally reduce the losses of every datapoint in the batch to one number
            (shifted**2).mean(-1)#features axis
        )
    

class CrossEntropy(Loss):
    """
    mean[ - Ylog(y_approx)  ]
    """
    def __init__(self , reduction : str = 'mean' , axis :int = -1 ) -> None  :
        super(CrossEntropy , self).__init__(reduction)
        self.axis = axis
     

    def __call__(self , y : Variable , Y : Variable ) ->Variable  : # mean[ - Ylog(y_approx)  ]
        return self.reduction(
            -( Y * (y+1e-12).log() ).mean(self.axis)
        )
    
class BinaryCrossEntropy(Loss):
    """
    mean[ - Ylog(y_approx) + (1-Y)log(1-y_approx)  ]
    """
    def __init__(self , reduction : str = 'mean' , axis :int = -1 ) -> None  :
        super(BinaryCrossEntropy , self).__init__(reduction)
        self.axis = axis
     

    def __call__(self , y : Variable , Y : Variable ) ->Variable  : 
        return self.reduction(
            -( (Y * (y+1e-12).log() ) + ( (1- Y) * ((1-y) +1e-12).log() )  ).mean(self.axis)
        )


class Huber(Loss):
     """
     loss = 0.5 * x^2                  if |x| <= d
     loss = 0.5 * d^2 + d * (|x| - d)  if |x| > d
     """
     def __init__(self , reduction : str = 'mean', delta:float = 1.0 ) -> None  :
        super(Huber , self).__init__(reduction)
        self.δ = delta
    
     def __call__(self , y : Variable , Y : Variable ) ->Variable  :
        
        x : Variable = Y - y
        abs_x : Variable = x.abs()
        lt : Variable = .5 * (x**2)
        ge : Variable = .5 * (self.δ ** 2) + self.δ * (abs_x - self.δ)

        return self.reduction(lt.where( abs_x <= self.δ , ge ))
     

class CosineSimilarity(Loss):
     """
     loss = -sum[ l2_norm(y_true) * l2_norm(y_pred) ]
     """
     def __init__(self , reduction : str = 'mean', axis :int = -1) -> None  :
        super(CosineSimilarity , self).__init__(reduction)
        self.axis = axis
    
     def __call__(self , y : Variable , Y : Variable ) ->Variable  :
         return self.reduction(
             - (Y.l2_normalize(self.axis) * y.l2_normalize(self.axis)).sum(self.axis)
             )
        