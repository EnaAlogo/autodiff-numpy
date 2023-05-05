from ml.AutoGradContext import stop_gradient , np ,cupy , cuda_is_available , Device
from ml.Variable import Variable

class Module():
    __training:bool = True
    
    # not sure if this works all the time and its painfully slow but its nice 
    def parameters(self) ->list[Variable]:
        params = []
        for attr in dir(self):
            attr = getattr(self, attr)
            if isinstance(attr , Variable) and attr.requires_grad == True:
                params.append(attr)
            elif isinstance(attr , Module):
                if hasattr(attr , 'parameters'):
                    params = params + attr.parameters()
        return params
    
    def cuda(self) ->None :
        if not cuda_is_available():
            raise RuntimeError('cuda is not available on this machine')
        for attr in dir(self):
            attr = getattr(self, attr)
            if isinstance(attr , Variable):
                Variable.to_(attr,Device.CUDA)
            elif isinstance(attr , Module):
                Module.cuda(attr)


    def training(self) ->bool: 
        return Module.__training
 
    def set_training(self, val : bool) -> None :
        if not isinstance(val , bool):
            raise ValueError(f'bool expected got {val.__class__}')
        Module.__training = val