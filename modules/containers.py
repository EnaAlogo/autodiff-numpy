import ml 
from modules.layers import Layer , Module


class Sequential(Module):
    def __init__(self , modules : list[Layer] ) ->None:
        super(Sequential,self).__init__()
        self.modules = modules

    def add(self, module: Layer) ->None:
        self.modules.append(module)

    def __call__(self, x) :
        for module in self.modules:
            x = module(x)
        return x
    
    def parameters(self):
        return [p for module in self.modules for p in module.parameters()]

