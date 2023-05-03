import ml 
from modules.layers import Layer
from modules import Module
import functools

class Sequential(Module):
    def __init__(self , modules : list[Layer] ) ->None:
        super(Sequential,self).__init__()
        self.modules = modules

    def add(self, module: Layer) ->None:
        self.modules.append(module)

    def __call__(self, x) :
        return functools.reduce( lambda  y , f : f(y) , self.modules , x )

    def parameters(self):
        return [p for module in self.modules for p in module.parameters()]

