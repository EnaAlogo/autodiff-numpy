from ml.Variable import Variable

class Module():
    training:bool = True
    
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