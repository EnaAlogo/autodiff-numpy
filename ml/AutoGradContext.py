
class Context:
    no_grad :bool = False

class stop_gradient:
    def __enter__(self):
        Context.no_grad = True
    def __exit__(self , type, value, traceback):
        Context.no_grad = False






