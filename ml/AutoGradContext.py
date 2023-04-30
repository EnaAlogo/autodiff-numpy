import traceback

class Context:
    no_grad :bool = False

class stop_gradient:
    #make stop grad be a decorator aswell
    def __init__(self , arg = None)->None:
        self.arg = arg

    def __call__(self , *args , **kwds):
        Context.no_grad = True
        try:
         ret = self.arg(*args,**kwds)
         Context.no_grad = False
         return ret
        except Exception as e :
            Context.no_grad = False
            print(f'exception occured {e}')
            traceback.print_exc()
            return None
        
    
    def __enter__(self):
        Context.no_grad = True

    def __exit__(self , type, value, traceback):
        Context.no_grad = False







