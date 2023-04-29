from ml.Variable import Variable


def add(a , b):
    a = a if isinstance(a, Variable) else Variable(a)
    return a+b
def subtract(a , b):
    a = a if isinstance(a, Variable) else Variable(a)
    return a-b
def multiply(a , b):
    a = a if isinstance(a, Variable) else Variable(a)
    return a * b
def divide(a , b):
    a = a if isinstance(a, Variable) else Variable(a)
    return a / b

sin = Variable.sin

cos = Variable.cos

def maximum(a , b):
    a = a if isinstance(a, Variable) else Variable(a)
    return a.maximum(b)
def minimum(a , b):
    a = a if isinstance(a, Variable) else Variable(a)
    return a.minimum(b)

sqrt = Variable.sqrt

rsqrt = Variable.rsqrt

log = Variable.log

exp = Variable.exp

abs = Variable.abs

l2_norm = Variable.l2_normalize

def outer(a , b):
    a = a if isinstance(a, Variable) else Variable(a)
    return a.outer_product(b)

def inner(a , b):
    a = a if isinstance(a, Variable) else Variable(a)
    return a.inner_product(b)

clamp = Variable.clamp

clip = clamp

def power( a , b):
    a = a if isinstance(a, Variable) else Variable(a)
    return a**b