from ml.Variable import Variable
from ml.nn.initializers import eye

triu = Variable.triu

tril = Variable.tril

diag = Variable.diag

diagonal = Variable.diagonal

inverse = Variable.inverse

pseudo_inverse = Variable.p_inverse

vdot = Variable.vdot

def dot(a , b) ->Variable:
    a = a if isinstance(a , Variable) else Variable(a)
    return a.dot(b)

trace = Variable.trace

identity = eye

def matmul(a: Variable , b:Variable , transpose_b : bool = False) -> Variable:
    b = b if not transpose_b else b.T
    return a.matmul(b)

tensordot = Variable.tensordot