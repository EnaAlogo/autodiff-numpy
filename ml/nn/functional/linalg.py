from ml.Variable import Variable , register_gradient as __register_gradient
from ml.nn.initializers import eye
from ml.autograd.LinalgOps import BatchedMatrixMultiplication , MatrixVectorProduct

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

def matmul(a: Variable , b:Variable ,
           transpose_a:bool = False, 
           transpose_b : bool = False,
           adjoint_a : bool = False,
           adjoing_b : bool = False) -> Variable:
    if adjoint_a:
        a = a.adjoint()
    elif transpose_a:
        a = a.transpose(-2,-1)
    if adjoing_b:
        b= b.adjoint()
    elif transpose_b:
        b = b.transpose(-2,-1)
    return a@b

tensordot = Variable.tensordot