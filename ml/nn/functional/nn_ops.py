import numpy as np
import math
from ml.AutoGradContext import get_backend
from ml.Variable import Variable
from ml.autograd.ArithmeticOps import Add
from ml.nn.initializers import zeros
from typing import Tuple
from ml.nn.functional.ops import concat


############ helpers ############################################################
def _im_indices( C ,  fhei , fwi , stride , oh , ow,
                    dilations  ):
    """
    indices are int tensors and never require gradient so we are free to use raw numpy here, 
    i basically just want the indices theres no actual operation 
    """
    # should i use get_backend here?
    d1 , d2 = dilations
    i0 = np.repeat(np.arange(fhei), fwi)
    i0 = np.tile(i0, C) * d1
    i1 = stride[0] * np.repeat(np.arange(oh), ow)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    s1 = np.tile(np.arange(fwi), fhei ) 
    s1 = np.tile(s1, C)* d2
    j1 = stride[1] * np.tile(np.arange(ow), oh)
    j = s1.reshape(-1, 1) + j1.reshape(1, -1)
    d = np.repeat(np.arange(C), fhei * fwi).reshape(-1, 1)
    return i, j, d

def im2col(X, C, fhei, fwi, stride, pad , oh , ow ,dilation = (1,1)):
    """
    padding , advanced indexing and concatenating are already implemented and will
    be tracked by autograd engine
    """
    x_pad = X.pad(((0,0), (0,0), (pad[0], pad[1]), (pad[2], pad[3]))) if sum(pad) != 0  else X
    i, j, d = _im_indices(C, fhei, fwi, stride, oh ,ow , dilation)
    cols = x_pad[:, d, i, j]
    cols = concat(cols, axis=-1)
    return cols

def col2im(DX : Variable , N:int , C:int, H:int , W:int,  fhei, fwi, stride, pad , oh , ow ,dilation = (1,1)):
    """
    padding , advanced indexing and concatenating are already implemented and will
    be tracked by autograd engine
    """
    i, j, d = _im_indices(C, fhei, fwi, stride, H , W , dilation)
    im = zeros((N,C,oh + pad[0] + pad[1],ow +pad[2] + pad[3] ) ,device = DX.device())
    cols_reshaped = DX.reshape(C * fhei * fwi, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    """ 
    warning!!! massive hack very bad practice ->
    normally inplace operations should not be possible because mutating 
    a tensor can cause problems when calculating gradients in the backward pass since
    the stored tensors inside the autograd nodes are of course not deep copies that 
    would be terrible ,however... since the 'im' tensor was just created and theres no 
    connection with any other node AND the addition operation does not store any tensor
    we are 'free' to do this and make sure we connect it with the input for proper back prop
    """
    ret = im.assign(cols_reshaped , (slice(None) ,d,i,j) ,get_backend(im).add )

    return ret if sum(pad) == 0 else ret[:,:,pad[0]:-pad[1],pad[2]:-pad[3]]

def _calculate_padding_same(in_height , in_width, filter_height ,filter_width, strides):
    """ 
    https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2 
    """
    stride_height , stride_width = strides
    if (in_height % strides[0] == 0):
       pad_along_height = max(filter_height - stride_height, 0)
    else:
       pad_along_height = max(filter_height - (in_height % stride_height), 0)
    if (in_width % strides[1] == 0):
       pad_along_width = max(filter_width - stride_width, 0)
    else:
       pad_along_width = max(filter_width - (in_width % stride_width), 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    return pad_top , pad_bottom, pad_left, pad_right 

def _calulcate_padding_and_output_shape(pad , strides ,
                                        in_height , in_width ,
                                        filter_height , filter_width,
                                        dilations = (1,1)
                                        )->Tuple[int]:
    filter_height = filter_height  * dilations[0] - (dilations[0] -1)
    filter_width = filter_width  * dilations[1] - (dilations[1] -1)
    if isinstance(pad , int):
        H = math.ceil( (in_height + 2*pad - filter_height + 1) / strides[0] )
        W = math.ceil( (in_width + 2*pad- filter_width + 1) / strides[1] )
        pad =  tuple ((pad,pad,pad,pad))
    elif isinstance(pad ,str):
        pad = pad.lower()
        if pad == 'valid':
             pad = (0,0,0,0)
             H = math.ceil( (in_height - filter_height + 1) / strides[0] )
             W = math.ceil( (in_width - filter_width + 1) / strides[1] )
        elif pad == 'same':
             pad = _calculate_padding_same(in_height , in_width,filter_height , filter_width, strides)
             H = math.ceil( in_height / strides[0])
             W = math.ceil( in_width / strides[1])
        else: raise ValueError('only VALID , SAME or explicit pad values supported')
    elif isinstance(pad, (list,tuple)):
        if len(pad) != 4 : raise ValueError('explicit padding needs to be in the form of (top , bottom , left ,right) or int')
        H = int( (in_height + pad[0] + pad[1] - filter_height ) / strides[0]) +1
        W = int( (in_width + pad[2] + pad[3]- filter_width ) / strides[1]) +1
    else : raise ValueError(f'padding expected to be string , int , tuple or list but got {pad.__class__}')
    return pad , H , W
        
 

def pool2d( X : Variable , 
           pad :Tuple[int], 
           size :Tuple[int], 
           stride:Tuple[int] ,
           outH :int, 
           outW:int, 
           mode = Variable.max  ,
           NHWC : bool = True):
        """
        these ugly tranposes make me wanna implement my own einops , one day i may,
        basically we make sure the data are arranged in the correct order for the computation
        and then returning the correct format 
        """
        assert mode in (Variable.max , Variable.mean)
        # b h w c -> b c h w
        X = X.transpose(0,-1,1,2) if NHWC else X             
        N, C, _, _ = X.shape
        image_2col = im2col(X , X.shape[1], size[0], size[1], stride,pad,  outH , outW)
        image_2col = image_2col.reshape(C, image_2col.shape[0]//C, -1)
        y = mode(image_2col, axis=1)
        y = tuple(y.split(N,axis=1))
        y = y[0].cat( y[1:] )
        # b c h w -> b h w c  
        y = y.reshape(N, C, outH, outW)
        return y.transpose(0 , 2, -1 , 1) if NHWC else y

def convolve2d(X :Variable ,
               W :Variable ,  
               pad :Tuple[int] , 
               stride :Tuple[int], 
               outH :int, 
               outW :int, 
               dilations : Tuple[int],
               NHWC : bool = True):
        """
        torch mainly uses NCHW and tensorflow NHWC i personally prefer the latter ,
        when features are the second axis the kernel is ( filters , features , size0 , size1 ),
        and when the features are last the kernel is ( size0 , size1 , features , filters ),
        making sure the data are arranged properly is what these ugly tranposes are doing
        """
        X = X.transpose(0 , -1 , 1, 2) if NHWC else X
        w = W.transpose(-1 , 2 , 0 , 1)  if NHWC else W
        N = X.shape[0]
        in_feats = w.shape[1]
        C = w.shape[0]
        image_2col = im2col(X, in_feats , w.shape[-2], w.shape[-1], stride, pad , outH ,outW , dilations)
        kernel_2col = w.reshape(w.shape[0], -1)
        y = kernel_2col.matmul(image_2col) 
        y = tuple(y.split(N , axis=1))
        y = y[0].cat(y[1:] ).reshape(N, C , outH ,outW)
        return y.transpose(0 , 2, -1 , 1)  if NHWC else y

def deconvolve2d(
               X :Variable ,
               W :Variable , 
               pad :Tuple[int] , 
               stride :Tuple[int], 
               outH :int, 
               outW :int, 
               dilations : Tuple[int],
               NHWC : bool = True):
        """
        torch mainly uses NCHW and tensorflow NHWC i personally prefer the latter ,
        when features are the second axis the kernel is ( filters , features , size0 , size1 ),
        and when the features are last the kernel is ( size0 , size1 , features , filters ),
        making sure the data are arranged properly is what these ugly tranposes are doing
        """
        X = X.transpose(0 , -1 , 1, 2) if NHWC else X
        w = W.transpose(-1 , 2 , 0 , 1)  if NHWC else W
        num_filters, _, filter_height, filter_width = w.shape
        dout_reshaped = X.transpose(1, 2, 3, 0).reshape(num_filters, -1)
        dx_cols = w.reshape(num_filters ,-1 ).T.matmul(dout_reshaped)
        y = col2im(dx_cols,X.shape[0],w.shape[1],X.shape[2],X.shape[3],w.shape[-2],w.shape[-1],stride,pad,outH , outW , dilations )
        return y.transpose(0 , 2, -1 , 1)  if NHWC else y


def __validate_conv_args(data_format : str , image_shape , kernel_shape , strides , pad , dilations):
     df = data_format.lower()
     if df == 'nhwc':
          h , w = image_shape[1] , image_shape[2]
          k1 , k2 = kernel_shape[0] , kernel_shape[1]
          nhwc = True
     elif df == 'nchw':
          h , w = image_shape[2] , image_shape[3]
          k1 , k2 = kernel_shape[-2] , kernel_shape[-1]
          nhwc=False
     else : raise ValueError('only NHWC  and NCHW formats supported')
     strides = tuple( (strides,strides,) ) if isinstance(strides , int) else strides
     pad , H , Wd  = _calulcate_padding_and_output_shape(pad , strides , h , w , k1 , k2 , dilations)
     if H<=0 or Wd <=0 : raise ValueError(f'dimension error in conv2d found out height = {H} out width = {Wd}')
     return nhwc , strides , pad , H , Wd

############### usable functions ###################################################################


def conv2d(
            Image : Variable ,
            Kernel : Variable ,
            strides : Tuple[int] | int = (1,1) ,
            pad: str | Tuple[int] | int = 'VALID',
            dilations:int | Tuple[int] = 1,
            data_format:str = 'NHWC'
            ) -> Variable :
     dilations = tuple((dilations,dilations)) if isinstance(dilations,int) else dilations
     nhwc , strides , pad , outH , outW = __validate_conv_args(data_format, Image.shape , Kernel.shape , strides , pad , dilations)
     return convolve2d(Image , Kernel , pad , strides , outH , outW ,dilations , nhwc )


def conv2d_transpose(
            Image : Variable ,
            Kernel : Variable ,
            output_shape : Tuple[int],
            strides : Tuple[int] | int = (1,1) ,
            pad: str | Tuple[int] | int = 'VALID',
            dilations:int | Tuple[int] = 1,
            data_format:str = 'NHWC'
            ) -> Variable :
     dilations = tuple((dilations,dilations)) if isinstance(dilations,int) else dilations
     nhwc , strides , pad , _ , _ = __validate_conv_args(data_format, Image.shape , Kernel.shape , strides , pad , dilations)
     outH ,outW =( output_shape[1] , output_shape[2] ) if nhwc else (output_shape[2] , output_shape[3])
     return deconvolve2d(Image , Kernel , pad , strides , outH , outW ,dilations , nhwc )



def max_pool2d( X :Variable,
                pool_size : int | Tuple[int],
                strides : int | Tuple[int] = (1,1),
                pad : str | Tuple[int] | list[int] | int ='valid',  
                data_format : str = 'NHWC'
                ) -> Variable:
     df = data_format.lower()
     if df == 'nhwc':
          nhwc = True
          h ,w = X.shape[1] , X.shape[2]
     elif df == 'nchw':
          nhwc=False
          h ,w = X.shape[2] , X.shape[3]
     else :raise ValueError('only NHWC  and NCHW formats supported')
     strides = tuple( (strides,strides,) ) if isinstance(strides , int) else strides
     pool_size = tuple((pool_size,pool_size,)) if isinstance(pool_size,int) else pool_size
     pad , outH , outW = _calulcate_padding_and_output_shape(pad , strides , h , w , pool_size[0] , pool_size[1])
     return pool2d(X , pad , pool_size , strides ,outH , outW , mode = Variable.max , NHWC= nhwc)

def avg_pool2d( X :Variable,
                pool_size : int | Tuple[int] ,
                strides : int | Tuple[int] = (1,1),
                pad: str | Tuple[int] | list[int] | int = 'valid'  ,
                data_format : str = 'NHWC'
                ) -> Variable:
     df = data_format.lower()
     if df == 'nhwc':
          nhwc = True
          h ,w = X.shape[1] , X.shape[2]
     elif df == 'nchw':
          nhwc=False
          h ,w = X.shape[2] , X.shape[3]
     else :raise ValueError('only NHWC  and NCHW formats supported')
     strides = tuple( (strides,strides,) ) if isinstance(strides , int) else strides
     pool_size = tuple((pool_size,pool_size,)) if isinstance(pool_size,int) else pool_size
     pad , outH , outW = _calulcate_padding_and_output_shape(pad , strides , h , w , pool_size[0] , pool_size[1])
     return pool2d(X , pad , pool_size , strides ,outH , outW , mode = Variable.mean , NHWC= nhwc)


def moments(x : Variable , axis : list | tuple = -1 , 
            keepdims = False , correction : int = 1 ) -> Tuple[Variable]:
    if isinstance(axis , int):
        axis = (axis,)
    mean : Variable = x.mean(axis , keepdims = True)
    shift = x - mean
    scale =  1.0 / (np.prod( [ x.shape[ax] for ax in axis ] )-correction)
             
    variance = (shift*shift).sum(axis , keepdims) * scale if correction != 0 else (shift*shift).mean(axis,keepdims)

    return mean if keepdims else mean.squeeze(axis) , variance

def batch_norm(
               x : Variable , 
               mean : Variable , 
               variance : Variable , 
               gamma : Variable = None,
               beta : Variable = None ,
               eps : float =  1e-5
               )->Variable:
    inv_std = (variance+eps).rsqrt()
    y = (x - mean) * inv_std
    if gamma is not None:
        y = y * gamma
    if beta is not None:
        y =  y + beta
    return y

