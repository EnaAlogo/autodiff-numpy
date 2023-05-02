import numpy as np
import math
from ml.Variable import Variable
from typing import Tuple
from ml.nn.functional.ops import concat


############ helpers ############################################################
def _im2col_indices(X_shape, fhei , fwi , stride , oh , ow):
    """
    indices are int tensors and never require gradient so we are free to use raw numpy here, 
    i basically just want the indices theres no actual operation 
    """
    _, C, H, W = X_shape
    i0 = np.repeat(np.arange(fhei), fwi)
    i0 = np.tile(i0, C)
    i1 = stride[0] * np.repeat(np.arange(oh), ow)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    s1 = np.tile(np.arange(fwi), fhei)
    s1 = np.tile(s1, C)
    j1 = stride[1] * np.tile(np.arange(ow), oh)
    j = s1.reshape(-1, 1) + j1.reshape(1, -1)
    d = np.repeat(np.arange(C), fhei * fwi).reshape(-1, 1)
    return i, j, d

def _im2col(X, fhei, fwi, stride, pad , oh , ow):
    """
    padding , advanced indexing and concatenating are already implemented and will
    be tracked by autograd engine
    """
    x_pad = X.pad(((0,0), (0,0), (pad[0], pad[1]), (pad[2], pad[3])))
    i, j, d = _im2col_indices(X.shape, fhei, fwi, stride, oh ,ow)
    cols = x_pad[:, d, i, j]
    cols = concat(cols, axis=-1)
    return cols

def __get_padding(in_height , in_width, filter_height ,filter_width, strides):
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

def pool2d( X : Variable , pad , size , stride , mode = Variable.max , NHWC : bool = True):
        """
        these ugly tranposes make me wanna implement my own einops , one day i may,
        basically we make sure the data are arranged in the correct order for the computation
        and then returning the correct format 
        """
        assert mode in (Variable.max , Variable.mean)
        # b h w c -> b c h w
        X = X.transpose(0,-1,1,2) if NHWC else X
        if isinstance(pad , int):
            pad = (pad,pad,pad,pad)
        elif isinstance(pad , str):
             pad = pad.lower()
             if pad == 'valid':
                  pad = (0,0,0,0)
             elif pad == 'same':
                  pad = __get_padding(X.shape[2] , X.shape[3], size[0] , size[1] , stride)
             else: raise ValueError('only VALID , SAME or explicit pad values supported')
             
        N, C, H_prev, W_prev = X.shape
        if sum(pad) == 0:
           """ 
           https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2 
           """
           H = math.ceil( (H_prev - size[0] + 1) / stride[0] )
           W = math.ceil( (W_prev - size[1] + 1) / stride[1] )
        else:
           H = math.ceil( H_prev / stride[0])
           W = math.ceil( W_prev / stride[1])
        image_2col = _im2col(X, size[0], size[1], stride,pad,  H , W)
        image_2col = image_2col.reshape(C, image_2col.shape[0]//C, -1)
        y = mode(image_2col, axis=1)
        y = tuple(y.split(N,axis=1))
        y = y[0].cat( y[1:] )
        # b c h w -> b h w c  
        y = y.reshape(N, C, H, W)
        return y.transpose(0 , 2, -1 , 1) if NHWC else y

def convolve2d(X :Variable , W :Variable ,  pad  , stride , NHWC : bool = True):
        """
        torch mainly uses NCHW and tensorflow NHWC i personally prefer the later ,
        when features are the second axis the kernel is ( filters , features , size0 , size1 ),
        and when the features are last the kernel is ( size0 , size1 , features , filters ),
        making sure the data are arranged properly is what these ugly tranposes are doing
        """
        X = X.transpose(0 , -1 , 1, 2) if NHWC else X
        w = W.transpose(-1 , 2 , 0 , 1)  if NHWC else W
        N, _, H, Wd = X.shape
        C = w.shape[0]
        if sum(pad) == 0:
           """ 
           https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2 
           """
           H = math.ceil( (H - w.shape[-2] + 1) / stride[0] )
           Wd = math.ceil( (Wd - w.shape[-1] + 1) / stride[1] )
        else:
           H = math.ceil( H / stride[0])
           Wd = math.ceil( Wd / stride[1])
        image_2col = _im2col(X, w.shape[-2], w.shape[-1], stride,pad , H , Wd)
        kernel_2col = w.reshape(w.shape[0], -1)
        y = kernel_2col.dot(image_2col) 
        y = tuple(y.split(N , axis=1))
        y = y[0].cat(y[1:] ).reshape(N, C , H , Wd)
        return y.transpose(0 , 2, -1 , 1)  if NHWC else y


def __validate_conv_args(data_format : str , image_shape , kernel_shape , strides , pad):
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
     if isinstance(pad ,str):
          pad = pad.lower()
          if pad == 'valid':
               pad = (0,0,0,0)
          elif pad == 'same':
               pad = __get_padding(h , w, k1 , k2 , strides)
          else: raise ValueError('only VALID , SAME or explicit pad values supported')
     elif isinstance(pad , int):
          pad = tuple( (pad,pad,pad,pad) )
     return nhwc , strides , pad 

############### usable functions ###################################################################

def conv2d(
            Image : Variable , Kernel : Variable ,
            strides : Tuple[int] | int = (1,1) ,
            pad: str | Tuple[int] | int = 'VALID',
            data_format = 'NHWC'
            ) -> Variable :
     
     nhwc , strides , pad = __validate_conv_args(data_format, Image.shape , Kernel.shape , strides , pad)
     return convolve2d(Image , Kernel , pad , strides , nhwc)

def max_pool2d( X :Variable,
                pool_size ,
                stride,
                pad,  
                data_format : str = 'NHWC'
                ) -> Variable:
     df = data_format.lower()
     if df == 'nhwc':
          nhwc = True
     elif df == 'nchw':
          nhwc=False
     else :raise ValueError('only NHWC  and NCHW formats supported')
     return pool2d(X , pad , pool_size , stride , NHWC= nhwc)

def avg_pool2d( X :Variable,
                pool_size ,
                stride,
                pad,  
                data_format : str = 'NHWC'
                ) -> Variable:
     df = data_format.lower()
     if df == 'nhwc':
          nhwc = True
     elif df == 'nchw':
          nhwc=False
     else :raise ValueError('only NHWC  and NCHW formats supported')
     return pool2d(X , pad , pool_size , stride , mode = Variable.mean , NHWC= nhwc)


def moments(x : Variable , axis : list | tuple = -1 , 
            keepdims = False , correction : int = 1 ):
    if isinstance(axis , int):
        axis = (axis,)
    mean : Variable = x.mean(axis , keepdims = True)
    shift = x - mean
    scale =  1.0 / (np.prod( [ x.shape[ax] for ax in axis ] )-correction)
             
    variance = (shift**2).sum(axis , keepdims) * scale

    return mean if keepdims else mean.squeeze(axis) , variance

def batch_norm(
               x : Variable , 
               mean : Variable , 
               variance : Variable , 
               gamma : Variable = None,
               beta : Variable = None ,
               eps : float =  1e-5
               ):
    inv_std = (variance+eps).rsqrt()
    y = (x - mean) * inv_std
    if gamma is not None:
        y = y * gamma
    if beta is not None:
        y =  y + beta
    return y
