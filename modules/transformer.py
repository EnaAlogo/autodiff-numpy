import math
import ml
from ml.Variable import Variable
from modules.layers import Layer , Linear , Dropout , LayerNorm , Lambda , Embedding
from modules.containers import Sequential

class CausalSelfAttention(Layer):

    def __init__(self ,
                  n_embed ,
                  n_head  , 
                  block_size,
                  attn_dropout_rate,
                  resid_dropout_rate
                  ):
        super(CausalSelfAttention,self).__init__()

        self.c_attn = Linear(3 * n_embed , weight_initializer= ml.initializers.RandomNormal(mean=0,std=.02))
        self.c_proj = Linear(n_embed)
        self.attn_dropout = Dropout(attn_dropout_rate)
        self.resid_dropout = Dropout(resid_dropout_rate)
        self.bias = ml.ones((1,1,block_size,block_size), requires_grad = False, dtype='bool').tril()
        self.n_head = n_head
        self.n_embd = n_embed

    @property
    def parameters_(self): return self.c_attn.parameters_ + self.c_proj.parameters_

    def call(self , x):
        B , T , C = x.shape
        q , k , v = self.c_attn(x).split( 3 , axis=2)

        k :ml.tensor = k.reshape(B , T , self.n_head , C // self.n_head).transpose(1,2)
        q :ml.tensor = q.reshape(B , T , self.n_head , C // self.n_head).transpose(1,2)
        v :ml.tensor = v.reshape(B , T , self.n_head , C // self.n_head).transpose(1,2)

        #(B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = ml.linalg.matmul(k , q, transpose_b=True) * (1.0 / math.sqrt(k.shape[-1]))
   
        att = ml.ops.where(self.bias[:,:,:T,:T] == 0  , float('-inf') , att )
        att = ml.activations.softmax(att , dim= -1)
        att = self.attn_dropout(att)

        y = att @ v 
        y = y.transpose(1,2).contiguous().reshape(B,T,C)
        y = self.resid_dropout(self.c_proj(y))
        return y
    

class Block(Layer):

    def __init__(self,
                 n_embed ,
                 n_head  , 
                 block_size,
                 attn_dropout_rate,
                 resid_dropout_rate
                 ):
        super(Block , self).__init__()

        self.ln_1 = LayerNorm()
        self.attn = CausalSelfAttention(n_embed ,n_head  , block_size,attn_dropout_rate, resid_dropout_rate)
        self.ln_2 = LayerNorm()
        self.mlp = Sequential([
            Linear(4 * n_embed),
            Lambda( lambda x : ml.activations.gelu(x , use_approx= False) ),
            Linear(n_embed),
            Dropout(resid_dropout_rate)
        ])

    @property
    def parameters_(self): return self.ln_1.parameters() + self.attn.parameters() + self.ln_2.parameters()\
    + self.mlp.parameters()

    def call(self , x):
        x  = self.attn(self.ln_1(x)) + x #residual conn
        x  = self.mlp(self.ln_2(x)) + x #residual conn
        return x
    

class Transformer(Layer):
    def __init__(self ,
                  n_layers,
                  vocab_size , 
                  block_size ,
                  n_embd , 
                  n_head,
                  dropout_rate,
                  attn_dropout_rate,
                  resid_dropout_rate
                  ):
        super(Transformer , self).__init__()
        self.wte = Embedding(vocab_size , n_embd)
        self.wpe = Embedding(block_size , n_embd)
        self.drop = Dropout(dropout_rate)
        self.h = Sequential([
            Block(n_embd,n_head,block_size,attn_dropout_rate , resid_dropout_rate) 
            for _ in range(n_layers)
            ])
        self.lm_head = Linear(vocab_size , use_bias= False)

    
    @property
    def parameters_(self):
        return self.wte.parameters_ + self.wpe.parameters_ + self.h.parameters() + self.lm_head.parameters_

    def call(self , IDX :ml.tensor ):
        B , T = IDX.shape

        pos :ml.tensor = ml.arange(0 , T , dtype='int64',requires_grad=False).unsqueeze(0)

        tok_embd = self.wte(IDX)
        pos_emb = self.wpe(pos)

        x = self.drop(tok_embd + pos_emb)
        x = self.h(x)

        return self.lm_head(x)
    

 ####################################################################################################################   

def rotate_half(x : Variable ) ->Variable :
    x  = x.reshape(*x.shape[:-1],x.shape[-1]//2 , 2 )
    x1, x2 = x.unstack(axis = -2)
    return ml.ops.concat((-x2 , x1), axis = -1)


def apply_rotary_pos_emb(pos : Variable , t : Variable) ->Variable:
    return ( t * pos.cos() ) + ( rotate_half(t) * pos.sin() )

class RotaryEmbedding(Layer):

    def __init__(self, dim):
        super().__init__()
        self.invFreq = 1. / (10000 ** (ml.arange(0, dim ,2 , dtype = 'float32' ,requires_grad= False)) / dim )
             
    def call(self, max_seq_len) ->Variable :
        seq : Variable = ml.arange(0 , max_seq_len , dtype = self.invFreq.dtype)
        freqs = seq.outer_product(self.invFreq)
        return ml.ops.concat((freqs,freqs) , axis = -1)
 

class ParallelTransformerBlock(Layer):
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.norm = LayerNorm(dim)

        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = (attn_inner_dim, dim_head, dim_head, (ff_inner_dim * 2))

        self.heads = heads
        self.scale = dim_head**-0.5
        self.rotary_emb = RotaryEmbedding(dim_head)

        self.fused_attn_ff_proj = Linear( sum(self.fused_dims), use_bias=False)
        self.attn_out = Linear( dim, use_bias=False)

        self.ff_out = Sequential([
            Lambda( ml.activations.swiglu ),
            Linear( dim, use_bias=False)
            ])

        # for caching causal mask and rotary embeddings
        self.mask = None
        self.pos_emb = None

    def get_mask(self, n):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]
        mask = ml.ones((n, n),  dtype='bool').triu(1)
        return mask

    def get_rotary_embedding(self, n, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n]
        pos_emb = self.rotary_emb(n)
        return pos_emb
    
    def call(self, x, attn_mask=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n,  h = x.shape[1],  self.heads

        # pre layernorm

        x = self.norm(x)

        # attention queries, keys, values, and feedforward inner

        q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)

        # split heads
        # they use multi-query single-key-value attention, yet another Noam Shazeer paper
        # they found no performance loss past a certain scale, and more efficient decoding obviously
        # https://arxiv.org/abs/1911.02150

        q :Variable = q.reshape(q.shape[0] , q.shape[1] // h  , n , -1 )
        #rearrange(q, "b n (h d) -> b h n d", h=h)

        # rotary embeddings

        positions = self.get_rotary_embedding(n)
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))

        # scale

        q = q * self.scale

        # similarity
        # (b h w c) X (b 1 j c) -> b h w j
        sim = ml.linalg.matmul(q, k , transpose_b= True )

        # causal mask

        causal_mask = self.get_mask(n)
        sim = ml.ops.where(causal_mask , float('-inf') , sim)

        # extra attention mask - for masking out attention from text CLS token to padding

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
            #rearrange(attn_mask, 'b i j -> b 1 i j')
            sim = ml.ops.where(~attn_mask , float('-inf') , sim )
           # sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        # attention
        sim = sim - sim.max(axis = -1 , keepdims= True).detach()
        attn = ml.activations.softmax(sim , -1)
        
        # (b h w c) X (b 1 c d) -> b h w d
        out = ml.linalg.matmul( attn  , v )

        # merge heads
        out = out.reshape( *out.shape[:-2] , -1 )

        return self.attn_out(out) + self.ff_out(ff)
    

class CrossAttention(Layer):
    def __init__(
        self,
        dim,
        context_dim=None,
        dim_head=64,
        heads=8,
        parallel_ff=False,
        ff_mult=4,
        norm_context=False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        context_dim = context_dim if context_dim is not None else dim

        self.norm = LayerNorm()
        self.context_norm = LayerNorm() if norm_context else lambda x : x

        self.to_q = Linear(inner_dim, use_bias=False)
        self.to_kv = Linear( dim_head * 2, use_bias=False)
        self.to_out = Linear( dim, use_bias=False)

        # whether to have parallel feedforward

        ff_inner_dim = ff_mult * dim

        self.ff = Sequential([
            Linear( ff_inner_dim * 2, use_bias=False),
            Lambda( ml.activations.swiglu ),
            Linear( dim, use_bias=False)
            ]) if parallel_ff else None
        
    def call(self, x, context):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        # pre-layernorm, for queries and context

        x = self.norm(x)
        context = self.context_norm(context)

        # get queries

        q = self.to_q(x)
        q = q.reshape(*q.shape[:2] , q.shape[-1] // self.heads , -1)
        # scale

        q = q * self.scale

        # get key / values

        k, v = self.to_kv(context).split(2, dim=-1)

        # query / key similarity
         # (b h w c) X (b 1 j c) -> b h w j
        sim = ml.linalg.matmul(q,k,transpose_b = True)

        # attention

        attn = ml.activations.softmax(sim , -1 )

        # aggregate
        # (b h w c) X (b 1 c j) -> b h w j
        out = ml.linalg.matmul(attn , v )
 
        # merge and combine heads
        
        out = out.reshape(*out.shape[:2] , -1)
     
        out = self.to_out(out)

        # add parallel feedforward (for multimodal layers)

        if self.ff is not None:
            out = out + self.ff(x)

        return out