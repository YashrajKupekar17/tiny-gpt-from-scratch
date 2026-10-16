"""
Tiny GPT From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_char_vocab
def build_char_vocab(text):
    # TODO: return a dict mapping each unique character in text to an integer id in sorted order

    s = sorted(set(text))
    vocab = {}
    for i in range(len(s)):
        vocab[s[i]]=i
    return vocab
    pass

# Step 2 - build_id_to_char
def build_id_to_char(char_to_id):
    # TODO: invert the char->id dict into an id->char dict for decoding
    id_to_char = {char_to_id[k]: k for k in char_to_id}
    # for k in char_to_id:
    #     id_to_char[char_to_id[k]]=k

    return id_to_char
    pass

# Step 3 - encode_text
import numpy as np

def encode_text(text, char_to_id):
    # TODO: convert the string into a 1D int64 numpy array of token ids using char_to_id
    enc = np.array([char_to_id[i] for i in text],dtype=np.int64)
    return enc
    pass

# Step 4 - decode_ids
def decode_ids(ids, id_to_char):
    # TODO: convert a sequence of integer token ids back into the original string
    s=''
    for i in ids:
        s+=id_to_char[i]
    return s
    pass

# Step 5 - make_batches
import numpy as np

def make_batches(data, batch_size, block_size, rng):
    # TODO: sample batch_size random windows of length block_size from data and return (x, y) where y is x shifted by one.
    idxs = rng.integers(0, len(data) - block_size, size=batch_size)
    x = np.array([data[i:i+block_size] for i in idxs],dtype=np.int64)
    y = np.array([data[i+1:i+block_size+1] for i in idxs],dtype=np.int64) 
    return (x,y)

# Step 6 - token_embedding_lookup
def token_embedding_lookup(token_ids, embedding_table):
    # TODO: return embedding vectors for each token id, output shape (B, T, d_model)
    embeddings = embedding_table[token_ids]
    return embeddings

# Step 7 - add_positional_embeddings
import numpy as np

def add_positional_embeddings(token_embeds, pos_embedding_table):
    # TODO: add learned positional embeddings (sliced to length T) to token embeddings
    # token_embeds --> (B,T,C)
    # pos_embedding table -->(block_size,C)
    B,T,C = token_embeds.shape
    pos = pos_embedding_table[:T]
    embedding = token_embeds+pos
    return embedding

# Step 8 - linear_projection
import numpy as np

def linear_projection(x, weight, bias):
    # TODO: apply a generic dense linear projection x @ weight + bias (bias may be None)
    out = x @ weight
    if bias is not None :
        out = out + bias
    return out

# Step 9 - compute_attention_scores
def compute_attention_scores(q, k):
    # TODO: return raw attention scores Q @ K^T over the last two dims, shape (..., T, T)
    # k --> (B,T,head_dim)
    # q --> (B,T,head_dim)
    k_trans = np.swapaxes(k, -2, -1)
    # k_trans --> (B,head_dim,T)
    scores = q @ k_trans
    return scores

# Step 10 - scale_attention_scores
import numpy as np

def scale_attention_scores(scores, head_dim):
    # TODO: rescale raw attention scores by 1/sqrt(head_dim) for stable softmax
    out = scores*((head_dim**(1/2))**-1)
    return out

# Step 11 - apply_causal_mask
import numpy as np

def apply_causal_mask(scores):
    # TODO: build an upper-triangular causal mask and fill future positions in scores with -inf
    T = scores.shape[-1]
    tril = np.tril(np.ones((T,T)))
    scores = np.where(tril==0,-np.inf,scores)
    return scores

# Step 12 - softmax_attention_weights
import numpy as np

def softmax_attention_weights(masked_scores):
    # TODO: apply a numerically stable softmax over the last axis of masked_scores
    maxes = np.max(masked_scores,-1,keepdims=True)
    exp_scores = np.exp(masked_scores-maxes)
    exp_sum = np.sum(exp_scores,-1,keepdims=True)
    out = (exp_scores*(exp_sum**-1))
    return out

# Step 13 - attention_context
import numpy as np

def attention_context(attn_weights, v):
    # TODO: multiply attention weights by the value tensor to produce per-token context vectors
    # attn_weights --> (B, T,T)
    # v --> (B,T,head_dim)
    out = attn_weights @ v
    return out

# Step 14 - split_heads
import numpy as np

def split_heads(x, num_heads):
    """Reshape (B, T, d_model) into (B, num_heads, T, head_dim)."""
    # TODO: split the last axis into (num_heads, head_dim) and move heads in front of T
    B, T, d_model = x.shape
    head_dim = d_model // num_heads
    x = x.reshape(B, T,num_heads, head_dim)   # step 1: split last axis only
    x = np.transpose(x, (0, 2, 1, 3))          # step 2: swap T and num_heads
    return x

# Step 15 - merge_heads
import numpy as np

def merge_heads(x):
    # TODO: merge (B, num_heads, T, head_dim) back into (B, T, num_heads * head_dim)
    B, num_heads, T, head_dim = x.shape
    x = np.transpose(x,(0,2,1,3))
    x= x.reshape(B,T,head_dim*num_heads)
    return x

# Step 16 - project_qkv
def project_qkv(x, w_q, b_q, w_k, b_k, w_v, b_v):
    # TODO: project x into separate query, key, and value tensors via three linear projections.
    q = linear_projection(x,w_q,b_q)
    k = linear_projection(x,w_k,b_k)
    v = linear_projection(x,w_v,b_v)
    return (q,k,v)

