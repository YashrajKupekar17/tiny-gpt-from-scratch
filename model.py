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

