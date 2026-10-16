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

