# utils.py
import torch

def load_vocabulary(text_file_path):
    with open(text_file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    return vocab_size, stoi, itos, encode, decode

# You can keep get_batch here too if you ever need it in a separate script
# def get_batch(...):
#    ...
