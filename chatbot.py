import torch

with open("chat_data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Create vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Char to int and vice versa
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

# Encoder/decoder
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


import torch.nn as nn
import torch.nn.functional as F

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, embed_size=64, context_len=64):
        super().__init__()
        self.context_len = context_len
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size)

    def forward(self, idx):
        tok_emb = self.token_embedding(idx)  # (B, T, C)
        logits = self.lm_head(tok_emb)       # (B, T, vocab_size)
        return logits


import torch.optim as optim

model = TinyGPT(vocab_size)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

data = torch.tensor(encode(text), dtype=torch.long)

def get_batch(batch_size=32, block_size=64):
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

for step in range(10000):
    xb, yb = get_batch()
    logits = model(xb)
    loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step}, loss {loss.item():.4f}")


def generate(model, start_text, max_new_tokens=100):
    model.eval()
    idx = torch.tensor(encode(start_text), dtype=torch.long).unsqueeze(0)
    for _ in range(max_new_tokens):
        logits = model(idx[:, -model.context_len:])
        next_id = torch.argmax(logits[0, -1], dim=-1, keepdim=True)
        idx = torch.cat((idx, next_id.unsqueeze(0)), dim=1)
    return decode(idx[0].tolist())

print(generate(model, "User: "))
