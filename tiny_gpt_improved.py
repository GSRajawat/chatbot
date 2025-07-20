import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# --- Data Preparation ---
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

data = torch.tensor(encode(text), dtype=torch.long)

# --- Model Hyperparameters ---
batch_size = 32
block_size = 64 # Context length
embed_size = 64 # Dimension of token embeddings
num_heads = 4   # Number of attention heads
num_layers = 4  # Number of Transformer blocks
dropout_rate = 0.1
learning_rate = 1e-3
training_steps = 1000

# --- Helper Function for Batching ---
def get_batch(batch_size=batch_size, block_size=block_size):
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# --- Attention Head ---
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, T, C = x.shape # Batch, Time (sequence length), Channels (embedding size)
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * (C**-0.5) # (B, T, T) scaled dot product
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # Apply decoder mask
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # Perform the weighted aggregation of the values
        v = self.value(x) # (B, T, head_size)
        out = wei @ v     # (B, T, head_size)
        return out

# --- Multi-Head Attention ---
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_size, embed_size) # Projection back to original embedding size
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

# --- Feed-Forward Network ---
class FeedFoward(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size), # Typically 4x the embedding size
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.net(x)

# --- Transformer Block ---
class Block(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        head_size = embed_size // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedFoward(embed_size)
        self.ln1 = nn.LayerNorm(embed_size) # Layer normalization before attention
        self.ln2 = nn.LayerNorm(embed_size) # Layer normalization before feed-forward

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # Residual connection + LayerNorm + Attention
        x = x + self.ffwd(self.ln2(x)) # Residual connection + LayerNorm + Feed-forward
        return x

# --- Full TinyGPT Model ---
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, embed_size=embed_size, block_size=block_size, num_heads=num_heads, num_layers=num_layers):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
        self.position_embedding_table = nn.Embedding(block_size, embed_size)
        self.blocks = nn.Sequential(*[Block(embed_size, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_size) # Final layer normalization
        self.lm_head = nn.Linear(embed_size, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Token and positional embeddings
        tok_emb = self.token_embedding_table(idx)             # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T)) # (T, C)
        x = tok_emb + pos_emb                                 # (B, T, C) combine them

        # Apply transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x) # Apply final layer norm
        logits = self.lm_head(x)                              # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # Reshape for F.cross_entropy
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # Get predictions
            logits, loss = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


# --- Model Instantiation and Training ---
model = TinyGPT(vocab_size).to("cpu") # .to("cpu") is explicitly mentioned for clarity, remove if you have a GPU
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

for step in range(training_steps):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step}, loss {loss.item():.4f}")

# --- Text Generation ---
print("\n--- Generated Text ---")
# Start with a prompt, encode it, add a batch dimension
start_text = "User: "
context = torch.tensor(encode(start_text), dtype=torch.long).unsqueeze(0)
generated_indices = model.generate(context, max_new_tokens=100)[0].tolist()
print(decode(generated_indices))

# After your training loop (e.g., at the end of your original script)
# Save the model
model_path = "C:/Users/GOVT LAW COLLEGE 107/Documents/ai assistant/tiny_gpt_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
