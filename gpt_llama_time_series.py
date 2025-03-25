import torch
import torch.nn as nn
from torch.nn import functional as F
from icecream import ic

# hyperparameters
config = {'hidden_size': 768, 'num_layers': 2, 'num_heads': 12,
          'intermediate_size': 3072, 'pad_token_id': 1, 'rope_theta': 10000.0}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 8
block_size = 16
max_iters = 2000
eval_iters = 50
eval_interval = 500
learning_rate = 3e-4


torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
# encoder: take a string, output a list of integers
def encode(s): return [stoi[c] for c in s]
# decoder: take a list of integers, output a string
def decode(l): return ''.join([itos[i] for i in l])


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / \
            (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached,
                         device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer(
            "sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(
                seq_len=seq_len, device=x.device, dtype=x.dtype)
        res1 = self.cos_cached[:seq_len].to(dtype=x.dtype)

        # print(f'diy res1 shape: {res1.shape}')
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, rope_theta, max_length=4096):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.max_position_embeddings = max_length
        self.rope_theta = rope_theta

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(self, x):
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Queries and Keys (fixed shape mismatch)
        cos, sin = self.rotary_emb(v, seq_len=T)
        q, k = apply_rotary_pos_emb(
            q, k, cos, sin, position_ids=torch.arange(T, device=x.device).unsqueeze(0))

        tril = torch.tril(torch.ones(T, T)).to(device)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_scores = attn_scores.masked_fill(tril[:T, :T] == 0, float("-inf"))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = (attn_probs @ v).transpose(1,
                                                 2).contiguous().view(B, T, C)

        return self.o_proj(attn_output)


class FeedFoward(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Linear(intermediate_size, hidden_size)
        )

    def forward(self, x):
        return self.net(x)


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, rope_theta):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.sa = Attention(hidden_size, num_heads, rope_theta)
        self.ffwd = FeedFoward(hidden_size, intermediate_size)

    def forward(self, x):
        x = x + self.sa(self.ln_1(x))
        x = x + self.ffwd(self.ln_2(x))
        return x


class LanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_layers,
        num_heads,
        intermediate_size,
        rope_theta,
        max_length=4096,
    ):
        super().__init__()
        self.embeddings_table = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.Sequential(
            *[DecoderLayer(hidden_size, num_heads, intermediate_size, rope_theta) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        self.apply(self._init_weights)

    def forward(self, idx, targets):
        x = self.embeddings_table(idx)
        x = self.layers(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = LanguageModel(vocab_size=vocab_size, hidden_size=config['hidden_size'], num_layers=config['num_layers'], num_heads=config['num_heads'], 
                      intermediate_size=config['intermediate_size'], rope_theta=config['rope_theta'])
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

if __name__ == "__main__":
    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            # Save the trained model
            # torch.save(model.state_dict(), model_save_path)
            # print(f"Model saved to {model_save_path}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
