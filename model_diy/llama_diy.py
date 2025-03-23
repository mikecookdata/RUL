import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

device = "cuda" if torch.cuda.is_available() else "cpu"


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    # print(f'position_ids: {position_ids}')
    # print(f'cos shape: {cos.shape}')
    # print(f'sin shape: {sin.shape}')
    # print(f'hg q shape: {q.shape}')
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
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        res1 = self.cos_cached[:seq_len].to(dtype=x.dtype)

        # print(f'diy res1 shape: {res1.shape}')
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaAttention(nn.Module):
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



        # Corrected: Ensure rope_freq only applies to `head_dim`
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )


    def forward(self, x):
        B, T, C = x.shape
        # print(f"Input shape: {x.shape}")

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # print(f'diy query_states: {q}')
        # print(f'diy key_states: {k}')
        # print(f'diy value_states: {v}')
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Queries and Keys (fixed shape mismatch)
        cos, sin = self.rotary_emb(v, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=torch.arange(T, device=x.device).unsqueeze(0))


        # print(f'diy query_states before attn: {q}')
        # print(f'diy key_states before attn: {k}')
        # print(f'diy value_states before attn: {v}')


                # Register the lower triangular mask for causal attention
        tril = torch.tril(torch.ones(T, T)).to(device)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_scores = attn_scores.masked_fill(tril[:T, :T] == 0, float("-inf"))
        attn_probs = F.softmax(attn_scores, dim=-1)
        # print(f'diy attn_probs: {attn_probs}')
        attn_output = (attn_probs @ v).transpose(1, 2).contiguous().view(B, T, C)

        return self.o_proj(attn_output)


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class LlamaDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, rope_theta):
        super().__init__()
        self.self_attn = LlamaAttention(hidden_size, num_heads, rope_theta)
        self.mlp = LlamaMLP(hidden_size, intermediate_size)
        self.input_layernorm = LlamaRMSNorm(hidden_size)
        self.post_attention_layernorm = LlamaRMSNorm(hidden_size)

    def forward(self, x):
        residual = x
        x = self.input_layernorm(x)
        # print(f'1st norm in diy decodelayer: {x}')

        x = self.self_attn(x)
        # print(f'after self_attn in diy decodelayer: {x}')
        # print(f'shape: {x.shape}')

        x = x + residual
        # print(f'after self_attn & +residual in diy decodelayer: {x}')

        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class LlamaModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_layers,
        num_heads,
        intermediate_size,
        padding_idx,
        rope_theta,
        max_length=4096,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(
            vocab_size, hidden_size, padding_idx=padding_idx
        )
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(hidden_size, num_heads, intermediate_size, rope_theta)
                for _ in range(num_layers)
            ]
        )
        self.norm = LlamaRMSNorm(hidden_size)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        # print(f"inputs_embeds from diy {x}")
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = LlamaModel(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            intermediate_size=config["intermediate_size"],
            padding_idx=config["pad_token_id"],
            rope_theta=config["rope_theta"],
        )
        self.lm_head = nn.Linear(
            config["hidden_size"], config["vocab_size"], bias=False
        )

    def forward(self, input_ids):
        hidden_states = self.model(input_ids)
        return self.lm_head(hidden_states)


def generate_text(model, input_ids, max_length=50):
    input_ids = input_ids.to(device)
    # Generate tokens autoregressively
    for i in range(max_length):
        # if i > 0:
        #     continue
        with torch.no_grad():
            if (
                type(model(input_ids))
                == transformers.modeling_outputs.CausalLMOutputWithPast
            ):
                logits = model(input_ids, output_attentions=True).logits
            else:
                logits = model(input_ids)

            next_token_probs = logits[:, -1, :].softmax(dim=-1)
            next_token = torch.argmax(next_token_probs, dim=-1, keepdim=True)
            # next_token = torch.multinomial(
            #     next_token_probs, num_samples=1
            # )

            # Append new token to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Stop if EOS (end of sentence) token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode output tokens to text
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

from transformers import AutoConfig

# Model config
model_name = "JackFram/llama-68m"
# model_name = "meta-llama/Llama-2-7b-hf"

# Load config
config = AutoConfig.from_pretrained(model_name)
# print(config)
model_config = {
    "vocab_size": config.vocab_size,
    "hidden_size": config.hidden_size,
    "num_layers": config.num_hidden_layers,
    "num_heads": config.num_attention_heads,
    "intermediate_size": config.intermediate_size,
    "pad_token_id": config.pad_token_id,
    "rope_theta": config.rope_theta
}

# Initialize custom model
custom_model = LlamaForCausalLM(model_config)

# Load pretrained weights
pretrained_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example usage
prompt = "once upon a time"
# Tokenize and move input to GPU
inputs = tokenizer(prompt, return_tensors="pt").to(device)
input_ids = inputs["input_ids"]

# pretrained model generation
# print(pretrained_model._modules)
print(generate_text(pretrained_model, input_ids))


# custom model
custom_model.load_state_dict(pretrained_model.state_dict(), strict=True)
# custom_model.half()
custom_model.to(device).eval()
# print(custom_model._modules)
print(generate_text(custom_model, input_ids))