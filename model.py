import math
import torch

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        x_normed = x * torch.rsqrt(var + self.eps)
        return self.weight * x_normed

class FeedForward(torch.nn.Module):
    def __init__(self, hidden_dim, ff_dim):
        super().__init__()
        self.w1 = torch.nn.Linear(hidden_dim, ff_dim, bias=False)
        self.w2 = torch.nn.Linear(ff_dim, hidden_dim, bias=False)
        self.w3 = torch.nn.Linear(hidden_dim, ff_dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))

class Attention(torch.nn.Module):
    def __init__(self, n_heads, hidden_dim, head_dim):
        super().__init__()
        self.n_heads = n_heads
        self.scale = head_dim ** -0.5
        self.q_linear = torch.nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.k_linear = torch.nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.v_linear = torch.nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.wo = torch.nn.Linear(n_heads * head_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        q = self.q_linear(x).view(bsz, seqlen, self.n_heads, -1).transpose(1,2)
        k = self.k_linear(x).view(bsz, seqlen, self.n_heads, -1).transpose(1,2)
        v = self.v_linear(x).view(bsz, seqlen, self.n_heads, -1).transpose(1,2)

        scores = torch.einsum("bsid,bsjd->bsij", q, k) * self.scale
        scores = scores.masked_fill(torch.tril(torch.ones(seqlen, seqlen, device=x.device)) == 0, float('-inf'))
        scores = torch.nn.functional.softmax(scores, dim=-1)

        output = torch.einsum("bsij,bsjd->bsid", scores, v)
        output = output.transpose(1,2).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)

class TransformerBlock(torch.nn.Module):
    def __init__(self, num_heads, hidden_dim, ff_dim):
        super().__init__()
        self.n_heads = num_heads
        self.hidden_dim = hidden_dim
        self.attention = Attention(num_heads, hidden_dim, hidden_dim // num_heads)
        self.feed_forward = FeedForward(hidden_dim, ff_dim)
        self.attention_norm = RMSNorm(hidden_dim)
        self.ffn_norm = RMSNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x))
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(-torch.arange(0, d_model, 2) * math.log(10000.0) / d_model)
        pos_enc = torch.zeros((1, max_len, d_model))
        pos_enc[0, :, 0::2] = torch.sin(position * div_term)
        pos_enc[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_enc', pos_enc)
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor):
        return self.norm(x + self.pos_enc[:, :x.size(1)])

class LanguageModel(torch.nn.Module):
    def __init__(self, vocab_size, num_blocks=16, num_heads=8, hidden_dim=768, ff_dim=2048):
        super(LanguageModel, self).__init__()
        self.tok_emb = torch.nn.Embedding(vocab_size, hidden_dim)
        self.positional_encodings = PositionalEncoding(hidden_dim)
        self.transformer_blocks = torch.nn.ModuleList([TransformerBlock(num_heads, hidden_dim, ff_dim) for _ in range(num_blocks)])
        self.norm_out = RMSNorm(hidden_dim)
        self.linear_out = torch.nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor):
        x = self.tok_emb(token_ids)
        x = self.positional_encodings(x)
        
        for block in self.transformer_blocks:
            x = block(x)

        return self.linear_out(self.norm_out(x))