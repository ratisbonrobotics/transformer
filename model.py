import torch

class SRMSNorm(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.normalize(x, dim=-1) * self.scale

class FeedForward(torch.nn.Module):
    def __init__(self, hidden_dim, ff_dim):
        super().__init__()
        self.in_linear = torch.nn.Linear(hidden_dim, ff_dim, bias=False)
        self.out_linear = torch.nn.Linear(ff_dim, hidden_dim, bias=False)

        torch.nn.init.kaiming_normal_(self.in_linear.weight, nonlinearity='relu')
        torch.nn.init.xavier_uniform_(self.out_linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_linear(torch.nn.functional.gelu(self.in_linear(x), approximate='tanh'))

class Attention(torch.nn.Module):
    def __init__(self, n_heads, hidden_dim, head_dim, seq_len):
        super().__init__()
        self.n_heads = n_heads
        self.scale = head_dim**-0.5
        self.q_linear = torch.nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.k_linear = torch.nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.v_linear = torch.nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.o_linear = torch.nn.Linear(n_heads * head_dim, hidden_dim, bias=False)
        self.register_buffer("mask", torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool())

        torch.nn.init.xavier_uniform_(self.q_linear.weight)
        torch.nn.init.xavier_uniform_(self.k_linear.weight)
        torch.nn.init.xavier_uniform_(self.v_linear.weight)
        torch.nn.init.xavier_uniform_(self.o_linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_linear(x).view(batch_size, seq_len, self.n_heads, -1).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.n_heads, -1).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.n_heads, -1).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(2, 3)) * self.scale
        scores = scores.masked_fill(self.mask[:seq_len, :seq_len], float('-inf'))
        scores = torch.nn.functional.softmax(scores, dim=-1)

        output = torch.matmul(scores, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.o_linear(output)

class TransformerBlock(torch.nn.Module):
    def __init__(self, num_heads, hidden_dim, ff_dim, seq_len):
        super().__init__()
        self.attention = Attention(num_heads, hidden_dim, hidden_dim // num_heads, seq_len)
        self.feed_forward = FeedForward(hidden_dim, ff_dim)
        self.attention_norm = SRMSNorm(hidden_dim)
        self.ffn_norm = SRMSNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x))
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out

class LanguageModel(torch.nn.Module):
    def __init__(self, vocab_size=32768, seq_len=2048, num_blocks=16, num_heads=8, hidden_dim=768, ff_dim=2048):
        super(LanguageModel, self).__init__()
        self.tok_emb = torch.nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = torch.nn.Embedding(seq_len, hidden_dim)
        self.register_buffer("pos", torch.arange(seq_len, dtype=torch.int))
        self.pos_norm = SRMSNorm(hidden_dim)
        self.transformer_blocks = torch.nn.ModuleList([TransformerBlock(num_heads, hidden_dim, ff_dim, seq_len) for _ in range(num_blocks)])
        self.out_norm = SRMSNorm(hidden_dim)
        self.out_linear = torch.nn.Linear(hidden_dim, vocab_size, bias=False)

        torch.nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        torch.nn.init.xavier_uniform_(self.out_linear.weight)

    def forward(self, token_ids: torch.Tensor):
        x = self.tok_emb(token_ids) + self.pos_emb(self.pos)
        x = self.pos_norm(x)
        
        for block in self.transformer_blocks:
            x = block(x)

        return self.out_linear(self.out_norm(x))