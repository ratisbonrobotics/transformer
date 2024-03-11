import torch

class FeedForward(torch.nn.Module):
    def __init__(self, hidden_dim, ff_dim):
        super().__init__()
        self.w1 = torch.nn.Linear(hidden_dim, ff_dim, bias=False)
        self.w2 = torch.nn.Linear(ff_dim, hidden_dim, bias=False)

        torch.nn.init.kaiming_normal_(self.w1.weight, nonlinearity='relu')
        torch.nn.init.xavier_uniform_(self.w2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.nn.functional.gelu(self.w1(x), approximate='tanh'))

class Attention(torch.nn.Module):
    def __init__(self, n_heads, hidden_dim, head_dim):
        super().__init__()
        self.n_heads = n_heads
        self.scale = head_dim**-0.5
        self.q_linear = torch.nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.k_linear = torch.nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.v_linear = torch.nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.wo = torch.nn.Linear(n_heads * head_dim, hidden_dim, bias=False)

        torch.nn.init.xavier_uniform_(self.q_linear.weight)
        torch.nn.init.xavier_uniform_(self.k_linear.weight)
        torch.nn.init.xavier_uniform_(self.v_linear.weight)
        torch.nn.init.xavier_uniform_(self.wo.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        q = self.q_linear(x).view(bsz, seqlen, self.n_heads, -1).transpose(1,2)
        k = self.k_linear(x).view(bsz, seqlen, self.n_heads, -1).transpose(1,2)
        v = self.v_linear(x).view(bsz, seqlen, self.n_heads, -1).transpose(1,2)

        scores = torch.matmul(q, k.transpose(2, 3)) * self.scale
        scores = scores.masked_fill(torch.tril(torch.ones(seqlen, seqlen, device=x.device)) == 0, float('-inf'))
        scores = torch.nn.functional.softmax(scores, dim=-1)

        output = torch.matmul(scores, v)
        output = output.transpose(1,2).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)

class TransformerBlock(torch.nn.Module):
    def __init__(self, num_heads, hidden_dim, ff_dim):
        super().__init__()
        self.n_heads = num_heads
        self.hidden_dim = hidden_dim
        self.attention = Attention(num_heads, hidden_dim, hidden_dim // num_heads)
        self.feed_forward = FeedForward(hidden_dim, ff_dim)
        self.attention_norm = torch.nn.LayerNorm(hidden_dim)
        self.ffn_norm = torch.nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x))
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out

class LanguageModel(torch.nn.Module):
    def __init__(self, vocab_size, num_blocks=16, num_heads=8, hidden_dim=768, ff_dim=2048):
        super(LanguageModel, self).__init__()
        self.tok_emb = torch.nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = torch.nn.Embedding(2048, hidden_dim)
        self.pos_norm = hidden_dim**-0.5
        self.transformer_blocks = torch.nn.ModuleList([TransformerBlock(num_heads, hidden_dim, ff_dim) for _ in range(num_blocks)])
        self.out_norm = torch.nn.LayerNorm(hidden_dim)
        self.out_linear = torch.nn.Linear(hidden_dim, vocab_size, bias=False)

        torch.nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        torch.nn.init.xavier_uniform_(self.out_linear.weight)

    def forward(self, token_ids: torch.Tensor):
        x = self.tok_emb(token_ids)
        x = x + self.pos_emb(torch.arange(token_ids.shape[1], device=token_ids.device)) * self.pos_norm
        
        for block in self.transformer_blocks:
            x = block(x)

        return self.out_linear(self.out_norm(x))