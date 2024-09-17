class AttentionHead(nn.Module):
    def __init__(self, d, d_k):
        super().__init__()
        self.W_Q = nn.Parameter(torch.randn(d, d_k))
        self.W_K = nn.Parameter(torch.randn(d, d_k))
        self.W_V = nn.Parameter(torch.randn(d, d_k))
        self.d_k = d_k

    def forward(self, x, mask):
        Q = x @ self.W_Q
        K = x @ self.W_K
        V = x @ self.W_V
        Q = rope(Q)
        K = rope(K)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        return attn_weights @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, d, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(d, d // num_heads) for _ in range(num_heads)])
        self.W_O = nn.Parameter(torch.randn(d, d))

    def forward(self, x, mask):
        x = torch.cat([head(x, mask) for head in self.heads], dim=-1)
        return x @ self.W_O


class FeedForward(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.W_1 = nn.Parameter(torch.randn(d, d))
        self.W_2 = nn.Parameter(torch.randn(d, d))
        self.B_2 = nn.Parameter(torch.randn(d))

    def forward(self, x):
        x = x @ self.W_1
        x = torch.relu(x)
        x = x @ self.W_2 + self.B_2
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d, num_heads):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = MultiHeadAttention(d, num_heads)
        self.norm2 = RMSNorm(d)
        self.ff = FeedForward(d)

    def forward(self, x, mask):
        attn_out = self.attn(self.norm1(x), mask)
        x = x + attn_out
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d)
        self.layers = nn.ModuleList([DecoderBlock(d, num_heads) for _ in range(num_layers)])
        self.norm = RMSNorm(d)
        self.output = nn.Linear(d, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        seq_len = x.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return self.output(x)


for epoch in range(10):
    model.train()
    for input_seq, target_seq in train_dataloader:
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)
        logits = model(input_seq)
        loss = criterion(logits.transpose(1, 2), target_seq)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
