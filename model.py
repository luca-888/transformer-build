import torch
import torch.nn
import math

class InputEmbedding(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model # 
        self.vocab_size = vocab_size # 词汇表大小
        self.embedding = nn.Embedding(vocab_size, d_model) # 词嵌入层，这里是一个线性层

    def forward(self, x: torch.Tensor):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model)) -> sin(pos / e^(log(10000)/d_model * 2i))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model)) -> cos(pos / e^(log(10000)/d_model * 2i))
        # Small trick for numerical stability
        pe = torch.zeros(seq_len, d_model) # (seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        return self.dropout(x)


