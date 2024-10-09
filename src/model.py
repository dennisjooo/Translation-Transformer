import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLP(nn.Module):
    def __init__(self, n_embed: int, n_hidden: int, dropout_p: float = 0.1):
        """
        Initialize the MLP module.

        Args:
            n_embed (int): The input embedding dimension.
            n_hidden (int): The hidden layer dimension.
            dropout_p (float, optional): Dropout probability. Defaults to 0.1.
        """
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(n_embed, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_embed)
        self.activation = nn.GELU(approximate='tanh')
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_embed).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, n_embed).
        """
        x = self.activation(self.fc1(x))
        return self.dropout(self.fc2(x))

class SelfAttention(nn.Module):
    def __init__(self, n_embed: int, n_head: int, is_causal: bool = False, dropout_p: float = 0.1):
        """
        Initialize the SelfAttention module.

        Args:
            n_embed (int): The input embedding dimension.
            n_head (int): The number of attention heads.
            is_causal (bool, optional): Whether to use causal attention. Defaults to False.
            dropout_p (float, optional): Dropout probability. Defaults to 0.1.
        """
        super(SelfAttention, self).__init__()
        
        self.n_head = n_head
        self.n_embed = n_embed
        self.head_dim = n_embed // n_head
        self.is_causal = is_causal
        self.dropout_p = dropout_p
        assert n_embed % n_head == 0, 'n_embed must be divisible by n_head'
        
        self.W_q = nn.Linear(n_embed, n_embed, bias=False)
        self.W_k = nn.Linear(n_embed, n_embed, bias=False)
        self.W_v = nn.Linear(n_embed, n_embed, bias=False)
        self.output = nn.Linear(n_embed, n_embed, bias=False)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the SelfAttention module.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, seq_len, n_embed).
            k (torch.Tensor): Key tensor of shape (batch_size, seq_len, n_embed).
            v (torch.Tensor): Value tensor of shape (batch_size, seq_len, n_embed).
            mask (torch.Tensor, optional): Attention mask tensor of shape (batch_size, 1, 1, seq_len). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, n_embed).

        Raises:
            ValueError: If input tensors have different embedding dimensions.
        """
        if not all(x.size(-1) == q.size(-1) for x in [k, v]):
            raise ValueError("All inputs must have the same embedding dimension")
        
        batch_size, seq_len, _ = q.size()
        
        q = self.W_q(q).view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)
        k = self.W_k(k).view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)
        v = self.W_v(v).view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)
        
        attention = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=mask,  # Pass the mask to the attention mechanism
            is_causal=self.is_causal,
            dropout_p=self.dropout_p if self.training else 0.0
        )
        
        attention = attention.contiguous().view(batch_size, seq_len, self.n_embed)
        
        return self.output(attention)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize the PositionalEncoding module.

        Args:
            d_model (int): The model dimension.
            max_len (int, optional): Maximum sequence length. Defaults to 5000.
        """
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PositionalEncoding module.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, d_model).

        Returns:
            torch.Tensor: Output tensor with positional encoding added.
        """
        return x + self.pe[:x.size(0)]

class EncoderBlock(nn.Module):
    def __init__(self, n_embed: int, n_head: int, n_hidden: int, dropout_p: float = 0.1):
        """
        Initialize the EncoderBlock module.

        Args:
            n_embed (int): The input embedding dimension.
            n_head (int): The number of attention heads.
            n_hidden (int): The hidden layer dimension in the MLP.
            dropout_p (float, optional): Dropout probability. Defaults to 0.1.
        """
        super(EncoderBlock, self).__init__()
        
        self.attn = SelfAttention(n_embed, n_head, dropout_p=dropout_p)
        self.mlp = MLP(n_embed, n_hidden, dropout_p=dropout_p)
        self.norm1 = nn.LayerNorm(n_embed)
        self.norm2 = nn.LayerNorm(n_embed)
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the EncoderBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_embed).
            mask (torch.Tensor, optional): Attention mask tensor of shape (batch_size, 1, 1, seq_len). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, n_embed).
        """
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm, mask=mask)  # Pass the mask
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)
        return self.dropout(x)

class Encoder(nn.Module):
    def __init__(self, n_embed: int, n_head: int, n_hidden: int, n_layers: int, vocab_size: int, 
                 max_len: int, dropout_p: float = 0.1):
        """
        Initialize the Encoder module.

        Args:
            n_embed (int): The input embedding dimension.
            n_head (int): The number of attention heads.
            n_hidden (int): The hidden layer dimension in the MLP.
            n_layers (int): The number of encoder layers.
            vocab_size (int): The size of the vocabulary.
            max_len (int): The maximum sequence length.
            dropout_p (float, optional): Dropout probability. Defaults to 0.1.
        """
        super(Encoder, self).__init__()
        
        self.layers = nn.ModuleList([
            EncoderBlock(n_embed, n_head, n_hidden, dropout_p) for _ in range(n_layers)
        ])
        self.pe = PositionalEncoding(n_embed, max_len)
        self.embed = nn.Embedding(vocab_size, n_embed)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the Encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).
            mask (torch.Tensor, optional): Attention mask tensor of shape (batch_size, 1, 1, seq_len). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, n_embed).
        """
        x = self.pe(self.embed(x))
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, n_embed: int, n_head: int, n_hidden: int, dropout_p: float = 0.1):
        """
        Initialize the DecoderBlock module.

        Args:
            n_embed (int): The input embedding dimension.
            n_head (int): The number of attention heads.
            n_hidden (int): The hidden layer dimension in the MLP.
            dropout_p (float, optional): Dropout probability. Defaults to 0.1.
        """
        super(DecoderBlock, self).__init__()
        
        self.attn1 = SelfAttention(n_embed, n_head, is_causal=True, dropout_p=dropout_p)
        self.attn2 = SelfAttention(n_embed, n_head)
        self.mlp = MLP(n_embed, n_hidden, dropout_p=dropout_p)
        self.norm1 = nn.LayerNorm(n_embed)
        self.norm2 = nn.LayerNorm(n_embed)
        self.norm3 = nn.LayerNorm(n_embed)
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, src_mask: torch.Tensor = None, trg_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the DecoderBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_embed).
            enc_output (torch.Tensor): Encoder output tensor of shape (batch_size, src_seq_len, n_embed).
            src_mask (torch.Tensor, optional): Source attention mask tensor of shape (batch_size, 1, 1, src_seq_len). Defaults to None.
            trg_mask (torch.Tensor, optional): Target attention mask tensor of shape (batch_size, 1, 1, seq_len). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, n_embed).
        """
        x_norm = self.norm1(x)
        x = x + self.attn1(x_norm, x_norm, x_norm, mask=trg_mask)  # Pass target mask
        x_norm = self.norm2(x)
        x = x + self.attn2(x_norm, enc_output, enc_output, mask=src_mask)  # Pass source mask
        x_norm = self.norm3(x)
        x = x + self.mlp(x_norm)
        return self.dropout(x)

class Decoder(nn.Module):
    def __init__(self, n_embed: int, n_head: int, n_hidden: int, n_layers: int, vocab_size: int, 
                 max_len: int, dropout_p: float = 0.1):
        """
        Initialize the Decoder module.

        Args:
            n_embed (int): The input embedding dimension.
            n_head (int): The number of attention heads.
            n_hidden (int): The hidden layer dimension in the MLP.
            n_layers (int): The number of decoder layers.
            vocab_size (int): The size of the vocabulary.
            max_len (int): The maximum sequence length.
            dropout_p (float, optional): Dropout probability. Defaults to 0.1.
        """
        super(Decoder, self).__init__()
        
        self.layers = nn.ModuleList([
            DecoderBlock(n_embed, n_head, n_hidden, dropout_p) for _ in range(n_layers)
        ])
        self.pe = PositionalEncoding(n_embed, max_len)
        self.embed = nn.Embedding(vocab_size, n_embed)
        
    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, src_mask: torch.Tensor = None, trg_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the Decoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).
            enc_output (torch.Tensor): Encoder output tensor of shape (batch_size, src_seq_len, n_embed).
            src_mask (torch.Tensor, optional): Source attention mask tensor of shape (batch_size, 1, 1, src_seq_len). Defaults to None.
            trg_mask (torch.Tensor, optional): Target attention mask tensor of shape (batch_size, 1, 1, seq_len). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, n_embed).
        """
        x = self.pe(self.embed(x))
        for layer in self.layers:
            x = layer(x, enc_output, src_mask=src_mask, trg_mask=trg_mask)
        return x

class Transformer(nn.Module):
    def __init__(self, n_embed: int, n_head: int, n_hidden: int, n_layers: int, vocab_size: int, 
                 max_len: int, pad_idx: int = 0, dropout_p: float = 0.1):
        """
        Initialize the Transformer module.

        Args:
            n_embed (int): The input embedding dimension.
            n_head (int): The number of attention heads.
            n_hidden (int): The hidden layer dimension in the MLP.
            n_layers (int): The number of encoder and decoder layers.
            vocab_size (int): The size of the vocabulary.
            max_len (int): The maximum sequence length.
            pad_idx (int, optional): The index used for padding. Defaults to 0.
            dropout_p (float, optional): Dropout probability. Defaults to 0.1.
        """
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(n_embed, n_head, n_hidden, n_layers, vocab_size, max_len, dropout_p)
        self.decoder = Decoder(n_embed, n_head, n_hidden, n_layers, vocab_size, max_len, dropout_p)
        self.output = nn.Linear(n_embed, vocab_size)
        self.norm = nn.LayerNorm(n_embed)
        self.pad_idx = pad_idx
    
    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer.

        Args:
            src (torch.Tensor): Source tensor of shape (batch_size, src_seq_len).
            trg (torch.Tensor): Target tensor of shape (batch_size, trg_seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, trg_seq_len, vocab_size).
        """
        # Create source padding mask
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, src_len)

        # Pass through encoder with source mask
        enc_output = self.encoder(src, mask=src_mask)

        # Create target padding mask
        trg_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, trg_len)

        # Pass through decoder with target and source masks
        dec_output = self.decoder(trg, enc_output, src_mask=src_mask, trg_mask=trg_mask)
        
        output = self.output(self.norm(dec_output))
        return output