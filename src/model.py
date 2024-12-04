import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

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

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 5000):
        """
        Initialize the RotaryEmbedding module.

        Args:
            dim (int): The dimension of the embeddings (must be divisible by 2).
            max_len (int, optional): Maximum sequence length. Defaults to 5000.
        """
        super().__init__()
        
        assert dim % 2 == 0, "Dimension must be divisible by 2"
        
        # Create position indices tensor
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Create position sequence
        position = torch.arange(max_len).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", position, self.inv_freq)  # [max_len, dim/2]
        
        # Create rotation matrices
        emb = torch.cat((freqs, freqs), dim=-1)  # [max_len, dim]
        
        # Reshape the embeddings correctly
        cos = emb.cos().view(max_len, 1, dim)  # [max_len, 1, dim]
        sin = emb.sin().view(max_len, 1, dim)  # [max_len, 1, dim]
        
        self.register_buffer("cos_cached", cos.unsqueeze(0))  # [1, max_len, 1, dim]
        self.register_buffer("sin_cached", sin.unsqueeze(0))  # [1, max_len, 1, dim]
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def _apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary positional embeddings to the query and key tensors.
        
        Args:
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            cos (torch.Tensor): Cosine part of rotary embedding.
            sin (torch.Tensor): Sine part of rotary embedding.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Rotated query and key tensors.
        """
        # Reshape cos and sin to match q and k dimensions [batch, head, seq, dim]
        cos = cos.view(1, 1, -1, cos.size(-1))  # [1, 1, seq, dim]
        sin = sin.view(1, 1, -1, sin.size(-1))  # [1, 1, seq, dim]
        
        return (
            q * cos + self._rotate_half(q) * sin,
            k * cos + self._rotate_half(k) * sin
        )
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to query and key tensors.

        Args:
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            seq_len (int): Length of the sequence.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Rotated query and key tensors.
        """
        cos = self.cos_cached[:, :seq_len, ...]
        sin = self.sin_cached[:, :seq_len, ...]
        return self._apply_rotary_pos_emb(q, k, cos, sin)

class SelfAttention(nn.Module):
    def __init__(self, n_embed: int, n_head: int, max_len: int = 512, is_causal: bool = False, dropout_p: float = 0.1):
        """
        Initialize the SelfAttention module.

        Args:
            n_embed (int): The input embedding dimension.
            n_head (int): The number of attention heads.
            max_len (int, optional): Maximum sequence length. Defaults to 512.
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
        
        # Pass max_len to RotaryEmbedding
        self.rope = RotaryEmbedding(self.head_dim, max_len=max_len)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the SelfAttention module.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, seq_len, n_embed).
            k (torch.Tensor): Key tensor of shape (batch_size, seq_len, n_embed).
            v (torch.Tensor): Value tensor of shape (batch_size, seq_len, n_embed).
            mask (torch.Tensor, optional): Attention mask tensor of shape (batch_size, 1, 1, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, n_embed).
        """
        if not all(x.size(-1) == q.size(-1) for x in [k, v]):
            raise ValueError("All inputs must have the same embedding dimension")
        
        batch_size, seq_len, _ = q.size()
        
        # Linear projections and reshape
        q = self.W_q(q).reshape(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)
        k = self.W_k(k).reshape(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)
        v = self.W_v(v).reshape(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        q, k = self.rope(q, k, seq_len)
        
        # Use PyTorch's optimized attention
        attention = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0
        )
        
        # Reshape for output
        attention = attention.transpose(1, 2).reshape(batch_size, seq_len, self.n_embed)
        
        return self.output(attention)

class EncoderBlock(nn.Module):
    def __init__(self, n_embed: int, n_head: int, n_hidden: int, max_len: int = 512, dropout_p: float = 0.1):
        """
        Initialize the EncoderBlock module.

        Args:
            n_embed (int): The input embedding dimension.
            n_head (int): The number of attention heads.
            n_hidden (int): The hidden layer dimension in the MLP.
            max_len (int, optional): Maximum sequence length. Defaults to 512.
            dropout_p (float, optional): Dropout probability. Defaults to 0.1.
        """
        super(EncoderBlock, self).__init__()
        
        # Pass max_len to SelfAttention
        self.attn = SelfAttention(n_embed, n_head, max_len=max_len, dropout_p=dropout_p)
        self.mlp = MLP(n_embed, n_hidden, dropout_p=dropout_p)
        self.norm1 = nn.LayerNorm(n_embed)
        self.norm2 = nn.LayerNorm(n_embed)
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the EncoderBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_embed).
            mask (torch.Tensor, optional): Attention mask tensor of shape (batch_size, 1, 1, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, n_embed).
        """
        # Self-attention block with single pre-norm
        norm_x = self.norm1(x)
        x = x + self.attn(norm_x, norm_x, norm_x, mask=mask)
        
        # MLP block with pre-norm
        norm_x = self.norm2(x)
        x = x + self.mlp(norm_x)
        
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
        
        # Pass max_len to EncoderBlock
        self.layers = nn.ModuleList([
            EncoderBlock(n_embed, n_head, n_hidden, max_len=max_len, dropout_p=dropout_p) for _ in range(n_layers)
        ])
        self.embed = nn.Embedding(vocab_size, n_embed)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the Encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).
            mask (torch.Tensor, optional): Attention mask tensor of shape (batch_size, 1, 1, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, n_embed).
        """
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, n_embed: int, n_head: int, n_hidden: int, max_len: int = 512, dropout_p: float = 0.1):
        """
        Initialize the DecoderBlock module.

        Args:
            n_embed (int): The input embedding dimension.
            n_head (int): The number of attention heads.
            n_hidden (int): The hidden layer dimension in the MLP.
            max_len (int, optional): Maximum sequence length. Defaults to 512.
            dropout_p (float, optional): Dropout probability. Defaults to 0.1.
        """
        super(DecoderBlock, self).__init__()
        
        self.attn1 = SelfAttention(n_embed, n_head, max_len=max_len, is_causal=True, dropout_p=dropout_p)
        self.attn2 = SelfAttention(n_embed, n_head, max_len=max_len, dropout_p=dropout_p)
        self.mlp = MLP(n_embed, n_hidden, dropout_p=dropout_p)
        self.norm1 = nn.LayerNorm(n_embed)
        self.norm2 = nn.LayerNorm(n_embed)
        self.norm3 = nn.LayerNorm(n_embed)
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, 
                src_mask: torch.Tensor = None, trg_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the DecoderBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_embed).
            enc_output (torch.Tensor): Encoder output tensor of shape (batch_size, src_seq_len, n_embed).
            src_mask (torch.Tensor, optional): Source attention mask tensor of shape (batch_size, 1, 1, src_seq_len). 
            Defaults to None.
            trg_mask (torch.Tensor, optional): Target attention mask tensor of shape (batch_size, 1, 1, seq_len). 
            Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, n_embed).
        """
        # Self-attention block with single pre-norm
        norm_x = self.norm1(x)
        x = x + self.attn1(norm_x, norm_x, norm_x, mask=trg_mask)
        
        # Cross-attention block with pre-norm
        norm_x = self.norm2(x)
        x = x + self.attn2(norm_x, enc_output, enc_output, mask=src_mask)
        
        # MLP block with pre-norm
        norm_x = self.norm3(x)
        x = x + self.mlp(norm_x)
        
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
            DecoderBlock(n_embed, n_head, n_hidden, max_len=max_len, dropout_p=dropout_p) 
            for _ in range(n_layers)
        ])
        self.embed = nn.Embedding(vocab_size, n_embed)
        
    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, 
                src_mask: torch.Tensor = None, trg_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the Decoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).
            enc_output (torch.Tensor): Encoder output tensor of shape (batch_size, src_seq_len, n_embed).
            src_mask (torch.Tensor, optional): Source attention mask tensor of shape (batch_size, 1, 1, src_seq_len).
            trg_mask (torch.Tensor, optional): Target attention mask tensor of shape (batch_size, 1, 1, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, n_embed).
        """
        x = self.embed(x)
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
        src_mask, trg_mask = self.create_masks(src, trg)

        # Pass through encoder and decoder with appropriate masks
        enc_output = self.encoder(src, mask=src_mask)
        dec_output = self.decoder(trg, enc_output, src_mask=src_mask, trg_mask=trg_mask)
        
        output = self.output(self.norm(dec_output))
        return output
    
    
    def create_masks(self, src: torch.Tensor, trg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create source and target masks for the Transformer.

        Args:
            src (torch.Tensor): Source tensor of shape (batch_size, src_seq_len).
            trg (torch.Tensor): Target tensor of shape (batch_size, trg_seq_len).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Source mask and target mask.
        """
        # Create source padding mask more efficiently
        src_mask = (src != self.pad_idx)[:, None, None, :]
        
        # Create target masks more efficiently
        trg_pad_mask = (trg != self.pad_idx)[:, None, None, :]
        
        # Create causal mask once during initialization and cache it
        if not hasattr(self, '_causal_mask') or self._causal_mask.size(2) < trg.size(1):
            seq_len = trg.size(1)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            self._causal_mask = causal_mask[None, None, :, :].to(trg.device)
        
        # Use the cached causal mask
        trg_mask = trg_pad_mask & ~self._causal_mask[:, :, :trg.size(1), :trg.size(1)]
        
        return src_mask, trg_mask