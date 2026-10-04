import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism as described in the Transformer paper (Vaswani et al., 2017).

    This module splits the input into multiple attention heads, computes scaled dot-product
    attention for each head, and concatenates the results. It’s a key building block for
    self-attention in the encoder and encoder-decoder attention in the decoder of a Transformer.
    During pre-training, it enables the model to capture diverse relationships within sequences.

    Attributes:
        d_model (int): Total dimensionality of the input and output.
        num_heads (int): Number of attention heads.
        d_k (int): Dimensionality of each head (d_model // num_heads).
        W_q (nn.Linear): Linear layer for queries.
        W_k (nn.Linear): Linear layer for keys.
        W_v (nn.Linear): Linear layer for values.
        W_o (nn.Linear): Linear layer for output projection.

    Args:
        d_model (int): Dimensionality of the input/output (e.g., 256, 512).
        num_heads (int): Number of attention heads (e.g., 4, 8).
    """
    def __init__(self, d_model: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        # Ensure d_model is divisible by num_heads for even splitting
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head

        # Linear transformations for queries, keys, values, and output
        self.W_q = nn.Linear(d_model, d_model, bias=False)  # Query projection
        self.W_k = nn.Linear(d_model, d_model, bias=False)  # Key projection
        self.W_v = nn.Linear(d_model, d_model, bias=False)  # Value projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)  # Output projection

    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                                   mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention for a single head or across multiple heads.

        Args:
            Q (torch.Tensor): Queries tensor of shape [batch_size, num_heads, seq_len, d_k].
            K (torch.Tensor): Keys tensor of shape [batch_size, num_heads, seq_len, d_k].
            V (torch.Tensor): Values tensor of shape [batch_size, num_heads, seq_len, d_k].
            mask (torch.Tensor, optional): Mask tensor of shape [batch_size, num_heads, seq_len, seq_len]
                                          to prevent attention to certain positions (e.g., padding).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (output, attention_weights)
                - output: Attention-weighted values [batch_size, num_heads, seq_len, d_k].
                - attn: Attention weights [batch_size, num_heads, seq_len, seq_len].
        """
        d_k = Q.size(-1)
        # Compute attention scores: (Q * K^T) / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # [batch_size, num_heads, seq_len, seq_len]
        
        # Apply mask (if provided) to prevent attending to padded or future tokens
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # -1e9 ensures masked positions get near-zero weight
        
        # Softmax to get attention weights
        attn = torch.softmax(scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
        # Apply attention weights to values
        output = torch.matmul(attn, V)  # [batch_size, num_heads, seq_len, d_k]
        return output, attn

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform multi-head attention on queries, keys, and values.

        Args:
            Q (torch.Tensor): Queries tensor of shape [batch_size, seq_len, d_model].
            K (torch.Tensor): Keys tensor of shape [batch_size, seq_len, d_model].
            V (torch.Tensor): Values tensor of shape [batch_size, seq_len, d_model].
            mask (torch.Tensor, optional): Mask tensor of shape [batch_size, 1, seq_len, seq_len]
                                          or broadcastable shape to prevent attention to certain positions.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (output, attention_weights)
                - output: Attention output [batch_size, seq_len, d_model].
                - attn: Attention weights [batch_size, num_heads, seq_len, seq_len].
        """
        batch_size = Q.size(0)
        
        # Project queries, keys, and values
        Q = self.W_q(Q)  # [batch_size, seq_len, d_model]
        K = self.W_k(K)  # [batch_size, seq_len, d_model]
        V = self.W_v(V)  # [batch_size, seq_len, d_model]
        
        # Split into multiple heads: [batch_size, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute scaled dot-product attention for all heads
        output, attn = self.scaled_dot_product_attention(Q, K, V, mask)  # [batch_size, num_heads, seq_len, d_k]
        
        # Concatenate heads and project back to d_model
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # [batch_size, seq_len, d_model]
        output = self.W_o(output)  # Final projection: [batch_size, seq_len, d_model]
        
        return output, attn