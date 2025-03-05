import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Adds positional encodings to token embeddings in Transformer models.

    This module implements the sinusoidal positional encoding from the Transformer paper
    (Vaswani et al., 2017). It generates fixed positional encodings based on sine and cosine
    functions, which are added to token embeddings to provide sequence order information.
    These encodings are pre-computed and stored as a buffer, making it efficient for inference
    and training.

    Attributes:
        pe (torch.Tensor): Pre-computed positional encoding tensor of shape
                           [1, max_seq_len, d_model].

    Args:
        d_model (int): Dimensionality of the model (must match token embedding size).
        max_seq_len (int, optional): Maximum sequence length supported (default: 5000).
    """
    def __init__(self, d_model: int, max_seq_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        # Validate inputs
        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError(f"d_model must be a positive integer, got {d_model}")
        if not isinstance(max_seq_len, int) or max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be a positive integer, got {max_seq_len}")

        # Pre-compute positional encodings
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # [max_seq_len, 1]
        # Divisor term for sinusoidal functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_seq_len, d_model]
        
        # Register as a buffer (not a parameter, so it’s not updated during training)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encodings to the input embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model], typically
                              the output of a TokenEmbedding layer.

        Returns:
            torch.Tensor: Tensor with positional encodings added, same shape as input
                          [batch_size, seq_len, d_model].

        Raises:
            ValueError: If sequence length exceeds max_seq_len.
        """
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.pe.size(1)}")
        
        # Add positional encodings to the input tensor
        x = x + self.pe[:, :seq_len]  # Broadcasting over batch dimension
        return x