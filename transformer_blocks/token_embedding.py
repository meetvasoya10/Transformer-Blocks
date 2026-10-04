import torch
import torch.nn as nn
import math
from typing import Optional

class TokenEmbedding(nn.Module):
    """
    A module to convert token indices into dense embeddings for Transformer models.

    This class maps integer token IDs to vectors of size `d_model`, scaling the output by
    sqrt(d_model) as per the Transformer paper (Vaswani et al., 2017). It’s typically the
    first step in a Transformer pipeline, preparing input tokens for positional encoding
    and attention layers.

    Attributes:
        embedding (nn.Embedding): The embedding layer mapping tokens to vectors.
        d_model (int): Dimensionality of the embedding vectors.
        padding_idx (Optional[int]): Index of the padding token to ignore in gradients.
        device (torch.device): Device on which the embeddings are stored.

    Args:
        vocab_size (int): Size of the vocabulary (number of unique tokens).
        d_model (int): Dimensionality of the output embeddings (e.g., 256, 512).
        padding_idx (Optional[int]): Token ID to be treated as padding (default: None).
        device (Optional[torch.device]): Device to place the embeddings on (default: CPU).
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: Optional[int] = None,
        device: Optional[torch.device] = None
    ):
        super(TokenEmbedding, self).__init__()
        # Input validation for robustness and clarity
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            raise ValueError(f"vocab_size must be a positive integer, got {vocab_size}")
        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError(f"d_model must be a positive integer, got {d_model}")

        # Define the embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx,  # Ensures padding tokens don’t affect gradients
            device=device
        )
        self.d_model = d_model
        self.padding_idx = padding_idx
        self.device = device or torch.device("cpu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform token indices into scaled embedding vectors.

        Args:
            x (torch.Tensor): Input tensor of token IDs with shape [batch_size, seq_len].

        Returns:
            torch.Tensor: Embedded tensor of shape [batch_size, seq_len, d_model], scaled
                          by sqrt(d_model) to stabilize training.

        Raises:
            ValueError: If input is not a tensor, has wrong dtype, or contains invalid indices.
        """
        # Validate input tensor
        if not torch.is_tensor(x):
            raise ValueError("Input must be a PyTorch tensor")
        if x.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"Input tensor must be of integer type, got {x.dtype}")
        if x.min() < 0 or x.max() >= self.embedding.num_embeddings:
            raise ValueError(
                f"Token indices must be in range [0, {self.embedding.num_embeddings - 1}], "
                f"got min {x.min()} and max {x.max()}"
            )
        
        # Embed tokens and scale by sqrt(d_model) as per Transformer convention
        embedded = self.embedding(x)
        return embedded * math.sqrt(self.d_model)

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of the embeddings (d_model)."""
        return self.d_model

    @property
    def vocabulary_size(self) -> int:
        """Return the size of the vocabulary (number of unique tokens)."""
        return self.embedding.num_embeddings