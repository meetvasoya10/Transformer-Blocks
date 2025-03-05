import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from token_embedding import TokenEmbedding
from positional_encoding import PositionalEncoding
from multi_head_attention import MultiHeadAttention

class EncoderLayer(nn.Module):
    """
    A single layer of a Transformer encoder, implementing self-attention and a feed-forward network.

    This module applies multi-head self-attention followed by a position-wise feed-forward network,
    with residual connections and layer normalization. It’s designed to be stacked in a FullEncoder.

    Attributes:
        self_attn (MultiHeadAttention): Multi-head self-attention mechanism.
        ffn (nn.Sequential): Feed-forward network with GELU activation.
        norm1 (nn.LayerNorm): Layer normalization after self-attention.
        norm2 (nn.LayerNorm): Layer normalization after FFN.
        dropout (nn.Dropout): Dropout layer for regularization.

    Args:
        d_model (int): Dimensionality of the input/output (e.g., 256, 512).
        num_heads (int): Number of attention heads (e.g., 4, 8).
        d_ff (int): Dimensionality of the FFN’s hidden layer (e.g., 512, 2048).
        dropout (float, optional): Dropout probability (default: 0.1).
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # GELU for smoother gradients (common in modern Transformers)
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder layer.

        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, d_model].
            mask (torch.Tensor, optional): Mask tensor [batch_size, 1, seq_len, seq_len].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (output, attention_weights)
                - output: [batch_size, seq_len, d_model].
                - attn: [batch_size, num_heads, seq_len, seq_len].
        """
        attn_output, attn = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x, attn
    
    def _init_weights(self):
        """Initialize weights with Xavier normal for linear layers and constants for normalization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)

class FullEncoder(nn.Module):
    """
    A complete Transformer encoder with multiple stacked EncoderLayers.

    This module processes input sequences through token embedding, positional encoding, and
    a stack of encoder layers, producing contextualized representations. It’s commonly pre-trained
    with objectives like Masked Language Modeling (MLM) to learn general sequence understanding.

    Attributes:
        token_emb (TokenEmbedding): Converts token IDs to embeddings.
        pos_enc (PositionalEncoding): Adds positional information.
        layers (nn.ModuleList): Stack of EncoderLayer instances.
        dropout (nn.Dropout): Dropout for regularization.
        mlm_head (nn.Linear): Linear layer for MLM pre-training output.

    Args:
        vocab_size (int): Size of the vocabulary (e.g., 30000).
        d_model (int): Dimensionality of embeddings and model (e.g., 256, 512).
        num_heads (int): Number of attention heads per layer (e.g., 4, 8).
        d_ff (int): Dimensionality of the FFN’s hidden layer (e.g., 512, 2048).
        num_layers (int): Number of stacked encoder layers (e.g., 6).
        dropout (float, optional): Dropout probability (default: 0.1).
    """
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ff: int, num_layers: int, 
                 dropout: float = 0.1):
        super().__init__()
        self.token_emb = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.pos_enc = PositionalEncoding(d_model=d_model, max_seq_len=512)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        # MLM head for pre-training (can be omitted if not pre-training for MLM)
        self.mlm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the full encoder.

        Args:
            x (torch.Tensor): Input tensor of token IDs [batch_size, seq_len].
            mask (torch.Tensor, optional): Mask tensor [batch_size, 1, seq_len, seq_len].

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: (logits, encoded_output)
                - logits: MLM predictions [batch_size, seq_len, vocab_size].
                - encoded_output: Contextualized representations [batch_size, seq_len, d_model].
                - attns: List of attention weights from each layer [num_layers, batch_size, num_heads, seq_len, seq_len].
        """
        # Embed tokens and add positional encodings
        x = self.token_emb(x)  # [batch_size, seq_len, d_model]
        x = self.pos_enc(x)
        x = self.dropout(x)
        
        # Stack encoder layers
        attns = []
        for layer in self.layers:
            x, attn = layer(x, mask)
            attns.append(attn)
        
        # MLM head for pre-training output
        logits = self.mlm_head(x)  # [batch_size, seq_len, vocab_size]
        return logits, x  # Return both logits and encoded output for flexibility