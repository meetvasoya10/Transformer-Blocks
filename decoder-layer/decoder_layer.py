import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from token_embedding import TokenEmbedding
from positional_encoding import PositionalEncoding
from multi_head_attention import MultiHeadAttention

class DecoderLayer(nn.Module):
    """
    A single layer of a Transformer decoder, implementing masked self-attention and encoder-decoder attention.

    This module applies masked multi-head self-attention, multi-head attention over encoder outputs,
    and a feed-forward network, with residual connections and normalization.

    Attributes:
        self_attn (MultiHeadAttention): Masked self-attention for target sequence.
        enc_dec_attn (MultiHeadAttention): Attention over encoder outputs.
        ffn (nn.Sequential): Feed-forward network with ReLU activation.
        norm1 (nn.LayerNorm): Normalization after self-attention.
        norm2 (nn.LayerNorm): Normalization after encoder-decoder attention.
        norm3 (nn.LayerNorm): Normalization after FFN.
        dropout (nn.Dropout): Dropout layer for regularization.

    Args:
        d_model (int): Dimensionality of the input/output (e.g., 256, 512).
        num_heads (int): Number of attention heads (e.g., 4, 8).
        d_ff (int): Dimensionality of the FFN’s hidden layer (e.g., 512, 2048).
        dropout (float, optional): Dropout probability (default: 0.1).
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),  # ReLU (can be swapped with GELU)
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, enc_output: torch.Tensor, 
                tgt_mask: Optional[torch.Tensor] = None, 
                memory_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the decoder layer.

        Args:
            tgt (torch.Tensor): Target sequence [batch_size, tgt_seq_len, d_model].
            enc_output (torch.Tensor): Encoder output [batch_size, src_seq_len, d_model].
            tgt_mask (torch.Tensor, optional): Mask for target [batch_size, 1, tgt_seq_len, tgt_seq_len].
            memory_mask (torch.Tensor, optional): Mask for encoder [batch_size, 1, tgt_seq_len, src_seq_len].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (output, self_attn, enc_dec_attn)
                - output: [batch_size, tgt_seq_len, d_model].
                - self_attn: [batch_size, num_heads, tgt_seq_len, tgt_seq_len].
                - enc_dec_attn: [batch_size, num_heads, tgt_seq_len, src_seq_len].
        """
        self_attn_output, self_attn = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(self_attn_output))
        enc_dec_output, enc_dec_attn = self.enc_dec_attn(tgt, enc_output, enc_output, memory_mask)
        tgt = self.norm2(tgt + self.dropout(enc_dec_output))
        ffn_output = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout(ffn_output))
        return tgt, self_attn, enc_dec_attn

class FullDecoder(nn.Module):
    """
    A complete Transformer decoder with multiple stacked DecoderLayers.

    This module processes target sequences through token embedding, positional encoding, and a stack
    of decoder layers, producing output logits. It’s designed for pre-training or fine-tuning tasks
    like sequence generation or reconstruction, often paired with a FullEncoder.

    Attributes:
        token_emb (TokenEmbedding): Converts target token IDs to embeddings.
        pos_enc (PositionalEncoding): Adds positional information.
        layers (nn.ModuleList): Stack of DecoderLayer instances.
        dropout (nn.Dropout): Dropout for regularization.
        fc_out (nn.Linear): Final linear layer to produce logits over vocabulary.

    Args:
        vocab_size (int): Size of the vocabulary (e.g., 30000).
        d_model (int): Dimensionality of embeddings and model (e.g., 256, 512).
        num_heads (int): Number of attention heads per layer (e.g., 4, 8).
        d_ff (int): Dimensionality of the FFN’s hidden layer (e.g., 512, 2048).
        num_layers (int): Number of stacked decoder layers (e.g., 6).
        dropout (float, optional): Dropout probability (default: 0.1).
    """
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ff: int, num_layers: int, 
                 dropout: float = 0.1):
        super(FullDecoder, self).__init__()
        self.token_emb = TokenEmbedding(vocab_size=vocab_size, d_model=d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model=d_model, max_seq_len=5000)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt: torch.Tensor, enc_output: torch.Tensor, 
                tgt_mask: Optional[torch.Tensor] = None, 
                memory_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through the full decoder.

        Args:
            tgt (torch.Tensor): Target sequence of token IDs [batch_size, tgt_seq_len].
            enc_output (torch.Tensor): Encoder output [batch_size, src_seq_len, d_model].
            tgt_mask (torch.Tensor, optional): Mask for target [batch_size, 1, tgt_seq_len, tgt_seq_len].
            memory_mask (torch.Tensor, optional): Mask for encoder [batch_size, 1, tgt_seq_len, src_seq_len].

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]: (logits, self_attns, enc_dec_attns)
                - logits: Predictions [batch_size, tgt_seq_len, vocab_size].
                - self_attns: List of self-attention weights [num_layers, batch_size, num_heads, tgt_seq_len, tgt_seq_len].
                - enc_dec_attns: List of encoder-decoder attention weights [num_layers, batch_size, num_heads, tgt_seq_len, src_seq_len].
        """
        # Embed target tokens and add positional encodings
        tgt = self.token_emb(tgt)  # [batch_size, tgt_seq_len, d_model]
        tgt = self.pos_enc(tgt)
        
        # Stack decoder layers
        self_attns, enc_dec_attns = [], []
        for layer in self.layers:
            tgt, self_attn, enc_dec_attn = layer(tgt, enc_output, tgt_mask, memory_mask)
            self_attns.append(self_attn)
            enc_dec_attns.append(enc_dec_attn)
        
        # Final output layer
        output = self.fc_out(tgt)  # [batch_size, tgt_seq_len, vocab_size]
        return output, self_attns, enc_dec_attns