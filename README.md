# Transformer Blocks

A modular collection of Transformer components for educational purposes.

## Overview

This repository provides the foundational building blocks of a Transformer model, as described in "Attention is All You Need" (Vaswani et al., 2017). These components can be assembled into full Transformer architectures (e.g., encoder-decoder models) and are ideal for learning about Transformer design and pre-training techniques.

### Components
- **`TokenEmbedding`**: Converts token IDs into dense embeddings.
- **`PositionalEncoding`**: Adds sinusoidal positional information to embeddings.
- **`MultiHeadAttention`**: Implements multi-head scaled dot-product attention.
- **`FullEncoder`**: Stacks multiple encoder layers for sequence encoding.
- **`FullDecoder`**: Stacks multiple decoder layers for sequence decoding.

Each module is implemented in PyTorch and documented with educational comments and docstrings.

## Purpose

The goal is to provide a clear, reusable set of Transformer blocks for:
- Understanding the inner workings of Transformers.
- Experimenting with pre-training strategies (e.g., Masked Language Modeling, sequence reconstruction).
- Building custom Transformer-based models.

This repo does not include training scripts or pre-trained weights, focusing instead on the architectural components and pre-training concepts.

---

## Pre-Training Guide

Pre-training a Transformer involves learning general representations from large datasets. Here’s how you might use these blocks for pre-training:

### 1. Data Preparation
- **Dataset**: Gather a large text corpus (e.g., Wikipedia, books).
- **Tokenization**: Convert text to token IDs using a vocabulary (e.g., 30,000 tokens) with special tokens (`<pad>`, `<unk>`).
- **Format**: Create batches of shape `[batch_size, seq_len]` (e.g., `[128, 128]`).

Example:
```python
import nltk
vocab = {"<pad>": 0, "<unk>": 1, "hello": 2, "world": 3}
text = "hello world"
tokens = nltk.word_tokenize(text.lower())
input_ids = [vocab.get(token, 1) for token in tokens]  # [2, 3]
```

### 2. Model Assembly
Assemble the Transformer blocks into an encoder-decoder architecture:
```python
import torch.nn as nn
from encoder_layer import FullEncoder
from decoder_layer import FullDecoder

class Transformer(nn.Module):
    def __init__(self, vocab_size=30000, d_model=256, num_heads=4, d_ff=512, num_layers=6):
        super().__init__()
        self.encoder = FullEncoder(vocab_size, d_model, num_heads, d_ff, num_layers)
        self.decoder = FullDecoder(vocab_size, d_model, num_heads, d_ff, num_layers)
    
    def forward(self, src, tgt):
        logits, enc_output = self.encoder(src)
        dec_output, _, _ = self.decoder(tgt, enc_output)
        return dec_output

model = Transformer()
```

- **Encoder**: Processes the source sequence into contextualized representations.
- **Decoder**: Reconstructs or generates a target sequence using encoder outputs.
- **Parameters**: `vocab_size`, `d_model`, `num_heads`, `d_ff`, and `num_layers` can be adjusted based on your needs.

### 3. Pre-Training Objectives
Define objectives to pre-train the model:
- **Encoder (Masked Language Modeling - MLM)**:
  - Randomly mask 15% of input tokens (e.g., replace with `<mask>` or a random token).
  - Predict the original tokens using the `mlm_head` in `FullEncoder`.
  - Loss: Cross-entropy over masked positions.
- **Decoder (Sequence Reconstruction)**:
  - Predict the full target sequence given the encoder’s output.
  - Loss: Cross-entropy over all positions.

Example:
```python
import torch
criterion = nn.CrossEntropyLoss(ignore_index=0)  # <pad> = 0
input_ids = torch.tensor([[2, 3, 0]])  # [batch_size, seq_len]
logits = model(input_ids, input_ids)   # [batch_size, seq_len, vocab_size]
loss = criterion(logits.view(-1, 30000), input_ids.view(-1))
```

- **MLM**: Pre-trains the encoder to understand context (e.g., BERT-like).
- **Reconstruction**: Pre-trains the decoder for sequence prediction (e.g., autoregressive tasks).

### 4. Training Tips
- **Batch Size**: Start with 128, adjust based on GPU memory.
- **Epochs**: 10-50, depending on dataset size and convergence.
- **Optimizer**: Use Adam with a learning rate of 0.001, optionally with a scheduler (e.g., cosine annealing).
- **Masking**: For MLM, mask tokens dynamically during training to improve robustness.
- **Hardware**: GPU recommended for efficiency (e.g., CUDA-enabled).

These steps provide a foundation for pre-training, which you can fine-tune for specific tasks (e.g., translation, generation) afterward.
