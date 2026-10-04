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

Each module is implemented in PyTorch and documented with educational comments and docstrings. All of them are importable from the `transformer_blocks` package.

## Purpose

The goal is to provide a clear, reusable set of Transformer blocks for:
- Understanding the inner workings of Transformers.
- Experimenting with pre-training strategies (e.g., Masked Language Modeling, sequence reconstruction).
- Building custom Transformer-based models.

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.10.

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
from transformer_blocks import FullEncoder, FullDecoder

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
  - Randomly mask 10% of input tokens, replacing each with `<mask>` (no random-token or keep-original replacement).
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

## Pre-training

The encoder is pre-trained with masked language modeling. In the training code, 10% of the tokens in each sample are replaced with `<mask>` (no random-token or keep-original replacement), and the model predicts the original tokens at those positions. Padding and `<unk>` tokens are never masked.

### 1. Prepare the data

```bash
python scripts/prepare_data.py --out data/cleaned_wiki_corpus.txt
```

This downloads the ConvoKit `wiki-corpus` and writes the cleaned text, one entry per line.

### 2. Train

Place `cot1.json` and `cot2.json` under `data/`, then run:

```bash
python scripts/train_encoder.py --wiki data/cleaned_wiki_corpus.txt --cot1 data/cot1.json --cot2 data/cot2.json --out checkpoints/pretrained_encoder.pth
```

All four arguments are optional and default to the paths shown. A Hugging Face token can be supplied through the `HF_TOKEN` environment variable (or a `.env` file); it is optional.

### Data sources

1. Cleaned ConvoKit `wiki-corpus` (Wikipedia talk pages), produced by `scripts/prepare_data.py`
2. `cot1.json`: chain-of-thought examples, using the `output` field
3. `cot2.json`: chain-of-thought examples, using the `output` field
4. `PrimeIntellect/verifiable-coding-problems` from Hugging Face (`prompt` field, first 100,000 examples)
5. `Salesforce/wikitext`, `wikitext-103-raw-v1` train split (first 500,000 non-empty examples)

The sources are concatenated and de-duplicated before training.

### Configuration

| Setting | Value |
|---------|-------|
| Vocabulary | Word-level (NLTK `word_tokenize`), 30,000 tokens |
| `d_model` | 256 |
| Attention heads | 4 |
| `d_ff` | 512 |
| Encoder layers | 6 |
| Batch size | 128 |
| Sequence length | 128 |
| Optimizer | Adam, learning rate 1e-3 |
| Scheduler | `ReduceLROnPlateau` |
| Precision | FP16 (mixed precision, CUDA) |

### Reference run

- 939,124 training samples
- 10 epochs
- About 90 minutes per epoch

## Tests

```bash
pytest tests/ -q
```
