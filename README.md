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
