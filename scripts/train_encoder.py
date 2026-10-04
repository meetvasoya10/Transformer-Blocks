#!/usr/bin/env python3
# train_encoder.py - Pre-train FullEncoder on cleaned_wiki_corpus.txt, CoT JSON files, and HF datasets with MLM, NLTK tokenization, FP16, minimal output, and dynamic ReduceLROnPlateau scheduler with error monitoring

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import argparse
import os
import nltk
import json
import sys
from collections import Counter
from typing import Tuple, List, Dict
from datasets import load_dataset
from torch.amp import GradScaler, autocast
from huggingface_hub import login
from tqdm import tqdm
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dotenv import load_dotenv

# Enable synchronous CUDA operations for debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

load_dotenv()

# Configure detailed logging for error monitoring
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="training_log.log"  # Log to a file for review
)
logger = logging.getLogger(__name__)

# Ensure NLTK punkt is downloaded
try:
    nltk.data.find('tokenizers/punkt')
    logger.info("NLTK punkt tokenizer already downloaded")
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')
    logger.info("NLTK punkt tokenizer downloaded successfully")

# Import FullEncoder and related utilities from full_encoder.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformer_blocks import FullEncoder, TokenEmbedding

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

class MLMPretrainingDataset(Dataset):
    def __init__(self, texts: List[str], vocab: dict, max_len: int = 128, mlm_prob: float = 0.10):
        self.texts = texts
        self.vocab = vocab
        self.max_len = max_len
        self.mlm_prob = mlm_prob
        self.pad_idx = vocab["<pad>"]  # Padding index (e.g., 0)
        self.unk_idx = vocab["<unk>"]  # Unknown token index (e.g., 1)
        self.mask_idx = vocab["<mask>"] if "<mask>" in vocab else max(vocab.values()) + 1  # Mask token index (e.g., 2)

        # Verify vocabulary indices are within bounds
        vocab_size = len(vocab)
        if self.pad_idx >= vocab_size or self.unk_idx >= vocab_size or self.mask_idx >= vocab_size:
            logger.error(f"Vocabulary indices out of bounds: pad_idx={self.pad_idx}, unk_idx={self.unk_idx}, mask_idx={self.mask_idx}, vocab_size={vocab_size}")
            raise ValueError(f"Vocabulary indices out of bounds: pad_idx={self.pad_idx}, unk_idx={self.unk_idx}, mask_idx={self.mask_idx}, vocab_size={vocab_size}")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            # Tokenization and initial padding handling
            tokens = tokenize_text(self.texts[idx])
            if not tokens:
                return (
                    torch.full((self.max_len,), self.pad_idx, dtype=torch.long),
                    torch.full((self.max_len,), -100, dtype=torch.long)
                )

            # Convert tokens to indices + padding
            input_ids = [self.vocab.get(token, self.unk_idx) for token in tokens[:self.max_len]]
            if len(input_ids) < self.max_len:
                input_ids += [self.pad_idx] * (self.max_len - len(input_ids))
            
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            labels = input_ids.clone()

            # Check if all tokens are invalid (pad/unk)
            non_pad_unk_mask = (input_ids != self.pad_idx) & (input_ids != self.unk_idx)
            if non_pad_unk_mask.sum().item() == 0:  # All tokens are pad/unk
                return input_ids, torch.full_like(input_ids, -100)

            # Validate indices
            vocab_size = len(self.vocab)
            if (input_ids < 0).any() or (input_ids >= vocab_size).any():
                raise ValueError(f"Invalid input_ids at index {idx}")

            # MLM masking logic
            rand = torch.rand(input_ids.shape)
            non_pad_unk = (input_ids != self.pad_idx) & (input_ids != self.unk_idx)
            mask_arr = (rand < self.mlm_prob) & non_pad_unk

            # Handle full masking case
            if mask_arr.sum() == non_pad_unk.sum():
                valid_indices = torch.nonzero(non_pad_unk).squeeze()
                if valid_indices.numel() == 0:  # Should never happen due to earlier check
                    return input_ids, torch.full_like(input_ids, -100)
                
                # Safely get keep index
                keep_idx = valid_indices[0].item() if valid_indices.dim() > 0 else valid_indices.item()
                mask_arr[keep_idx] = False

            input_ids[mask_arr] = self.mask_idx
            labels[~mask_arr] = -100

            return input_ids, labels
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {str(e)}")
            # Return safe fallback
            return (
                torch.full((self.max_len,), self.pad_idx, dtype=torch.long),
                torch.full((self.max_len,), -100, dtype=torch.long)
            )

        
def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text using NLTK's word_tokenize, handling concatenated words and whitespace, with error logging.
    """
    try:
        if not text or not isinstance(text, str):
            return []
        tokens = nltk.word_tokenize(text.replace('\n', ' ').strip())
        return [token for token in tokens if token and not token.isspace()]  # Filter out empty or whitespace tokens
    except Exception as e:
        logger.error(f"Error tokenizing text: {str(e)}")
        return []

def load_cleaned_corpus(file_path: str = 'cleaned_wiki_corpus.txt') -> List[str]:
    """
    Load local cleaned Wikipedia corpus, filtering empty or whitespace-only lines, with error logging.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(texts)} entries from {file_path}")
        return texts
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        raise

def load_cot_files(file_paths: List[str]) -> List[str]:
    """
    Load CoT1 and CoT2 JSON files, extracting non-empty text from 'output' fields, with error logging.
    """
    combined_texts = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    outputs = [example["output"] for example in data if "output" in example and example["output"].strip()]
                    combined_texts.extend(outputs)
                else:
                    output = data.get("output", "")
                    if output.strip():
                        combined_texts.append(output)
            logger.info(f"Loaded {len(outputs if isinstance(data, list) else 1 if output.strip() else 0)} entries from {file_path}")
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise
    return [text.strip() for text in combined_texts if text.strip()]

def load_huggingface_corpora(token: str = None) -> List[str]:
    """
    Load data from Hugging Face datasets (verifiable-coding-problems, wikitext, excluding Wikipedia), filtering empty texts, with error logging.
    """
    try:
        datasets = {
            "coding": load_dataset("PrimeIntellect/verifiable-coding-problems", token=token),
            "wikitext": load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", token=token)
        }
        
        combined_texts = []
        for name, dataset in datasets.items():
            if name == "coding":
                texts = [example["prompt"] for example in dataset["train"] if example["prompt"].strip()][:100000]
            else:  # wikitext
                texts = [example["text"] for example in dataset["train"] if example["text"].strip()][:500000]
            combined_texts.extend(texts)
            logger.info(f"Sampled {len(texts)} examples from {name} (out of total available)")
        
        return combined_texts
    except Exception as e:
        logger.error(f"Error loading Hugging Face datasets: {str(e)}")
        raise

def build_vocab(texts: List[str], max_vocab_size: int) -> Dict[str, int]:
    """
    Build vocabulary from texts, capped at max_vocab_size, reserving indices for special tokens, with error logging.
    """
    try:
        token_counts = Counter()
        for i, text in enumerate(texts):
            tokens = tokenize_text(text)
            if not tokens:
                logger.warning(f"Empty tokens for text at index {i} in vocabulary building")
            token_counts.update(tokens)

        frequent_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:max_vocab_size - 3]  # Reserve for <pad>, <unk>, <mask>
        vocab = {"<pad>": 0, "<unk>": 1, "<mask>": 2}
        for token, _ in frequent_tokens:
            vocab[token] = len(vocab)
        logger.info(f"Built vocabulary with {len(vocab)} tokens (capped at {max_vocab_size})")
        return vocab
    except Exception as e:
        logger.error(f"Error building vocabulary: {str(e)}")
        raise

def prepare_data(batch_size: int, seq_len: int, device: torch.device, texts: List[str]) -> Tuple[DataLoader, Dict[str, int]]:
    """
    Prepare data for masked language modeling (MLM), filtering short texts, with error logging.
    """
    try:
        # Filter out texts with fewer than 2 tokens
        def filter_short_texts(texts: List[str], min_tokens: int = 2) -> List[str]:
            filtered = [text for text in texts if len(tokenize_text(text)) >= min_tokens]
            logger.info(f"Filtered out {len(texts) - len(filtered)} short or empty texts (fewer than {min_tokens} tokens)")
            return filtered

        filtered_texts = filter_short_texts(texts)
        vocab = build_vocab(filtered_texts, max_vocab_size)
        dataset = MLMPretrainingDataset(filtered_texts, vocab, max_len=seq_len, mlm_prob=0.10)  # Reduced mlm_prob to 0.10
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=device.type == 'cuda', drop_last=True)  # drop_last to avoid partial batches
        logger.info(f"Prepared DataLoader with {len(dataset)} samples, batch_size={batch_size}, seq_len={seq_len}")
        return train_loader, vocab
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise

def train(model: FullEncoder, train_loader: DataLoader, num_epochs: int, learning_rate: float, device: torch.device, save_path: str) -> None:
    """
    Train the FullEncoder using MLM with FP16, minimal output, dynamic scheduling, and error monitoring.
    """
    try:
        criterion = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore padding (index 0)
        optimizer = optim.Adam(list(model.parameters()), lr=learning_rate)
        scaler = GradScaler('cuda' if device.type == 'cuda' else 'cpu')
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        logger.info(f"Starting training on device: {device_type}")

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", mininterval=1.0)
            for batch_idx, (inputs, targets) in enumerate(progress_bar):
                try:
                    # Validate input and target shapes
                    if inputs.size(0) != batch_size or inputs.size(1) != seq_len:
                        logger.warning(f"Unexpected input shape at batch {batch_idx}: {inputs.shape}. Expected ({batch_size}, {seq_len})")
                    if targets.size(0) != batch_size or targets.size(1) != seq_len:
                        logger.warning(f"Unexpected target shape at batch {batch_idx}: {targets.shape}. Expected ({batch_size}, {seq_len})")

                    inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                    optimizer.zero_grad()
                    
                    with autocast(device_type=device_type):
                        logits, _ = model(inputs)
                        # Validate logits shape
                        if logits.size(0) != batch_size or logits.size(1) != seq_len or logits.size(2) != model.token_emb.vocabulary_size:
                            logger.error(f"Unexpected logits shape at batch {batch_idx}: {logits.shape}. Expected ({batch_size}, {seq_len}, {model.token_emb.vocabulary_size})")
                            raise ValueError(f"Unexpected logits shape at batch {batch_idx}: {logits.shape}")

                        # Validate targets before loss calculation
                        valid_targets = targets[targets != -100]
                        if valid_targets.numel() > 0 and (valid_targets.min() < 0 or valid_targets.max() >= model.token_emb.vocabulary_size):
                            logger.error(f"Invalid target indices in batch {batch_idx} of epoch {epoch + 1}: min={valid_targets.min()}, max={valid_targets.max()}, vocab_size={model.token_emb.vocabulary_size}")
                            raise ValueError(f"Invalid target indices detected: min={valid_targets.min()}, max={valid_targets.max()}")

                        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                        # Validate loss
                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.error(f"NaN or Inf loss at batch {batch_idx} of epoch {epoch + 1}: {loss.item()}")
                            raise ValueError(f"NaN or Inf loss detected: {loss.item()}")
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    total_loss += loss.item()
                    progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{optimizer.param_groups[0]['lr']:.6f}"})
                except RuntimeError as e:
                    logger.error(f"Runtime error (CUDA or shape mismatch) in batch {batch_idx} of epoch {epoch + 1}: {str(e)}")
                    raise  # Re-raise to stop and investigate
                except Exception as e:
                    logger.error(f"Unexpected error in batch {batch_idx} of epoch {epoch + 1}: {str(e)}")
                    raise

            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
            print(f"Learning rate at epoch {epoch + 1}: {optimizer.param_groups[0]['lr']:.6f}")

            # Update scheduler based on loss plateau
            scheduler.step(avg_loss)

        # Save the final model with error handling
        try:
            torch.save({
                "encoder_state_dict": model.state_dict(),
                "vocab_size": model.token_emb.vocabulary_size,
                "d_model": model.token_emb.d_model,
                "vocab": vocab
            }, save_path)
            logger.info(f"Model saved to {save_path}")
            print(f"Model saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving model to {save_path}: {str(e)}")
            raise
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train FullEncoder with masked language modeling.")
    parser.add_argument("--wiki", default="data/cleaned_wiki_corpus.txt", help="Cleaned Wikipedia corpus (one entry per line).")
    parser.add_argument("--cot1", default="data/cot1.json", help="First CoT JSON file.")
    parser.add_argument("--cot2", default="data/cot2.json", help="Second CoT JSON file.")
    parser.add_argument("--out", default="checkpoints/pretrained_encoder.pth", help="Where to save the trained checkpoint.")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    try:
        # Define training parameters
        max_vocab_size = 30000  # Cap vocabulary at 30,000 tokens
        d_model = 256  # Embedding dimension
        batch_size = 128  # Batch size for training
        seq_len = 128  # Maximum sequence length
        num_epochs = 10 # Number of training epochs
        learning_rate = 0.001  # Initial learning rate
        save_path = args.out  # Path to save the model

        # Set device and check availability
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        if not torch.cuda.is_available():
            logger.warning("FP16 requires CUDA, falling back to FP32")
            print("FP16 requires CUDA, falling back to FP32")

        # Load and preprocess datasets
        cleaned_corpus = load_cleaned_corpus(args.wiki)
        cot_files = [args.cot1, args.cot2]  # Paths to CoT JSON files
        cot_texts = load_cot_files(cot_files)
        try:
            hf_texts = load_huggingface_corpora(os.environ.get("HF_TOKEN"))  # Use HF_TOKEN for Hugging Face authentication, excluding Wikipedia
        except Exception as e:
            logger.error(f"Failed to load Hugging Face datasets: {str(e)}. Continuing without HF data.")
            hf_texts = []  # Fallback to continue with local data

        # Combine datasets, remove duplicates, and filter short texts
        combined_corpus = list(dict.fromkeys(cleaned_corpus + cot_texts + hf_texts))  # Remove duplicates
        train_loader, vocab = prepare_data(batch_size, seq_len, device, combined_corpus)

        # Initialize and train the model
        model = FullEncoder(
            vocab_size=len(vocab),
            d_model=d_model,
            num_heads=4,
            d_ff=512,
            num_layers=6
        ).to(device)
        logger.info(f"Model initialized with vocab_size={len(vocab)}, d_model={d_model}")

        print(f"Model initialized with vocab_size={len(vocab)}, d_model={d_model}")
        train(model, train_loader, num_epochs, learning_rate, device, save_path)
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise