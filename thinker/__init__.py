"""Synthetic reasoning data for the looped-model experiments (mod-7 arithmetic and programs)."""

from .tokenizer import CharTokenizer, tokenizer, VOCAB_SIZE, PAD_ID, BOS_ID, EOS_ID
from .tasks import TASKS, MOD, expr_example, chain_example, generate
from .check import check, check_expr, check_chain

__all__ = [
    "CharTokenizer", "tokenizer", "VOCAB_SIZE", "PAD_ID", "BOS_ID", "EOS_ID",
    "TASKS", "MOD", "expr_example", "chain_example", "generate",
    "check", "check_expr", "check_chain",
]
