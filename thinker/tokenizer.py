"""Fixed character-level tokenizer.

Ids: <pad>=0, <bos>=1, <eos>=2, then 0-9, a-z, + - * % ( ) = : newline.
The capital letter E (the expression-task prefix "E:") is appended last so every
id above stays where the spec puts it.
"""

import string

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]
CHARS = list(string.digits) + list(string.ascii_lowercase) + list("+-*%()=:") + ["\n"] + ["E"]

PAD_ID, BOS_ID, EOS_ID = 0, 1, 2


class CharTokenizer:
    def __init__(self):
        self.itos = SPECIAL_TOKENS + CHARS
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}
        self.vocab_size = len(self.itos)

    def encode(self, text):
        ids = []
        for ch in text:
            if ch not in self.stoi or ch in SPECIAL_TOKENS:
                raise ValueError(f"unknown character {ch!r}")
            ids.append(self.stoi[ch])
        return ids

    def decode(self, ids):
        """Special tokens (<pad>, <bos>, <eos>) are dropped from the output."""
        out = []
        for i in ids:
            if not 0 <= i < self.vocab_size:
                raise ValueError(f"unknown token id {i}")
            if i >= len(SPECIAL_TOKENS):
                out.append(self.itos[i])
        return "".join(out)


tokenizer = CharTokenizer()
VOCAB_SIZE = tokenizer.vocab_size
