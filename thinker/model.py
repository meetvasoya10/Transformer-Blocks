"""Looped (recurrent-depth) decoder and a standard baseline.

LoopedThinker: embed -> prelude -> [adapter(cat(state, e)) -> shared core] x num_loops -> coda -> head.
StandardTransformer: embed -> n_layers distinct blocks -> head (no looping).

Batches are right-padded; attention is causal and loss/readout is taken at the "=" position,
so no attention mask is needed.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from thinker.tokenizer import VOCAB_SIZE


@dataclass
class ThinkerConfig:
    vocab_size: int = VOCAB_SIZE
    d_model: int = 256
    n_heads: int = 4
    ffn_hidden: int = 704
    n_prelude: int = 2
    n_core: int = 2
    n_coda: int = 2
    pos: str = "nope"  # "nope" or "rope"
    max_seq_len: int = 1024
    rope_base: float = 10000


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight


def _rope(x, base):
    """Rotate pairs (x[..., :hd/2], x[..., hd/2:]) by position-dependent angles. x: [B, H, T, hd]."""
    T, hd = x.shape[-2], x.shape[-1]
    inv_freq = 1.0 / (base ** (torch.arange(0, hd, 2, device=x.device, dtype=torch.float32) / hd))
    angles = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * inv_freq[None, :]
    cos, sin = angles.cos(), angles.sin()
    x1, x2 = x.float().chunk(2, dim=-1)
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.type_as(x)


class Block(nn.Module):
    """Pre-norm decoder block: RMSNorm -> causal attention -> residual, RMSNorm -> SwiGLU -> residual."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        assert cfg.pos in ("nope", "rope")
        self.n_heads = cfg.n_heads
        self.rope = cfg.pos == "rope"
        self.rope_base = cfg.rope_base
        self.attn_norm = RMSNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.attn_norm(x)).view(B, T, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        if self.rope:
            q, k = _rope(q, self.rope_base), _rope(k, self.rope_base)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        h = self.mlp_norm(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


def _init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)


class LoopedThinker(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or ThinkerConfig()
        c = self.cfg
        self.embed = nn.Embedding(c.vocab_size, c.d_model)
        self.prelude = nn.ModuleList(Block(c) for _ in range(c.n_prelude))
        self.adapter = nn.Linear(2 * c.d_model, c.d_model, bias=False)
        self.core = nn.ModuleList(Block(c) for _ in range(c.n_core))
        self.coda = nn.ModuleList(Block(c) for _ in range(c.n_coda))
        self.norm = RMSNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        _init_weights(self)
        self.lm_head.weight = self.embed.weight  # tie after init

    def _encode(self, input_ids):
        e = self.embed(input_ids)
        for blk in self.prelude:
            e = blk(e)
        return e

    def _loop(self, s, e):
        s = self.adapter(torch.cat([s, e], dim=-1))
        for blk in self.core:
            s = blk(s)
        return s

    def _decode(self, s):
        for blk in self.coda:
            s = blk(s)
        return self.lm_head(self.norm(s))

    def forward(self, input_ids, num_loops, backprop_loops=None, init_state=None, return_state=False):
        e = self._encode(input_ids)
        s = init_state if init_state is not None else torch.zeros_like(e)
        n_free = 0
        if backprop_loops is not None and backprop_loops < num_loops:
            n_free = num_loops - backprop_loops
        if n_free:
            with torch.no_grad():
                for _ in range(n_free):
                    s = self._loop(s, e)
        for _ in range(num_loops - n_free):
            s = self._loop(s, e)
        logits = self._decode(s)
        return (logits, s) if return_state else logits

    @torch.no_grad()
    def readout_per_loop(self, input_ids, answer_pos, num_loops):
        """Logits at answer_pos after each loop: [num_loops, B, vocab]."""
        e = self._encode(input_ids)
        s = torch.zeros_like(e)
        idx = torch.arange(input_ids.shape[0], device=input_ids.device)
        out = []
        for _ in range(num_loops):
            s = self._loop(s, e)
            out.append(self._decode(s)[idx, answer_pos])
        return torch.stack(out)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class StandardTransformer(nn.Module):
    def __init__(self, cfg=None, n_layers=6):
        super().__init__()
        self.cfg = cfg or ThinkerConfig()
        c = self.cfg
        self.embed = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(n_layers))
        self.norm = RMSNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        _init_weights(self)
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.norm(x))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
