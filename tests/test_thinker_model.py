import os
import random
import sys

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT)
from thinker.model import LoopedThinker, StandardTransformer, ThinkerConfig
from thinker.tasks import generate
from thinker.tokenizer import PAD_ID, VOCAB_SIZE, tokenizer

B, T = 3, 12


def small_cfg(pos="nope"):
    return ThinkerConfig(d_model=32, n_heads=2, ffn_hidden=64, n_prelude=1, n_core=2, n_coda=1, pos=pos, max_seq_len=64)


def make(kind, pos="nope"):
    torch.manual_seed(0)
    cfg = small_cfg(pos)
    model = LoopedThinker(cfg) if kind == "looped" else StandardTransformer(cfg, n_layers=3)
    return model.eval()


def run(model, ids):
    return model(ids, num_loops=3) if isinstance(model, LoopedThinker) else model(ids)


def ids_batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, VOCAB_SIZE, (B, T), generator=g)


@pytest.mark.parametrize("loops", [1, 4])
def test_looped_shapes(loops):
    out = make("looped")(ids_batch(), num_loops=loops)
    assert out.shape == (B, T, VOCAB_SIZE)


def test_standard_shapes():
    assert make("standard")(ids_batch()).shape == (B, T, VOCAB_SIZE)


@pytest.mark.parametrize("pos", ["nope", "rope"])
@pytest.mark.parametrize("kind", ["looped", "standard"])
def test_no_peeking(kind, pos):
    model = make(kind, pos)
    a = ids_batch(1)
    t = 5
    b = a.clone()
    b[:, t + 1:] = torch.randint(3, VOCAB_SIZE, b[:, t + 1:].shape)
    assert not torch.equal(a, b)
    with torch.no_grad():
        la, lb = run(model, a), run(model, b)
    assert torch.allclose(la[:, :t + 1], lb[:, :t + 1], atol=1e-5)
    assert not torch.allclose(la[:, t + 1:], lb[:, t + 1:], atol=1e-5)


def test_weight_sharing():
    cfg = small_cfg()
    model = LoopedThinker(cfg)
    core_keys = {k.split(".")[1] for k in model.state_dict() if k.startswith("core.")}
    assert core_keys == {str(i) for i in range(cfg.n_core)}
    n1 = model.num_params()
    with torch.no_grad():
        model(ids_batch(), num_loops=1)
        model(ids_batch(), num_loops=16)
    assert model.num_params() == n1
    assert model.lm_head.weight is model.embed.weight


def test_truncation_only_changes_gradients():
    model = make("looped")
    ids = ids_batch(2)
    model.zero_grad()
    full = model(ids, num_loops=10)
    trunc = model(ids, num_loops=10, backprop_loops=3)
    assert torch.allclose(full, trunc, atol=1e-6)
    trunc.logsumexp(-1).sum().backward()
    for name, p in model.core.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


def test_truncation_grads_differ_from_full():
    model = make("looped")
    ids = ids_batch(2)
    grads = []
    for bp in (None, 3):
        model.zero_grad()
        model(ids, num_loops=10, backprop_loops=bp).logsumexp(-1).sum().backward()
        grads.append(model.core[0].qkv.weight.grad.clone())
    assert not torch.allclose(grads[0], grads[1])


def test_resume_from_state():
    model = make("looped")
    ids = ids_batch(3)
    with torch.no_grad():
        full = model(ids, num_loops=6)
        _, s = model(ids, num_loops=3, return_state=True)
        resumed = model(ids, num_loops=3, init_state=s)
    assert torch.allclose(full, resumed, atol=1e-5)


def test_readout_per_loop():
    model = make("looped")
    ids = ids_batch(4)
    pos = torch.tensor([2, 7, 11])
    with torch.no_grad():
        per_loop = model.readout_per_loop(ids, pos, num_loops=5)
        full = model(ids, num_loops=5)
        one = model(ids, num_loops=1)
    assert per_loop.shape == (5, B, VOCAB_SIZE)
    idx = torch.arange(B)
    assert torch.allclose(per_loop[-1], full[idx, pos], atol=1e-5)
    assert torch.allclose(per_loop[0], one[idx, pos], atol=1e-5)


@pytest.mark.parametrize("pos", ["nope", "rope"])
def test_eval_deterministic(pos):
    model = make("looped", pos)
    ids = ids_batch(5)
    with torch.no_grad():
        assert torch.equal(model(ids, num_loops=4), model(ids, num_loops=4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_bf16_smoke():
    torch.manual_seed(0)
    rng = random.Random(0)
    prompts = [tokenizer.encode(generate("chain", 8, rng)["prompt"]) for _ in range(8)]
    maxlen = max(map(len, prompts))
    ids = torch.full((8, maxlen), PAD_ID, dtype=torch.long)
    pos = torch.zeros(8, dtype=torch.long)
    for i, p in enumerate(prompts):
        ids[i, :len(p)] = torch.tensor(p)
        pos[i] = len(p) - 1  # the "=" position
    ids, pos = ids.cuda(), pos.cuda()
    model = LoopedThinker().cuda().train()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(ids, num_loops=12, backprop_loops=4)
    target = torch.randint(0, 7, (8,), device="cuda") + 3
    loss = torch.nn.functional.cross_entropy(logits[torch.arange(8), pos].float(), target)
    loss.backward()
    assert torch.isfinite(loss)
    peak = torch.cuda.max_memory_allocated()
    print(f"\nPEAK_MEM_BYTES={peak} ({peak / 2**20:.1f} MiB), seq_len={maxlen}")
