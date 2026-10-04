import json
import keyword
import os
import random
import re
import subprocess
import sys
from collections import Counter

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT)
from thinker.tokenizer import tokenizer
from thinker.tasks import TASKS, generate
from thinker.check import check, check_expr, check_chain

TASK_NAMES = list(TASKS)
IDENT = re.compile(r"[a-z]{2,}")


def gen(task, difficulty, seed, n):
    rng = random.Random(seed)
    return [generate(task, difficulty, rng) for _ in range(n)]


def test_tokenizer_roundtrip_and_unknown_char():
    prompts = []
    for task in TASK_NAMES:
        for d in (1, 2, 4, 8, 16):
            prompts += [e["prompt"] for e in gen(task, d, 1, 100)]
    assert len(prompts) == 1000
    for p in prompts:
        assert tokenizer.decode(tokenizer.encode(p)) == p
    for bad in ("A", "é", " ", "x y", "E:(1+1)=!"):
        with pytest.raises(ValueError):
            tokenizer.encode(bad)
    assert tokenizer.encode("0a+\n")[0] == 3  # <pad>,<bos>,<eos> come first


@pytest.mark.parametrize("task", TASK_NAMES)
def test_same_seed_same_examples(task):
    assert gen(task, 8, 123, 50) == gen(task, 8, 123, 50)
    assert gen(task, 8, 123, 50) != gen(task, 8, 124, 50)


@pytest.mark.parametrize("task", TASK_NAMES)
@pytest.mark.parametrize("difficulty", [1, 4, 8, 16, 32])
def test_checkers_agree(task, difficulty):
    for e in gen(task, difficulty, 7, 500):
        assert 0 <= e["answer"] <= 6
        assert check(e), e["prompt"]
        assert check({**e, "answer": (e["answer"] + 1) % 7}) is False
        assert e["steps"][-1] == e["answer"]


def paren_depth(s):
    depth = best = 0
    for ch in s:
        if ch == "(":
            depth += 1
            best = max(best, depth)
        elif ch == ")":
            depth -= 1
    assert depth == 0
    return best


@pytest.mark.parametrize("difficulty", [1, 2, 3, 8, 16, 32])
def test_expr_depth_equals_difficulty(difficulty):
    for e in gen("expr", difficulty, 3, 200):
        assert paren_depth(e["prompt"]) == difficulty


def parse_chain(e):
    lines = e["prompt"][:-1].strip("\n").split("\n")
    assert lines[-1] == f"print({e['chain_vars'][-1]})"
    assigns = {}
    order = []
    for line in lines[:-1]:
        name, rhs = line.split("=")
        assert name not in assigns
        # every variable read must already be defined (valid execution order)
        assert all(v in assigns for v in IDENT.findall(rhs)), line
        assigns[name] = rhs
        order.append(name)
    return order, assigns


@pytest.mark.parametrize("difficulty", [1, 2, 5, 8, 16, 32])
def test_chain_structure(difficulty):
    for e in gen("chain", difficulty, 5, 100):
        order, assigns = parse_chain(e)
        root, chain = e["root_var"], e["chain_vars"]
        assert order[0] == root
        assert len(chain) == difficulty
        for name in order:
            assert len(name) == 2 and name.islower() and not keyword.iskeyword(name)
        distractors = set(order) - set(chain) - {root}
        assert len(distractors) == difficulty
        prev = root
        for name in chain:
            assert IDENT.findall(assigns[name]) == [prev]
            prev = name
        # chain lines appear in order, and no chain line reads a distractor
        assert [n for n in order if n in chain] == chain
        for name in chain:
            assert not set(IDENT.findall(assigns[name])) & distractors
        assert len(e["steps"]) == difficulty


def _result(check_fn, prompt):
    """The value the checker's own evaluation produces for `prompt` (the one answer it accepts)."""
    hits = [r for r in range(7) if check_fn(prompt, r)]
    assert len(hits) == 1, prompt
    return hits[0]


@pytest.mark.parametrize("difficulty", [16, 32])
def test_expr_sensitive_to_deepest_subexpression(difficulty):
    for e in gen("expr", difficulty, 21, 200):
        start, end = e["deep_span"]
        assert e["prompt"][start:end] in set("0123456")
        results = {
            _result(check_expr, e["prompt"][:start] + str(digit) + e["prompt"][end:])
            for digit in range(7)
        }
        assert len(results) == 7, e["prompt"]


@pytest.mark.parametrize("difficulty", [16, 32])
def test_chain_sensitive_to_root_constant(difficulty):
    for e in gen("chain", difficulty, 22, 200):
        first, rest = e["prompt"].split("\n", 1)
        assert first == f"{e['root_var']}={first[-1]}"
        results = {
            _result(check_chain, f"{first[:-1]}{digit}\n{rest}") for digit in range(7)
        }
        assert len(results) == 7, e["prompt"]


@pytest.mark.parametrize("task", TASK_NAMES)
def test_answer_distribution(task):
    counts = Counter(e["answer"] for e in gen(task, 6, 11, 7000))
    for digit in range(7):
        assert 0.11 <= counts[digit] / 7000 <= 0.18, (digit, counts)


def test_make_evalsets_is_reproducible(tmp_path):
    script = os.path.join(ROOT, "scripts", "make_evalsets.py")
    runs = []
    for i in range(2):
        manifest = tmp_path / f"manifest{i}.json"
        subprocess.run(
            [sys.executable, script, "--out-dir", str(tmp_path / f"data{i}"), "--manifest", str(manifest)],
            check=True, capture_output=True,
        )
        runs.append(json.loads(manifest.read_text()))
    assert runs[0] == runs[1]
    assert len(runs[0]) == 2 * 32
    assert all(len(e["sha256"]) == 64 and e["count"] == 200 for e in runs[0])
