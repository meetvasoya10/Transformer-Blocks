"""Synthetic generators. All arithmetic is mod 7; the answer is a single digit 0-6.

Both generators take (difficulty, rng) and return
{task, difficulty, prompt, answer, steps} (expr also returns deep_span, chain also returns
root_var and chain_vars).
Answers are made uniform over 0-6 by rejection sampling: a target digit is drawn
first and candidates are regenerated until they evaluate to it.
"""

import random

MOD = 7
EXPR_OPS = "+-*"
CHAIN_OPS = "+-*"
KEYWORDS = {"as", "if", "in", "is", "or"}  # two-letter Python keywords
MAX_ATTEMPTS = 100_000


def _apply(op, a, b):
    if op == "+":
        return (a + b) % MOD
    if op == "-":
        return (a - b) % MOD
    return (a * b) % MOD


# ---------------------------------------------------------------- (A) expr

class _Deep(int):
    """The constant at the bottom of the spine (the one a prompt's answer ultimately depends on)."""


def _node_value(node):
    if isinstance(node, int):
        return node
    op, l, r = node
    return _apply(op, _node_value(l), _node_value(r))


def _expr_side(depth, rng):
    """Shallow branch: a constant, or (when allowed) a depth-1 expression."""
    if depth >= 2 and rng.random() < 0.5:
        return (rng.choice(EXPR_OPS), rng.randrange(MOD), rng.randrange(MOD))
    return rng.randrange(MOD)


def _expr_tree(depth, rng):
    """Tree of exactly `depth` levels: a constant is an int, a node is (op, left, right).

    The spine (deepest path) ends in a _Deep constant. Under `*` the shallow branch is
    resampled until it is nonzero mod 7, so every spine node is a bijection of the deep value.
    """
    if depth == 0:
        return _Deep(rng.randrange(MOD))
    deep = _expr_tree(depth - 1, rng)
    op = rng.choice(EXPR_OPS)
    side = _expr_side(depth, rng)
    if op == "*":
        while _node_value(side) == 0:
            side = _expr_side(depth, rng)
    left, right = (deep, side) if rng.random() < 0.5 else (side, deep)
    return (op, left, right)


def _expr_render_eval(node, steps):
    """Return (text, value, deep_span); append each operator node's value to steps in evaluation order.

    deep_span is the (start, end) slice of text holding the _Deep constant, or None.
    """
    if isinstance(node, int):
        text = str(node)
        return text, int(node), ((0, len(text)) if isinstance(node, _Deep) else None)
    op, l, r = node
    ltxt, lval, lspan = _expr_render_eval(l, steps)
    rtxt, rval, rspan = _expr_render_eval(r, steps)
    val = _apply(op, lval, rval)
    steps.append(val)
    if lspan is not None:
        span = (lspan[0] + 1, lspan[1] + 1)
    elif rspan is not None:
        shift = 1 + len(ltxt) + 1
        span = (rspan[0] + shift, rspan[1] + shift)
    else:
        span = None
    return f"({ltxt}{op}{rtxt})", val, span


def expr_example(difficulty, rng):
    target = rng.randrange(MOD)
    for _ in range(MAX_ATTEMPTS):
        steps = []
        text, value, span = _expr_render_eval(_expr_tree(difficulty, rng), steps)
        if value == target:
            offset = len("E:")
            return {
                "task": "expr",
                "difficulty": difficulty,
                "prompt": f"E:{text}=",
                "answer": value,
                "steps": steps,
                "deep_span": (span[0] + offset, span[1] + offset),
            }
    raise RuntimeError("expr rejection sampling did not converge")


# --------------------------------------------------------------- (B) chain

def _chain_program(difficulty, rng):
    names = rng.sample(
        [a + b for a in "abcdefghijklmnopqrstuvwxyz" for b in "abcdefghijklmnopqrstuvwxyz"
         if a + b not in KEYWORDS],
        1 + 2 * difficulty,
    )
    root, chain_names, distractor_names = names[0], names[1:1 + difficulty], names[1 + difficulty:]

    # interleave: choose which of the 2*difficulty slots hold chain lines (order within kind kept)
    kinds = ["c"] * difficulty + ["d"] * difficulty
    rng.shuffle(kinds)

    root_val = rng.randrange(MOD)
    lines = [f"{root}={root_val}"]
    values = {root: root_val}
    chain_vals = []
    prev = root
    ci = di = 0
    for kind in kinds:
        op, c = rng.choice(CHAIN_OPS), rng.randrange(1, MOD)
        if kind == "c":
            name, src = chain_names[ci], prev
            ci += 1
        else:
            name, src = distractor_names[di], rng.choice(list(values))  # any earlier variable
            di += 1
        values[name] = _apply(op, values[src], c)
        lines.append(f"{name}=({src}{op}{c})%{MOD}")
        if kind == "c":
            prev = name
            chain_vals.append(values[name])
    lines.append(f"print({prev})")
    return root, chain_names, "\n".join(lines) + "\n", chain_vals


def chain_example(difficulty, rng):
    target = rng.randrange(MOD)
    for _ in range(MAX_ATTEMPTS):
        root, chain_vars, program, chain_vals = _chain_program(difficulty, rng)
        if chain_vals[-1] == target:
            return {
                "task": "chain",
                "difficulty": difficulty,
                "prompt": program + "=",
                "answer": chain_vals[-1],
                "steps": chain_vals,
                "root_var": root,
                "chain_vars": chain_vars,
            }
    raise RuntimeError("chain rejection sampling did not converge")


TASKS = {"expr": expr_example, "chain": chain_example}


def generate(task, difficulty, rng):
    return TASKS[task](difficulty, rng)
