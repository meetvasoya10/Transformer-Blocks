"""Independent checkers: recompute the answer with real Python, not the generator's arithmetic."""

import contextlib
import io

from .tasks import MOD


def check_expr(prompt, answer):
    try:
        body = prompt
        if not (body.startswith("E:") and body.endswith("=")):
            return False
        body = body[2:-1]
        return eval(body, {"__builtins__": {}}, {}) % MOD == answer
    except Exception:
        return False


def check_chain(prompt, answer):
    try:
        if not prompt.endswith("="):
            return False
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(prompt[:-1], {"__builtins__": {"print": print}}, {})
        return int(out.getvalue().strip()) == answer
    except Exception:
        return False


def check(example):
    fn = check_expr if example["task"] == "expr" else check_chain
    return fn(example["prompt"], example["answer"])
